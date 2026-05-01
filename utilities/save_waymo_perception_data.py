import os
import glob
import argparse
import numpy as np
import tensorflow as tf

# Imports do Waymo
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import frame_utils, box_utils

# Configurações do TensorFlow
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def make_sure_path_exists(caminho):
    os.makedirs(caminho, exist_ok=True)

def extract_data_to_numpy(obj):
    """
    Converte qualquer objeto Waymo (Tensor ou MatrixFloat) para Numpy de forma robusta.
    """
    if hasattr(obj, 'numpy'):
        return obj.numpy()
    
    if hasattr(obj, 'shape') and hasattr(obj, 'data'):
        try:
            real_shape = list(obj.shape.dims)
        except AttributeError:
            real_shape = list(obj.shape)
        data_flat = np.array(obj.data)
        return data_flat.reshape(real_shape)
        
    return np.array(obj)

def save_data_to_txt(data_array, filename):
    with open(filename, "w") as file:
        if data_array.ndim == 2:
            for row in data_array:
                line = " ".join([f"{val:.6f}" for val in row])
                file.write(f"{line}\n")
        else:
            line = " ".join([f"{val:.6f}" for val in data_array])
            file.write(f"{line}\n")

def save_lidar_points_to_bin(lidar_points, bin_path):
    # Salva no formato [x, y, z, intensity] como float32 (Padrão KITTI/Waymo)
    data_flat = lidar_points.flatten().astype(np.float32)
    with open(bin_path, "wb") as fs:
        fs.write(data_flat.tobytes())

def calculate_box_corners(box_label):
    box_tensor = tf.constant([
        [box_label.center_x, box_label.center_y, box_label.center_z,
         box_label.length, box_label.width, box_label.height, box_label.heading]
    ], dtype=tf.float32)  # Shape [1, 7]

    # Use get_upright_3d_box_corners to find the corners
    corners = box_utils.get_upright_3d_box_corners(box_tensor)

    # Convert corners to NumPy for further processing
    corners_np = corners.numpy()  # Shape [1, 8, 3]
    return corners_np[0]  # Return the 8 corners as a NumPy array

def extract_lidar_using_polar_features(frame):
    """
    Extrai XYZ e Intensidade usando keep_polar_features=True.
    Isso retorna um tensor [H, W, 6] onde:
      0: Range
      1: Intensity
      2: Elongation
      3, 4, 5: X, Y, Zz
    """
    # 1. Parsear dados básicos
    (range_images, camera_projections, _, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    # 2. Converter para Cartesiano COM features polares
    # O parametro keep_polar_features=True faz a mágica de concatenar tudo
    combined_cartesian = frame_utils.convert_range_image_to_cartesian(
        frame, 
        range_images, 
        range_image_top_pose, 
        ri_index=0, 
        keep_polar_features=True)

    # 3. Pegar o tensor do LiDAR TOP
    # Shape esperado: [Altura, Largura, 6]
    lidar_tensor = combined_cartesian[dataset_pb2.LaserName.TOP]
    
    # 4. Converter para Numpy
    lidar_grid = extract_data_to_numpy(lidar_tensor)

    # 5. Extração Inteligente
    # Canal 0 é o Range. Usamos ele para criar a máscara de validade.
    range_val = lidar_grid[..., 0]
    mask = range_val != -2  # Filtra pontos sem retorno (céu/distante)

    # Canal 1 é Intensidade
    intensity_val = lidar_grid[..., 0]
    
    # Canais 3, 4, 5 são X, Y, Z
    xyz_val = lidar_grid[..., 3:6]

    # 6. Aplicação da Máscara (Flattening automático)
    points_xyz = xyz_val[mask]             # [N, 3]
    points_intensity = intensity_val[mask] # [N]

    # 7. Junção Final [N, 4] -> (x, y, z, intensity)
    lidar_data = np.hstack((points_xyz, points_intensity[:, np.newaxis]))

    return lidar_data

def process_segment(tfrecord_path, output_root_path):
    segment_name = os.path.splitext(os.path.basename(tfrecord_path))[0]
    print(f"Processando: {segment_name}")

    # Estrutura de pastas
    objs_path = os.path.join(output_root_path, "objs_bbox", segment_name)
    poses_path = os.path.join(output_root_path, "poses", segment_name)
    bin_path = os.path.join(output_root_path, "bin_files", segment_name)

    make_sure_path_exists(objs_path)
    make_sure_path_exists(poses_path)
    make_sure_path_exists(bin_path)

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

    for index, data in enumerate(dataset):
        try:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            timestamp_str = str(frame.timestamp_micros)
            # --- 1. Salvar Pose ---
            frame_pose = np.array(frame.pose.transform).reshape(4, 4)
            save_data_to_txt(frame_pose, os.path.join(poses_path, f"{timestamp_str}.txt"))

            # --- 2. Salvar LiDAR (Método Smart) ---
            lidar_points = extract_lidar_using_polar_features(frame)
            save_lidar_points_to_bin(lidar_points, os.path.join(bin_path, f"{timestamp_str}.bin"))

            # --- 3. Salvar Labels (Bounding Boxes) ---
            frame_objs_path = os.path.join(objs_path, str(timestamp_str))
            make_sure_path_exists(frame_objs_path)
            
            track_count = 0
            for label in frame.laser_labels:
                if label.type != label_pb2.Label.Type.TYPE_UNKNOWN:
                    vertices = calculate_box_corners(label.box)
                    save_data_to_txt(vertices, os.path.join(frame_objs_path, f"{track_count}.txt"))
                    track_count += 1

            # Log de progresso a cada 20 frames
            if index % 20 == 0:
                print(f"  -> Frame {index}: {len(lidar_points)} pontos. (360 completo)")
                
        except Exception as e:
            print(f"ERRO no Frame {index}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Waymo Extractor (Polar Features)')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Pasta com .tfrecords')
    parser.add_argument('-o', '--output_dir', type=str, default="waymo_output", help='Pasta de saída')
    args = parser.parse_args()

    tf_files = glob.glob(os.path.join(args.input_dir, "*.tfrecord"))
    
    if tf_files:
        print(f"Encontrados {len(tf_files)} segmentos.")
        for tf_file in tf_files:
            process_segment(tf_file, args.output_dir)
    else:
        print(f"Nenhum arquivo encontrado em {args.input_dir}")