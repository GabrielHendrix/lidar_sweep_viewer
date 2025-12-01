import os
import cv2
import argparse
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import womd_lidar_utils
from waymo_open_dataset.protos import compressed_lidar_pb2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _load_scenario_data(tfrecord_file: str) -> scenario_pb2.Scenario:
    """Load a scenario proto from a tfrecord dataset file."""
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    data = next(iter(dataset))
    return scenario_pb2.Scenario.FromString(data.numpy())


def normalizar(image, meters):
    lines = (len(image))
    columns = (len(image[0]))
    resolution = ((meters * 100) / 256)
    for x in range(lines):
        for y in range(columns):
            pixel = image[x, y] 
            if (pixel < 0.0):
                image[x, y] = 0
            else:
                image[x, y] = (255 - ((pixel * 100) / resolution))

    return np.clip(image, 0, 255).astype(np.uint8)

def _get_laser_calib(
    frame_lasers: compressed_lidar_pb2.CompressedFrameLaserData,
    laser_name: dataset_pb2.LaserName.Name):
    for laser_calib in frame_lasers.laser_calibrations:
        if laser_calib.name == laser_name:
            return laser_calib
    return None

def process(filepath):
    WOMD_FILE = filepath
    womd_original_scenario = _load_scenario_data(WOMD_FILE)
    print(womd_original_scenario.scenario_id)

    # The corresponding compressed laser data file has the name
    # {scenario_id}.tfrecord. For simplicity, we rename the corresponding laser data
    # file 'ee519cf571686d19.tfrecord' to be 'womd_lidar_data.tfrecord'.
    LIDAR_DATA_FILE = filepath
    womd_lidar_scenario = _load_scenario_data(LIDAR_DATA_FILE)
    scenario_augmented = womd_lidar_utils.augment_womd_scenario_with_lidar_points(
        womd_original_scenario, womd_lidar_scenario)

    frame_points_xyz = {}  # map from frame indices to point clouds
    frame_points_feature = {}
    frame_i = 0

    # Extract point cloud xyz and features from each LiDAR and merge them for each
    # laser frame in the scenario proto.
    for frame_lasers in scenario_augmented.compressed_frame_laser_data:
        len(scenario_augmented.compressed_frame_laser_data)
        frame_pose = np.reshape(np.array(
            scenario_augmented.compressed_frame_laser_data[frame_i].pose.transform),
            (4, 4))
        for laser in frame_lasers.lasers:
            if laser.name == dataset_pb2.LaserName.TOP:
                c = _get_laser_calib(frame_lasers, laser.name)
                (points_xyz, points_feature,
                points_xyz_return2,
                points_feature_return2) = womd_lidar_utils.extract_top_lidar_points(
                    laser, frame_pose, c)
            #       print(points_feature)
                # print(points_feature[:,0])
                range_values_1 = points_feature[:, 0]      
                range_values_2 = points_feature_return2[:, 0]      
                shape = np.array([64, 2650, 1])
                # Converter o tensor para um array NumPy e imprimir
                range_array_1 = range_values_1.numpy()
                range_image_1 = np.array(range_array_1).reshape(shape)
            
                imagem_normalizada_1 = normalizar(range_image_1, 75)  # Normaliza a imagem

                imagem_redimensionada_1 = cv2.resize(imagem_normalizada_1, (1920, 128), 
                            interpolation = cv2.INTER_CUBIC)

                images = []
                
                images.append(imagem_redimensionada_1)

                # Adicionando espaçamento entre as imagens
                spacer_height = 10  # Tamanho do espaçamento entre as imagens
                spacer = np.ones((spacer_height, shape[1]), dtype=np.uint8) * 255  # Um espaço branco (255)

                # Concatenar todas as imagens verticalmente
                final_image = images[0]


                # final_image = cv2.addWeighted(imagem_redimensionada_2, 1.0, imagem_redimensionada_1, 1.0, 0)
                # Exibir a imagem usando OpenCV
                cv2.imshow('LiDAR', final_image)
                cv2.waitKey(1)  # Aguarda até que uma tecla seja pressionada para fechar a janela
                break
            
        frame_i += 1

if __name__ == "__main__":
    # Parse o argumento da linha de comando para receber o caminho do vídeo
    parser = argparse.ArgumentParser(description='Object Detection and Tracking in Video')
    parser.add_argument('-p', '--path', type=str, help='Caminho do vídeo para inferência', required=True)
    args = parser.parse_args()
    # path = '/media/lume/teste/datasets/uncompressed/lidar/testing'


    sorted_files = sorted([os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.tfrecord')])
    for file in sorted_files:
        print(f"Processando {file}...")
        process(file)