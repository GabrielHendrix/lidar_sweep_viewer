import os
import cv2
import math
import errno
import struct
import argparse
import numpy as np
import open3d as o3d
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import womd_lidar_utils, box_utils
from waymo_open_dataset.protos import compressed_lidar_pb2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _load_scenario_data(tfrecord_file: str) -> scenario_pb2.Scenario:
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    data = next(iter(dataset))
    return scenario_pb2.Scenario.FromString(data.numpy())


def _get_laser_calib(frame_lasers: compressed_lidar_pb2.CompressedFrameLaserData, laser_name: dataset_pb2.LaserName.Name):
    for laser_calib in frame_lasers.laser_calibrations:
        if laser_calib.name == laser_name:
            return laser_calib
    return None

def calculate_bounding_box_vertices(track_state):
    # Create a tensor representing the bounding box [center_x, center_y, center_z, length, width, height, heading]
    box_tensor = tf.constant([
        [track_state.center_x, track_state.center_y, track_state.center_z,
         track_state.length, track_state.width, track_state.height, track_state.heading]
    ], dtype=tf.float32)  # Shape [1, 7]

    # Use get_upright_3d_box_corners to find the corners
    corners = box_utils.get_upright_3d_box_corners(box_tensor)

    # Convert corners to NumPy for further processing
    corners_np = corners.numpy()  # Shape [1, 8, 3]
    return corners_np[0]  # Return the 8 corners as a NumPy array


def visualize_3d_points(points_xyz, points_colors, max_distance=80):
    """
    Visualiza uma nuvem de pontos 3D centralizada na origem com limite de distância.
    Args:
        points_xyz (np.ndarray): Array de pontos 3D no formato (N, 3).
        max_distance (float): Distância máxima permitida para os pontos.
    """

    
    # Calcular a distância euclidiana de cada ponto em relação à origem
    # A conversão para numpy é necessária aqui
    distances = np.linalg.norm(points_xyz, axis=1)
    
    # Filtrar pontos que estão dentro do limite de distância
    points_within_limit = points_xyz[distances <= max_distance]
    
    # Criar uma nuvem de pontos
    point_cloud = o3d.geometry.PointCloud()
    
    # Atribuir os pontos filtrados à nuvem
    point_cloud.points = o3d.utility.Vector3dVector(points_within_limit)
    point_cloud.colors = o3d.utility.Vector3dVector(points_colors)
    
    # Adicionar cores (opcional)
    colors = np.zeros_like(points_within_limit)  # Padrão: preto
    colors[:, 2] = 1.0  # Azul
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualizar
    o3d.visualization.draw_geometries([point_cloud])


def create_bounding_box_lines(vertices):
    """
    Cria um LineSet (cubo 3D) com base nos vértices de um bounding box.

    Args:
        vertices (np.ndarray): Vértices do bounding box (8x3).

    Returns:
        o3d.geometry.LineSet: Estrutura 3D do bounding box como um LineSet.
    """
    # Índices dos pontos que formam as arestas do cubo
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Base inferior
        [4, 5], [5, 6], [6, 7], [7, 4],  # Base superior
        [0, 4], [1, 5], [2, 6], [3, 7]   # Conexões entre bases
    ]

    # Definir as cores para as arestas (vermelho neste caso)
    colors = [[1.0, 0.0, 0.0] for _ in lines]

    # Criar o LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)  # Vértices
    line_set.lines = o3d.utility.Vector2iVector(lines)      # Arestas
    line_set.colors = o3d.utility.Vector3dVector(colors)    # Cor das arestas

    return line_set


def transform_vertices(vertices, lidar_pose):
    """
    Transforma os vértices do bounding box para o sistema de coordenadas do LIDAR.

    Args:
        vertices (np.ndarray): Vértices do bounding box (8x3).
        lidar_pose (np.ndarray): Matriz de pose do LIDAR (4x4).

    Returns:
        np.ndarray: Vértices transformados no sistema de coordenadas do LIDAR (8x3).
    """
    # Converter os vértices para coordenadas homogêneas
    vertices_homogeneas = np.hstack([vertices, np.ones((vertices.shape[0], 1))])  # (8x4)

    # Calcular a matriz inversa da pose do LIDAR
    lidar_pose_inversa = np.linalg.inv(lidar_pose)

    # Transformar os vértices para o sistema de coordenadas do LIDAR
    vertices_transformados = (lidar_pose_inversa @ vertices_homogeneas.T).T  # (8x4)

    # Retornar apenas as coordenadas x, y, z
    return vertices_transformados[:, :3]


def transformar_para_sistema_lidar_topo(track_coords, lidar_pose):
    track_coords_homogeneas = np.append(track_coords, 1.0)
    lidar_pose_inversa = np.linalg.inv(lidar_pose)
    track_coords_transformadas = np.dot(lidar_pose_inversa, track_coords_homogeneas)
    return track_coords_transformadas[:3]


# def normalizar(image, meters):
def normalizar(image, colors, meters):
    # for color in colors:
    #     print(color)
    lines, columns, _ = image.shape
    resolution = ((meters * 100) / 256)
    image_normalized = np.zeros((lines, columns), dtype=np.uint8)

    for x in range(lines):
        for y in range(columns):
            pixel = image[x, y]
            if pixel < 0.0:
                image_normalized[x, y] = 0
            else:
                image_normalized[x, y] = (255 - int((pixel * 100) / resolution))

    image_color = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
    mask = image <= -2.0
    for x in range(lines):
        for y in range(columns):
            if np.array_equal(colors[x, y], [1.0, 0.0, 0.0]):  # Verificar se é igual ao vermelho
                image_color[x, y] = [0, 0, 255]
            elif mask[x, y]:
                image_color[x, y] = np.array([0, 0, 255])
    return image_color

# Agora, vamos encontrar as correspondências na nuvem de pontos para as linhas (como os pontos mais próximos)
def find_nearest_points_on_lines(pcd, line_set, threshold=0.05):
    points = np.asarray(pcd.points)
    lines = np.asarray(line_set.lines)

    correspondences = []
    for line in lines:
        p1 = points[line[0]]
        p2 = points[line[1]]

        # Encontrar os pontos mais próximos da linha (usando distância mínima)
        for i, point in enumerate(points):
            # Calcular a distância de cada ponto da linha (p1 -> p2)
            line_vec = p2 - p1
            point_vec = point - p1
            proj_len = np.dot(point_vec, line_vec) / np.linalg.norm(line_vec)
            proj_len = np.clip(proj_len, 0, np.linalg.norm(line_vec))  # Projetar na linha
            closest_point = p1 + (proj_len * line_vec) / np.linalg.norm(line_vec)
            distance = np.linalg.norm(point - closest_point)

            # Se a distância for menor que o threshold, considerar como correspondência
            if distance < threshold:
                correspondences.append((i, line[0], line[1], distance))

    return correspondences

def point_in_polygon(px, py, vertices):
    """
    Verifica se um ponto (px, py) está dentro do polígono formado pelos vértices no plano XY.

    Args:
        px, py: Coordenadas do ponto a ser verificado.
        vertices: Vértices do polígono no formato (N, 2).

    Returns:
        bool: Verdadeiro se o ponto está dentro do polígono, Falso caso contrário.
    """
    inside = False
    n = len(vertices)
    x, y = px, py

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Próximo vértice (circular)
        # Verifica se o ponto cruza a borda do polígono
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside

    return inside


def color_points_within_bbox(points, all_bbox_vertices):
    """
    Altera a cor dos pontos dentro do bounding box 3D para vermelho.

    Args:
        points (o3d.geometry.PointCloud): Nuvem de pontos a ser verificada.
        bbox_vertices (np.ndarray): Vértices do bounding box (8x3).

    Returns:
        o3d.geometry.PointCloud: Nuvem de pontos com cores alteradas.
    """

    # Vértices da base inferior (0, 1, 2, 3)
    all_bbox_vertices_base = [(bbox_vertices[[0, 1, 2, 3]]) for bbox_vertices in all_bbox_vertices]

    # Altura mínima e máxima do bounding box
    bbox_max_height = [(np.max(bbox_vertices[:, 2])) for bbox_vertices in all_bbox_vertices]
    bbox_min_height = [(np.min(bbox_vertices[:, 2])) for bbox_vertices in all_bbox_vertices]

    # Extrair as coordenadas x, y da base inferior
    base_points = [(bbox_vertices_base[:, :2]) for bbox_vertices_base in all_bbox_vertices_base]  # Apenas x e y

    # Extrair os pontos da nuvem de pontos
    points_array = np.asarray(points.points)
    colors_array = np.asarray(points.colors)

    # Iterar sobre os pontos e verificar se estão dentro da base inferior e da altura
    for i, point in enumerate(points_array):
        px, py, pz = point

        # Verificar se o ponto está dentro do polígono formado pela base inferior
        for j, item in enumerate(all_bbox_vertices):
            if point_in_polygon(px, py, base_points[j]) and bbox_min_height[j] <= pz <= bbox_max_height[j]:
                colors_array[i] = [1.0, 0.0, 0.0]  # RGB vermelho
                break

    points.colors = o3d.utility.Vector3dVector(colors_array)  # Atualizar as cores

    return points


def project_to_range_image(vertices, laser_calib, height, width):
    """
    Projeta os vértices 3D para a imagem range.

    Args:
        vertices (np.ndarray): Vértices 3D no sistema de coordenadas do LiDAR.
        laser_calib (LaserCalibration): Objeto de calibração do LiDAR.
        height (int): Altura da range image.
        width (int): Largura da range image.

    Returns:
        np.ndarray: Coordenadas projetadas (u, v) para a imagem range.
    """
    # Construir a matriz extrínseca a partir de `laser_calib`
    extrinsic = np.array(laser_calib.extrinsic.transform).reshape(4, 4)

    # Aplicar transformação extrínseca
    vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])  # Coordenadas homogêneas
    points_cam = vertices_homogeneous @ extrinsic.T  # Transformar para o sistema de coordenadas da câmera

    # Evitar divisão por zero ao projetar
    points_cam[:, 0] = np.maximum(points_cam[:, 0], 1e-6)

    # Projeção esférica
    u = (np.arctan2(points_cam[:, 1], points_cam[:, 0]) / (2 * np.pi) + 0.5) * width
    v = (np.arcsin(points_cam[:, 2] / np.linalg.norm(points_cam[:, :3], axis=1)) / np.pi + 0.5) * height

    # Garantir que as coordenadas estejam nos limites da imagem
    u = np.clip(u, 0, width - 1).astype(int)
    v = np.clip(v, 0, height - 1).astype(int)

    return np.stack([u, v], axis=1)


def draw_bbox_on_range_image(range_image, vertices_projected):
    """
    Desenha as linhas do bounding box projetado na range image.

    Args:
        range_image (np.ndarray): Imagem range (altura, largura, 3).
        vertices_projected (np.ndarray): Coordenadas projetadas (u, v) para a imagem range.
    """
    # Linhas do bounding box (base inferior e superior + arestas verticais)
    bbox_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Base inferior
        [4, 5], [5, 6], [6, 7], [7, 4],  # Base superior
        [0, 4], [1, 5], [2, 6], [3, 7]   # Conexões entre bases
    ]

    for line in bbox_lines:
        pt1 = tuple(vertices_projected[line[0]])
        pt2 = tuple(vertices_projected[line[1]])
        cv2.line(range_image, pt1, pt2, (0, 255, 0), 1)  # Verde

def calculate_birdview_image(points_xyz, meters=75, scale=10):
    size_in_pixels = int(2 * meters * scale)
    birdview_image = np.zeros((size_in_pixels, size_in_pixels), dtype=np.uint8)

    for point in points_xyz:
        x, y = point[0], point[1]
        if -meters < x < meters and -meters < y < meters:
            x_pixel = int((x + meters) * scale)
            y_pixel = int((meters - y) * scale)
            birdview_image[y_pixel, x_pixel] = 255

    birdview_image_color = cv2.cvtColor(birdview_image, cv2.COLOR_GRAY2BGR)
    return birdview_image_color


def draw_bounding_box_birdview(vertices, frame_pose, birdview_image, meters=75, scale=10):
    vertices_transformados = np.array([transformar_para_sistema_lidar_topo(vertice, frame_pose) for vertice in vertices])
    
    # Ignorar o veículo (robô) usando a posição central (próximo de [0, 0, 0])
    center_x, center_y = vertices_transformados[0][:2]
    if np.linalg.norm([center_x, center_y]) < 7.0:  # Se a posição do centro está próxima de [0, 0, 0]
        return
    
    # ---- Bird's-Eye View (2D) ----
    for i in range(4):
        x1, y1 = vertices_transformados[i][:2]
        x2, y2 = vertices_transformados[(i + 1) % 4][:2]
        if -meters < x1 < meters and -meters < y1 < meters:
            x_pixel1 = int((x1 + meters) * scale)
            y_pixel1 = int((meters - y1) * scale)
            x_pixel2 = int((x2 + meters) * scale)
            y_pixel2 = int((meters - y2) * scale)
            cv2.line(birdview_image, (x_pixel1, y_pixel1), (x_pixel2, y_pixel2), (0, 255, 0), 2)


def make_sure_path_exists(caminho):
    try:
        os.makedirs(caminho)
    except OSError as excecao:
        if excecao.errno != errno.EEXIST:
            raise


# Função para salvar os dados em um arquivo de texto
def save_data_to_txt(points, filename):
    with open(filename, "w") as file:
        # Começar o bloco de listas
        # file.write("[[\n")
        
        # Escrever cada ponto no formato desejado
        for point in points:
            if (len(point) == 3):
                file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            if (len(point) == 4):
                file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.6f}\n")
            # file.write(f" [ {point[0]:.6f}  {point[1]:.6f}   {point[2]:.6f} ]\n")
        
        # Fechar o bloco de listas
        # file.write("]]\n")

    # print(f"Dados salvos em {filename}")


# Função para gravar os pontos LiDAR em um arquivo binário
def save_lidar_points_to_bin(lidar_points, bin_path):
    
    with open(bin_path, "ab") as fs:  # "ab" para abrir em modo binário e anexar
        for point in lidar_points:
            # print(point)
            # Converter os dados do ponto para o formato binário
            data = struct.pack('ffff', point[0], point[1], point[2], point[3])
            fs.write(data)

    # print(f"Pontos gravados no arquivo: {bin_path}")


def process(filepath, proto, bin_path, objs_path, poses_path):
    WOMD_FILE = filepath
    womd_original_scenario = _load_scenario_data(WOMD_FILE)
    print(f"Scenario ID: {womd_original_scenario.scenario_id}")

    scenario_augmented = womd_lidar_utils.augment_womd_scenario_with_lidar_points(womd_original_scenario, womd_original_scenario)
    frame_i = 0
    index = 0
    for frame_lasers in scenario_augmented.compressed_frame_laser_data:
        frame_pose = np.reshape(np.array(frame_lasers.pose.transform), (4, 4))
        # print(frame_pose)

        if (frame_i < 11): 
            save_data_to_txt(frame_pose, os.path.join(poses_path, str(index) + ".txt"))

            for laser in frame_lasers.lasers:
                if (laser.name == dataset_pb2.LaserName.TOP) and proto.tracks:
                    subdir = os.path.join(objs_path, str(index))
                    make_sure_path_exists(subdir)

                    laser_calib = _get_laser_calib(frame_lasers, dataset_pb2.LaserName.TOP)
                    # Extrair os pontos LIDAR
                    top_lidar_points = womd_lidar_utils.extract_top_lidar_points(laser, frame_pose, laser_calib)
                    points_xyz = top_lidar_points[0].numpy()
                    intensities = top_lidar_points[1][:, 0]

                    # Atribuir cores padrão (branco) para os pontos LIDAR
                    points_colors = np.ones_like(points_xyz) * [0.0, 0.0, 1.0]  # Azul RGB

                    geometries = []

                    # Adicionar nuvem de pontos à visualização
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
                    point_cloud.colors = o3d.utility.Vector3dVector(points_colors)
                    geometries.append(point_cloud)
                    all_bbox = []
                    all_bbox_lines = []

                    birdview_image = calculate_birdview_image(points_xyz)

                    # Adicionar os bounding boxes 3D
                    track_index = 0
                    for track in proto.tracks:
                        if track.states[frame_i].valid:
                            vertices = calculate_bounding_box_vertices(track.states[frame_i])
                            save_data_to_txt(vertices, os.path.join(subdir, str(track_index) + ".txt"))
                            track_index+=1


                    # Converta o TensorFlow tensor para um numpy array
                    intensities_values = intensities.numpy()
                    points_array = np.asarray(point_cloud.points)

                    
                    # Adiciona o valor de reflexão a cada ponto (tornando o array de 3 valores para 4)
                    points_array = np.hstack((points_array, intensities_values[:, np.newaxis]))
                    # print(points_array)
                    file = os.path.join(bin_path, str(index) + ".bin")
                    save_lidar_points_to_bin(points_array, file)
                    index+=1
                    # point_cloud = color_points_within_bbox(point_cloud, all_bbox)

        frame_i += 1


if __name__ == "__main__":
    # Argumento da linha de comando para o caminho do arquivo
    parser = argparse.ArgumentParser(description='Associar pontos de track com LiDAR')
    parser.add_argument('-tf', '--tfrecord', type=str, help='Caminho do arquivo .tfrecord', required=True)
    parser.add_argument('-s', '--scenario', type=str, help='Caminho do arquivo scenario', required=True)
    parser.add_argument('-t', '--tolerancia', type=float, help='Tolerância para associação dos pontos', default=3.0)
    args = parser.parse_args()

    raw_dataset = tf.data.TFRecordDataset([args.scenario])

    # Processar o primeiro record do dataset
    i = 0
    for raw_record in raw_dataset:
        # if i >= 7:
        # Obter os dados do arquivo TFRecord e convertê-los para string de bytes
        proto_string = raw_record.numpy()

        # Instanciar o proto e parsear o conteúdo
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(proto_string)

        # Acessar o campo scenario_id e outros campos do Scenario
        print(f"Scenario ID: {proto.scenario_id}")
        root_path = "/home/lume/astro/data/lidar_sweep_viewer" 
        make_sure_path_exists(root_path)
        objs_path = os.path.join(os.path.join(root_path, "objs_bbox"), str(proto.scenario_id))
        make_sure_path_exists(objs_path)
        poses_path = os.path.join(os.path.join(root_path, "poses"), str(proto.scenario_id))
        make_sure_path_exists(poses_path)
        bin_path = os.path.join(os.path.join(root_path, "bin_files"), str(proto.scenario_id))
        make_sure_path_exists(bin_path)
    
        lidar_path = proto.scenario_id + '.tfrecord'
        file = os.path.join(args.tfrecord, lidar_path)

        if os.path.isfile(file):
            print(f"Processando {file}...")
            process(file, proto, bin_path, objs_path, poses_path)
        else:
            print(f"Arquivo {file} não existe, passando...")

        # i+=1
