import os
import cv2
import math
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


def normalizar(image, meters):
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
            if mask[x, y]:
                image_color[x, y] = np.array([0, 0, 255])
    return image_color


def transformar_para_sistema_lidar_topo(track_coords, lidar_pose):
    track_coords_homogeneas = np.append(track_coords, 1.0)
    lidar_pose_inversa = np.linalg.inv(lidar_pose)
    track_coords_transformadas = np.dot(lidar_pose_inversa, track_coords_homogeneas)
    return track_coords_transformadas[:3]


def transformar_points_para_coordenada_veiculo(points_xyz, laser_calib):
    """
    Aplica a transformação para converter os pontos do LiDAR
    para o sistema de coordenadas do veículo, utilizando a matriz de extrínsecos.

    Parameters:
        points_xyz: np.ndarray de forma (N, 3) ou (3,) contendo as coordenadas dos pontos no sistema do LiDAR.
        laser_calib: Objeto contendo a matriz de calibração extrínseca do laser.

    Returns:
        points_xyz_transformados: np.ndarray de forma (N, 3) com os pontos no sistema de coordenadas do veículo.
    """
    # Obter e verificar a matriz de extrínsecos
    extrinsic = np.reshape(np.array(laser_calib.extrinsic.transform), [4, 4])
    if extrinsic.shape != (4, 4):
        raise ValueError(f"A matriz de extrínsecos deve ter forma (4, 4). Obtida: {extrinsic.shape}")

    # Garantir que points_xyz seja uma matriz 2D
    points_xyz = np.atleast_2d(points_xyz)  # Converte (3,) para (1, 3) se necessário
    if points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz deve ter 3 colunas (x, y, z). Obtido: {points_xyz.shape}")

    # Adicionar a coordenada homogênea (1) para cada ponto
    num_pontos = points_xyz.shape[0]
    pontos_homogeneos = np.hstack((points_xyz, np.ones((num_pontos, 1))))

    # Aplicar a matriz de extrínsecos (transformação do sensor para o veículo)
    pontos_veiculo_frame = np.dot(pontos_homogeneos, extrinsic.T)

    # Retornar as três primeiras coordenadas (x, y, z) dos pontos no sistema de referência do veículo
    return pontos_veiculo_frame[:3]


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


def calcular_resolucao_angular(lidar_calib, num_rows, num_cols):
    """
    Calcula a resolução angular da range image em termos de incremento de ângulo entre as células da matriz.
    
    Parameters:
        lidar_calib: Calibração do LiDAR contendo os ângulos de inclinação máximo e mínimo.
        num_rows: Número de linhas da range image.
        num_cols: Número de colunas da range image.
    
    Returns:
        Resolucao angular vertical e horizontal em graus por célula.
    """
    # Ângulo de inclinação vertical (campo de visão vertical)
    inclinacao_vertical_total = lidar_calib.beam_inclination_max - lidar_calib.beam_inclination_min
    # print(inclinacao_vertical_total)
    resolucao_angular_vertical = inclinacao_vertical_total / num_rows  # graus por linha

    # Ângulo horizontal (campo de visão horizontal em 360 graus)
    resolucao_angular_horizontal = 360.0 / num_cols  # graus por coluna
    return inclinacao_vertical_total
    # return resolucao_angular_vertical, resolucao_angular_horizontal

def projetar_para_lidar(vertices_transformados, laser_calib, lidar_vertical_fov, u_size, v_size):
    projected_points = []

    # Angles of inclination for LiDAR (in radians)
    min_pitch = laser_calib.beam_inclination_min
    max_pitch = laser_calib.beam_inclination_max
    # print(np.rad2deg(lidar_vertical_fov))
    vertices_transformados = vertices_transformados[:, :3]
    for vertice in vertices_transformados:
        # Extract x, y, z in the LiDAR frame
        x, y, z = vertice
        
        # Calculate distance from LiDAR to point
        distancia = np.sqrt(x**2 + y**2)

        # # Azimuth angle (yaw) calculation and mapping to x_range
        # azimuth_angle = np.arctan2(y, x)
        # x_range = int(((-azimuth_angle + np.pi) / (2 * np.pi)) * lidar_horizontal_resolution)

        # # Elevation angle (pitch) calculation and mapping to y_range
        # pitch = np.arctan2(z, distancia)
        # y_range = int(((-pitch + lidar_vertical_fov) / ( lidar_vertical_fov)) * lidar_vertical_resolution)

        # Azimuth angle (yaw) calculation and mapping to u
        yaw = np.arctan2(y, x)
        # u = int(((-yaw + np.pi) / (2 * np.pi)) * u_size)
        u = int(u_size * ((yaw + np.pi) / (2.0 * np.pi)))

        # Elevation angle (pitch) calculation and mapping to v
        distancia = np.sqrt(x**2.0 + y**2.0 + z**2.0)
        pitch = np.arctan2(z, distancia)
        v = int(((pitch + lidar_vertical_fov) / (2* lidar_vertical_fov)) * v_size)
        # v = int(((max_pitch - pitch) / (2*(max_pitch - min_pitch))) * v_size)
        # v = v_size - int(((pitch - min_pitch) / (max_pitch - min_pitch)) * v_size)

        # Add point if within range of image bounds
        if 0 <= u < u_size and 0 <= v < v_size:
            projected_points.append((u, v))


    return projected_points


def transformar_para_eixo_veiculo(pontos, frame_pose, lidar_pose):
    """
    Transforma os pontos do sistema de coordenadas do sensor para o sistema de coordenadas do veículo
    subtraindo as translações entre frame_pose e lidar_pose.

    Args:
        pontos: np.ndarray de forma (N, 3) ou (3,), contendo os pontos no sistema de coordenadas do sensor.
        frame_pose: np.ndarray de forma (4, 4), posição do sensor no mundo.
        lidar_pose: np.ndarray de forma (4, 4), posição do sensor em relação ao veículo.

    Returns:
        pontos_no_veiculo: np.ndarray de forma (N, 3), contendo os pontos no sistema de coordenadas do veículo.
    """
    # Garantir que pontos seja 2D
    pontos = np.atleast_2d(pontos)

    # Obter os vetores de translação de frame_pose e lidar_pose
    translacao_frame = frame_pose[:3, 3]
    translacao_lidar = lidar_pose[:3, 3]

    # Subtrair as translações
    translacao_total = translacao_frame - translacao_lidar

    # Ajustar os pontos para o sistema do veículo
    pontos_no_veiculo = pontos - translacao_total

    # return pontos_no_veiculo

    return pontos_no_veiculo[0][:3]


def draw_bounding_box3d_range_view(vertices, frame_pose, range_image, laser_calib, meters=75, scale=10):
    lidar_horizontal_resolution = range_image.shape[1]
    lidar_vertical_resolution = range_image.shape[0]

    lidar_vertical_fov = calcular_resolucao_angular(laser_calib, lidar_vertical_resolution, lidar_horizontal_resolution)
    extrinsic = np.reshape(np.array(laser_calib.extrinsic.transform), [4, 4])


    # print(frame_pose)
    # print(extrinsic)
    # Obter os vetores de translação de frame_pose e lidar_pose
    # translacao_frame = frame_pose[:3, 3]
    # translacao_lidar = extrinsic[:3, 3]

    # Subtrair as translações
    # translacao_total = translacao_frame + translacao_lidar
    # frame_pose[:3, 3] = translacao_total
    # frame_pose = extrinsic 
    vertices_transformados = np.array([transformar_para_sistema_lidar_topo(vertice, frame_pose) for vertice in vertices])
    # vertices_transformados = np.array([transformar_para_eixo_veiculo(vertice, frame_pose, extrinsic) for vertice in vertices_transformados])

    # vertices_transformados = np.array(vertices_transformados)  # Garante que é um array numpy
    # Ignorar o veículo (robô) usando a posição central (próximo de [0, 0, 0])
    print(vertices_transformados[0])
    center_x, center_y = vertices_transformados[0][:2]
    if np.linalg.norm([center_x, center_y]) < 7.0:  # Se a posição do centro está próxima de [0, 0, 0]
        return

    
    # Projeção no range view com espelhamento no eixo X
    projected_bbox = projetar_para_lidar(vertices_transformados, laser_calib, lidar_vertical_fov, lidar_horizontal_resolution, lidar_vertical_resolution)
    
    # Verifique se a projeção retornou 8 pontos
    if len(projected_bbox) < 8:
        print("Aviso: Projeção incompleta, ignorando esta caixa delimitadora.")
        return   
    
    # ---- Range View (3D) ----
    for i in range(4):
        cv2.line(range_image, projected_bbox[i], projected_bbox[(i + 1) % 4], (0, 255, 0), 2)
        cv2.line(range_image, projected_bbox[i + 4], projected_bbox[((i + 1) % 4) + 4], (0, 255, 0), 2)
        cv2.line(range_image, projected_bbox[i], projected_bbox[i + 4], (0, 255, 0), 2)


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


def process(filepath, proto):
    WOMD_FILE = filepath
    womd_original_scenario = _load_scenario_data(WOMD_FILE)
    print(f"Scenario ID: {womd_original_scenario.scenario_id}")

    scenario_augmented = womd_lidar_utils.augment_womd_scenario_with_lidar_points(womd_original_scenario, womd_original_scenario)
    frame_i = 0

    for frame_lasers in scenario_augmented.compressed_frame_laser_data:
        frame_pose = np.reshape(np.array(frame_lasers.pose.transform), (4, 4))
        if (frame_i < 11): 
            for laser in frame_lasers.lasers:
                if laser.name == dataset_pb2.LaserName.TOP:
                    laser_calib = _get_laser_calib(frame_lasers, dataset_pb2.LaserName.TOP)
                    points_feature = womd_lidar_utils.extract_top_lidar_points(laser, frame_pose, laser_calib)[1]
                    range_image = points_feature[:, 0]
                    shape = np.array([64, 2650, 1])
                    range_image_reshaped = np.array(range_image).reshape(shape)
                    normalized_image_range_image = normalizar(range_image_reshaped, 75)
                    
                    # Extrair os pontos LIDAR
                    points_xyz = womd_lidar_utils.extract_top_lidar_points(laser, frame_pose, laser_calib)[0].numpy()
                    
                    # A conversão para numpy é necessária aqui
                    distances = np.linalg.norm(points_xyz, axis=1)
                    
                    # Filtrar pontos que estão dentro do limite de distância
                    points_within_limit = points_xyz[distances <= 80]
        

                    # Atribuir cores padrão (branco) para os pontos LIDAR
                    points_colors = np.ones_like(points_within_limit) * [0.0, 0.0, 1.0]  # Azul RGB

                    geometries = []

                    # Adicionar nuvem de pontos à visualização
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(points_within_limit)
                    point_cloud.colors = o3d.utility.Vector3dVector(points_colors)
                    geometries.append(point_cloud)

                    # Adicionar os bounding boxes 3D
                    if proto.tracks:
                        for track in proto.tracks:
                            if track.states[frame_i].valid:
                                vertices = calculate_bounding_box_vertices(track.states[frame_i])
                                vertices_transformed = transform_vertices(vertices, frame_pose)  # Transformar os vértices
                                draw_bounding_box3d_range_view(vertices, frame_pose, normalized_image_range_image, laser_calib)
                                bbox_lines = create_bounding_box_lines(vertices_transformed)  # Criar LineSet
                                geometries.append(bbox_lines)  # Adicionar à lista de geometria

                    # Visualizar tudo
                    # o3d.visualization.draw_geometries(geometries)

                    # Exibir ambas as visualizações
                    cv2.imshow('LiDAR Range', cv2.resize(normalized_image_range_image, (1920, 128)))
                    
                    cv2.waitKey(1)

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
    for raw_record in raw_dataset:
        # Obter os dados do arquivo TFRecord e convertê-los para string de bytes
        proto_string = raw_record.numpy()

        # Instanciar o proto e parsear o conteúdo
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(proto_string)

        # Acessar o campo scenario_id e outros campos do Scenario
        print(f"Scenario ID: {proto.scenario_id}")

        lidar_path = proto.scenario_id + '.tfrecord'
        file = os.path.join(args.tfrecord, lidar_path)

        print(f"Processando {file}...")
        process(file, proto)
