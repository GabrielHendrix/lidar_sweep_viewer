import cv2
import numpy as np
import open3d as o3d
import sys

# Parâmetros do visualizador (ajuste conforme necessário)
RESOLUTION = 27.34  # igual ao C++

# ---
# Diretório de saída especificado pelo usuário
output_dir = "/home/lume/Desktop/range_images"  # padrão
if len(sys.argv) > 1:
    output_dir = sys.argv[1]
# Caminhos dos arquivos
img_path = f"{output_dir}/lidar_sweep_rgb_0001.png"
meta_path = f"{output_dir}/lidar_sweep_rgb_0001_meta.txt"
params_path = f"{output_dir}/lidar_sweep_rgb_0001_params.txt"

# ---
# Lê min_dist e max_dist do arquivo de metadados gerado junto com a imagem
with open(meta_path) as f:
    min_dist, max_dist = map(float, f.read().split())

# ---
# Lê parâmetros do sensor e varredura
params = {}
with open(params_path) as f:
    for line in f:
        key, *values = line.strip().split()
        if key in ["vertical_angles", "horizontal_angles_deltas"]:
            params[key] = np.array([float(v) for v in values])
        elif key in ["range_division_factor", "number_of_shots"]:
            params[key] = float(values[0])
        else:
            params[key] = values[0]
vertical_angles = params["vertical_angles"]
horizontal_angles_deltas = params["horizontal_angles_deltas"]
range_division_factor = params["range_division_factor"]

# ---
# Mapeamento de cor para distância (simétrico ao C++)
# C++: norm = (d - min_dist) / (max_dist - min_dist)
#       r = 255 * norm
#       g = 255 * (1.0 - norm)
#       b = ... (visualização)
# Python: norm = r / 255.0
#         d = min_dist + norm * (max_dist - min_dist)
# ---
def color_to_distance(rgb, min_dist, max_dist):
    # OpenCV: BGR
    r = rgb[2]
    norm = r / 255.0
    return min_dist + norm * (max_dist - min_dist)

# Reconstrução fiel da nuvem de pontos
# Considera ângulos e range conforme C++
def reconstruct_point_cloud_from_image(img_path, min_dist, max_dist, vertical_angles, horizontal_angles_deltas, range_division_factor):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    points = []
    for x in range(width):
        for y in range(height):
            rgb = img[y, x]
            if np.all(rgb == 0):
                continue  # ignora pontos sem retorno
            dist = color_to_distance(rgb, min_dist, max_dist)
            # C++: horizontal_angle depende do modelo
            horizontal_angle = -((x / width) * 360.0 + horizontal_angles_deltas[y])
            horizontal_angle = np.deg2rad(horizontal_angle)
            vertical_angle = np.deg2rad(vertical_angles[y])
            range_val = dist / range_division_factor
            X = range_val * np.cos(vertical_angle) * np.cos(horizontal_angle)
            Y = range_val * np.cos(vertical_angle) * np.sin(horizontal_angle)
            Z = range_val * np.sin(vertical_angle)
            points.append([X, Y, Z])
    return np.array(points)

# Visualização com Open3d
points = reconstruct_point_cloud_from_image(img_path, min_dist, max_dist, vertical_angles, horizontal_angles_deltas, range_division_factor)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
try:
    o3d.visualization.draw_geometries([pcd])
except AttributeError:
    o3d.draw_geometries([pcd])