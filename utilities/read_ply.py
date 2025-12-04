import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def find_and_correct_ground_plane(pcd, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
    """
    Usa RANSAC para encontrar o plano do solo, calcula seu ângulo e, em seguida,
    rotaciona e translada toda a nuvem de pontos para nivelar o solo em Z=0.

    Args:
        pcd (o3d.geometry.PointCloud): A nuvem de pontos de entrada.
        distance_threshold (float): Distância máxima de um ponto ao plano para ser considerado um inlier.

    Returns:
        tuple: (pcd_corrigida, angulo_graus) ou (None, None) se o plano não for encontrado.
    """
    print("\n--- Executando RANSAC para encontrar e corrigir o plano do solo ---")
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    
    if not inliers:
        print("RANSAC não conseguiu encontrar um plano.")
        return None, None

    [a, b, c, d] = plane_model
    print(f"Equação do plano encontrado: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # 1. CALCULAR O ÂNGULO (para informação)
    normal_vector = np.array([a, b, c])
    z_axis_vector = np.array([0, 0, 1])
    angle_rad = np.arccos(np.abs(np.dot(normal_vector, z_axis_vector)) / np.linalg.norm(normal_vector))
    angle_deg = np.degrees(angle_rad)
    print(f"O plano do solo está inclinado em aproximadamente {angle_deg:.2f} graus.")

    # 2. CALCULAR A MATRIZ DE ROTAÇÃO
    # Garantir que o vetor normal aponte para "cima" (Z positivo)
    if normal_vector[2] < 0:
        normal_vector = -normal_vector
        
    # --- CORREÇÃO APLICADA AQUI ---
    # Em vez de usar um método que não existe, calculamos o eixo de rotação (usando produto vetorial)
    # e o ângulo (usando produto escalar), e então criamos a matriz de rotação.
    
    # Eixo de rotação é perpendicular a ambos os vetores
    rotation_axis = np.cross(normal_vector, z_axis_vector)
    rotation_axis /= np.linalg.norm(rotation_axis) # Normaliza o eixo
    
    # O ângulo já foi calculado acima em radianos (angle_rad)
    # Criamos um vetor de rotação (axis * angle)
    rotation_vector = rotation_axis * angle_rad
    
    # Criamos a matriz de rotação a partir do vetor de rotação
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
    
    # 3. APLICAR A ROTAÇÃO
    pcd_corrected = o3d.geometry.PointCloud(pcd) # Criar uma cópia
    pcd_corrected.rotate(R, center=(0, 0, 0))

    # 4. CALCULAR E APLICAR A TRANSLAÇÃO
    inlier_cloud_corrected = pcd_corrected.select_by_index(inliers)
    mean_z_inliers = np.mean(np.asarray(inlier_cloud_corrected.points)[:, 2])
    
    translation_vector = [0, 0, -mean_z_inliers]
    pcd_corrected.translate(translation_vector)
    
    print(f"Nuvem de pontos corrigida. O plano do solo agora está em Z=0.")

    # --- Visualização da correção ---
    inlier_cloud_final = pcd_corrected.select_by_index(inliers)
    inlier_cloud_final.paint_uniform_color([0, 1, 0])
    
    outlier_cloud_final = pcd_corrected.select_by_index(inliers, invert=True)
    
    print("Mostrando nuvem de pontos CORRIGIDA (Plano=Verde)...")
    o3d.visualization.draw_geometries([inlier_cloud_final, outlier_cloud_final])

    return pcd_corrected, angle_deg


# Suas outras funções permanecem as mesmas
def generate_and_save_bird_eye_view(points, x_range=(-75, 75), y_range=(-75, 75), z_range=(-3, 0), image_size=1024):
    print(f"Filtrando pontos com Z entre {z_range[0]} e {z_range[1]}...")
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & 
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) & 
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    filtered_points = points[mask]
    if filtered_points.shape[0] == 0:
        print("Nenhum ponto encontrado no range de X, Y e Z especificado.")
        return None, None
    height_map = np.full((image_size, image_size), z_range[0] - 1, dtype=np.float32)
    x_res = (x_range[1] - x_range[0]) / image_size
    y_res = (y_range[1] - y_range[0]) / image_size
    for point in filtered_points:
        x, y, z = point
        px = int((x - x_range[0]) / x_res)
        py = int((y - y_range[0]) / y_res)
        px = min(px, image_size - 1)
        py = min(py, image_size - 1)
        height_map[py, px] = max(height_map[py, px], z)
    height_map[height_map == z_range[0] - 1] = z_range[0]
    z_delta = z_range[1] - z_range[0]
    if z_delta == 0: z_delta = 1.0 
    normalized_map = (height_map - z_range[0]) / z_delta
    normalized_map = np.clip(normalized_map, 0, 1)
    colored_image = plt.cm.viridis(normalized_map)
    colored_image = np.rot90(colored_image, k=1)
    contiguous_image = np.ascontiguousarray(colored_image)
    plt.imsave("bird_eye_view.png", (contiguous_image * 255).astype(np.uint8))
    print("Imagem 'bird_eye_view.png' salva com sucesso.")
    return height_map, z_range

def visualize_pillars(height_map, x_range=(-75, 75), y_range=(-75, 75), z_range=(-3, 0), image_size=1024):
    if height_map is None: return
    x_res = (x_range[1] - x_range[0]) / image_size
    y_res = (y_range[1] - y_range[0]) / image_size
    norm = plt.Normalize(vmin=z_range[0], vmax=z_range[1])
    cmap = plt.cm.viridis
    meshes = []
    for py in range(image_size):
        for px in range(image_size):
            height = height_map[py, px]
            if height > z_range[0]:
                x, y = px * x_res + x_range[0], py * y_res + y_range[0]
                pillar_height = height - z_range[0]
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=x_res, height=y_res, depth=pillar_height)
                center_x, center_y, center_z = x + x_res / 2, y + y_res / 2, z_range[0] + pillar_height / 2
                mesh_box.translate([center_x, center_y, center_z])
                color = cmap(norm(height))[:3]
                mesh_box.paint_uniform_color(color)
                meshes.append(mesh_box)
    if not meshes: return
    print("\nMostrando visualização 3D dos pilares... (Pode demorar para renderizar)")
    o3d.visualization.draw_geometries(meshes)


def main():
    ply_path = "/home/lume/Desktop/velodyne_0/0119.ply"

    try:
        pcd = o3d.io.read_point_cloud(ply_path)
    except Exception as e:
        print(f"Erro ao ler o arquivo {ply_path}: {e}")
        return

    print("Informações da nuvem de pontos original:")
    print(pcd)
    
    # --- MODIFICAÇÃO PRINCIPAL ---
    # 1. Encontra o plano e retorna a nuvem de pontos CORRIGIDA
    pcd_corrected, angle = find_and_correct_ground_plane(pcd)

    if pcd_corrected is None:
        print("Não foi possível corrigir a nuvem de pontos. Abortando.")
        return
        
    # 2. TODAS AS OPERAÇÕES SEGUINTES USARÃO A NUVEM CORRIGIDA
    points = np.asarray(pcd_corrected.points)
    
    # 3. Gerar imagem e obter mapa de altura a partir dos pontos CORRIGIDOS
    print("\n--- Geração do Bird's-Eye View (com nuvem corrigida) ---")
    x_filter_range = (-40, 40)
    y_filter_range = (-40, 40)
    # IMPORTANTE: Como o solo agora está em Z=0, o filtro de Z deve ser ajustado.
    # Este filtro agora pega pontos de 10cm abaixo do solo até 3m acima.
    z_filter_range = (0.0, 3.0) 
    height_map, z_range_used = generate_and_save_bird_eye_view(points, x_range=x_filter_range, y_range=y_filter_range, z_range=z_filter_range)

    # 4. Visualizar os pilares a partir dos dados CORRIGIDOS
    if height_map is not None:
        print("\n--- Visualização de Pilares 3D (com nuvem corrigida) ---")
        visualize_pillars(height_map, x_range=x_filter_range, y_range=y_filter_range, z_range=z_range_used)

if __name__ == "__main__":
    main()