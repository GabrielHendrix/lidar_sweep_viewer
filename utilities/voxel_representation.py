import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def find_and_correct_ground_plane(pcd, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
    """
    Usa RANSAC para encontrar o plano do solo, calcula seu ângulo e, em seguida,
    rotaciona e translada toda a nuvem de pontos para nivelar o solo em Z=0.
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

    normal_vector = np.array([a, b, c])
    z_axis_vector = np.array([0, 0, 1])
    angle_rad = np.arccos(np.abs(np.dot(normal_vector, z_axis_vector)) / np.linalg.norm(normal_vector))
    angle_deg = np.degrees(angle_rad)
    print(f"O plano do solo está inclinado em aproximadamente {angle_deg:.2f} graus.")

    if normal_vector[2] < 0:
        normal_vector = -normal_vector
        
    rotation_axis = np.cross(normal_vector, z_axis_vector)
    if np.linalg.norm(rotation_axis) > 1e-6: # Evita normalização de vetor nulo
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_vector = rotation_axis * angle_rad
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
    else: # Vetores já alinhados
        R = np.identity(3)

    pcd_corrected = o3d.geometry.PointCloud(pcd)
    pcd_corrected.rotate(R, center=(0, 0, 0))

    inlier_cloud_corrected = pcd_corrected.select_by_index(inliers)
    mean_z_inliers = np.mean(np.asarray(inlier_cloud_corrected.points)[:, 2])
    
    translation_vector = [0, 0, -mean_z_inliers]
    pcd_corrected.translate(translation_vector)
    
    print(f"Nuvem de pontos corrigida. O plano do solo agora está em Z=0.")

    # Visualização opcional da correção
    # inlier_cloud_final = pcd_corrected.select_by_index(inliers)
    # inlier_cloud_final.paint_uniform_color([0, 1, 0])
    # outlier_cloud_final = pcd_corrected.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud_final, outlier_cloud_final])

    return pcd_corrected, angle_deg


def generate_bev_and_visualize_voxels(pcd, x_range, y_range, z_range, voxel_size, image_size=1024):
    """
    Cria uma imagem bird's-eye view e uma visualização 3D a partir de uma grade de voxels.
    """
    print("\n--- Gerando BEV e visualização a partir de Voxels ---")
    print(f"Tamanho do Voxel: {voxel_size} metros")

    # 1. Filtrar a nuvem de pontos
    points = np.asarray(pcd.points)
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])

    if not pcd_filtered.has_points():
        print("Nenhum ponto encontrado no range especificado para voxelização.")
        return

    # 2. Criar a grade de voxels
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_filtered, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()
    print(f"Nuvem de pontos convertida em {len(voxels)} voxels.")

    # 3. Gerar a imagem Bird's-Eye View (esta parte não muda)
    height_map = np.full((image_size, image_size), z_range[0] - 1, dtype=np.float32)
    x_res = (x_range[1] - x_range[0]) / image_size
    y_res = (y_range[1] - y_range[0]) / image_size

    for voxel in voxels:
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        x, y, z = center

        # Calcula a área do voxel em coordenadas de pixel
        half_voxel_px = (voxel_size / x_res) / 2
        half_voxel_py = (voxel_size / y_res) / 2
        
        center_px = (x - x_range[0]) / x_res
        center_py = (y - y_range[0]) / y_res

        px_start = int(center_px - half_voxel_px)
        px_end = int(center_px + half_voxel_px)
        py_start = int(center_py - half_voxel_py)
        py_end = int(center_py + half_voxel_py)

        # Garante que as coordenadas estejam dentro dos limites da imagem
        px_start_clipped = max(0, px_start)
        px_end_clipped = min(image_size, px_end)
        py_start_clipped = max(0, py_start)
        py_end_clipped = min(image_size, py_end)

        # Preenche o retângulo no mapa de altura, garantindo que não haja sobreposição com valores menores
        if px_start_clipped < px_end_clipped and py_start_clipped < py_end_clipped:
            region = height_map[py_start_clipped:py_end_clipped, px_start_clipped:px_end_clipped]
            height_map[py_start_clipped:py_end_clipped, px_start_clipped:px_end_clipped] = np.maximum(region, z)

    height_map[height_map == z_range[0] - 1] = z_range[0]
    z_delta = z_range[1] - z_range[0]
    if z_delta == 0: z_delta = 1.0
    normalized_map = (height_map - z_range[0]) / z_delta
    normalized_map = np.clip(normalized_map, 0, 1)
    
    colored_image = plt.cm.viridis(normalized_map)
    colored_image = np.rot90(colored_image, k=1)
    contiguous_image = np.ascontiguousarray(colored_image)
    plt.imsave("bird_eye_view_voxels.png", (contiguous_image * 255).astype(np.uint8))
    print("Imagem 'bird_eye_view_voxels.png' salva com sucesso.")

    # 4. --- CORREÇÃO APLICADA AQUI ---
    # Em vez de tentar colorir o VoxelGrid, criamos uma malha (um cubo) para cada voxel.
    norm = plt.Normalize(vmin=z_range[0], vmax=z_range[1])
    cmap = plt.cm.viridis
    
    # Lista para guardar todos os cubos coloridos
    meshes = []
    for voxel in voxels:
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        color = cmap(norm(center[2]))[:3] # Cor baseada na altura Z

        # Criar um cubo (malha) para o voxel
        # O tamanho do cubo é o mesmo do voxel para que fiquem juntos
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        
        # O box é criado na origem, então o transladamos para o centro do voxel
        mesh_box.translate(center, relative=False)
        
        # Pintamos o cubo com a cor da altura
        mesh_box.paint_uniform_color(color)
        
        meshes.append(mesh_box)
    
    print("\nMostrando visualização 3D dos voxels (como malhas de cubos)...")
    o3d.visualization.draw_geometries(meshes)
    

def main():
    # Load binary point cloud
    bin_pcd = np.fromfile("/home/lume/astro/data/lidar_sweep_viewer/bin_files/edb310506f85823b/10.bin", dtype=np.float32)
    # Reshape and drop reflection values
    # Remodela para (N, 4) e remove os valores de reflexão para obter (N, 3)
    # points_2d = bin_pcd.reshape((-1, 4))[:, 0:3]
    # points_2d = bin_pcd.reshape((-1, 4))[:, 3]
    points_2d = bin_pcd.reshape((-1, 4))[:, 0:3]

    # Check if shape is (N, 2) and pad with zeros
    if points_2d.shape[1] == 2:
        # Create a column of zeros
        zeros = np.zeros((points_2d.shape[0], 1))
        # Stack them to make (N, 3)
        points_3d = np.hstack((points_2d, zeros))
    else:
        points_3d = points_2d

    # Now create the point cloud with float64
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d.astype(np.float64)))
    # ply_path = "/home/lume/Desktop/velodyne_0/0119.ply"

    # try:
    #     pcd = o3d.io.read_point_cloud(ply_path)
    # except Exception as e:
    #     print(f"Erro ao ler o arquivo {ply_path}: {e}")
    #     return

    # print("Informações da nuvem de pontos original:")
    # print(pcd)
    
    # 1. Encontra o plano e retorna a nuvem de pontos CORRIGIDA
    pcd_corrected, angle = find_and_correct_ground_plane(pcd)

    if pcd_corrected is None:
        print("Não foi possível corrigir a nuvem de pontos. Abortando.")
        return
        
    # 2. Gerar BEV e visualizar usando VOXELS a partir da nuvem corrigida
    x_filter_range = (-30, 30)
    y_filter_range = (-30, 30)
    z_filter_range = (0.0, 3.0) 
    
    # *** NOVO PARÂMETRO: Tamanho do Voxel ***
    # Altere este valor para deixar os "pontos" mais grossos ou mais finos.
    # Valor em metros. 0.2 = voxels de 20cm x 20cm x 20cm.
    voxel_size = 0.2
    
    generate_bev_and_visualize_voxels(
        pcd_corrected,
        x_range=x_filter_range,
        y_range=y_filter_range,
        z_range=z_filter_range,
        voxel_size=voxel_size, 
        image_size=224
    )


if __name__ == "__main__":
    main()