import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse # Usaremos argparse para lidar com as opções

# -----------------------------------------------------------------------------
# FUNÇÃO 1: CORREÇÃO DO PLANO DO SOLO (comum a todos os modos)
# -----------------------------------------------------------------------------
def find_and_correct_ground_plane(pcd, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
    """
    Usa RANSAC para encontrar o plano do solo e nivela a nuvem de pontos em Z=0.
    """
    print("\n--- Executando RANSAC para encontrar e corrigir o plano do solo ---")
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    if not inliers:
        print("RANSAC não conseguiu encontrar um plano.")
        return None, None
    [a, b, c, d] = plane_model
    normal_vector = np.array([a, b, c])
    z_axis_vector = np.array([0, 0, 1])
    angle_rad = np.arccos(np.abs(np.dot(normal_vector, z_axis_vector)) / np.linalg.norm(normal_vector))
    angle_deg = np.degrees(angle_rad)
    print(f"O plano do solo está inclinado em aproximadamente {angle_deg:.2f} graus.")

    if normal_vector[2] < 0: normal_vector = -normal_vector
    rotation_axis = np.cross(normal_vector, z_axis_vector)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_vector = rotation_axis * angle_rad
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
    else:
        R = np.identity(3)

    pcd_corrected = o3d.geometry.PointCloud(pcd)
    pcd_corrected.rotate(R, center=(0, 0, 0))
    inlier_cloud_corrected = pcd_corrected.select_by_index(inliers)
    mean_z_inliers = np.mean(np.asarray(inlier_cloud_corrected.points)[:, 2])
    pcd_corrected.translate([0, 0, -mean_z_inliers])
    print(f"Nuvem de pontos corrigida. O plano do solo agora está em Z=0.")
    return pcd_corrected, angle_deg

# -----------------------------------------------------------------------------
# FUNÇÕES PARA O MODO "points" (Seu método original)
# -----------------------------------------------------------------------------
def generate_bev_from_points(points, x_range, y_range, z_range, image_size=1024):
    """Gera um mapa de altura a partir de pontos brutos."""
    print("\n--- Gerando BEV a partir de Pontos ---")
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    filtered_points = points[mask]
    if filtered_points.shape[0] == 0:
        print("Nenhum ponto encontrado no range especificado.")
        return None, None
    
    height_map = np.full((image_size, image_size), z_range[0] - 1, dtype=np.float32)
    x_res = (x_range[1] - x_range[0]) / image_size
    y_res = (y_range[1] - y_range[0]) / image_size
    for point in filtered_points:
        x, y, z = point
        px = int((x - x_range[0]) / x_res)
        py = int((y - y_range[0]) / y_res)
        if 0 <= px < image_size and 0 <= py < image_size:
            height_map[py, px] = max(height_map[py, px], z)
            
    height_map[height_map == z_range[0] - 1] = z_range[0]
    # Salvar a imagem
    z_delta = z_range[1] - z_range[0]
    if z_delta == 0: z_delta = 1.0
    normalized_map = np.clip((height_map - z_range[0]) / z_delta, 0, 1)
    # --- Mantido 'inferno' para a imagem VISUAL ---
    colored_image = np.rot90(plt.cm.inferno(normalized_map), k=1)
    plt.imsave("bird_eye_view_points.png", (np.ascontiguousarray(colored_image) * 255).astype(np.uint8))
    print("Imagem 'bird_eye_view_points.png' salva com sucesso.")
    return height_map

def visualize_pillars_from_map(height_map, x_range, y_range, z_range):
    """Visualiza um mapa de altura como pilares 3D."""
    if height_map is None: return
    print("\nMostrando visualização 3D de Pilares...")
    image_size = height_map.shape[0]
    x_res = (x_range[1] - x_range[0]) / image_size
    y_res = (y_range[1] - y_range[0]) / image_size
    norm = plt.Normalize(vmin=z_range[0], vmax=z_range[1])
    cmap, meshes = plt.cm.inferno, []
    for py in range(image_size):
        for px in range(image_size):
            height = height_map[py, px]
            if height > z_range[0] + 1e-6:
                x, y = px * x_res + x_range[0], py * y_res + y_range[0]
                pillar_height = height - z_range[0]
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=x_res, height=y_res, depth=pillar_height)
                mesh_box.translate([x + x_res/2, y + y_res/2, z_range[0] + pillar_height/2])
                mesh_box.paint_uniform_color(cmap(norm(height))[:3])
                meshes.append(mesh_box)
    if meshes: o3d.visualization.draw_geometries(meshes)

def load_and_decode_bev_image(encoded_image_path, z_range, height_step):
    """Carrega uma imagem codificada e a converte de volta para um mapa de altura."""
    print(f"\n--- Decodificando imagem invertida de {encoded_image_path} ---")
    try:
        # Carrega a imagem e garante que está no formato uint8 [0-255]
        encoded_image_float = plt.imread(encoded_image_path)
         # Pixels brancos 100%
        white_mask = np.all(encoded_image_float == 1.0, axis=-1)
        max_r = ((z_range[1] - z_range[0]) / height_step) / 256
        max_r_mask = encoded_image_float[:, :, 0] > max_r
        # Define os brancos como preto
        encoded_image_float[white_mask] = 0.0
        encoded_image_float[max_r_mask] = 0.0
        # 🔵 Zera o canal azul (B)
        encoded_image_float[..., 2] = 0.0
        encoded_image_float[..., 1] = 0.0
        # Máscara para ignorar branco (1,1,1)
        mask_non_white = np.any(encoded_image_float < 1.0, axis=-1)

        # Seleciona apenas pixels válidos (não pretos)
        valid_pixels = encoded_image_float[mask_non_white]

        if valid_pixels.size > 0:
            # Calcula min e max para cada canal
            min_color = valid_pixels.min(axis=0)
            max_color = valid_pixels.max(axis=0)

            print("\nValores mínimos e máximos (ignorando preto):")
            print(f"  R: min={min_color[0]:.3f}, max={max_color[0]:.3f}")
            print(f"  G: min={min_color[1]:.3f}, max={max_color[1]:.3f}")
            print(f"  B: min={min_color[2]:.3f}, max={max_color[2]:.3f}")
        else:
            print("Todos os pixels são pretos após a substituição.")

        encoded_image = (encoded_image_float[:, :, :3] * 255).astype(np.uint8)
        
    except FileNotFoundError:
        print(f"Erro: Arquivo de imagem não encontrado em '{encoded_image_path}'"); return None

    # --- ALTERAÇÃO AQUI: Inverte a imagem para reverter à codificação original ---
    # Cria uma máscara onde os pixels são totalmente brancos (255,255,255)
    
    # encoded_image = 255 - encoded_image
    print("Imagem revertida para o formato de dados original.")

    encoded_image = np.rot90(encoded_image, k=-1)
    image_size = encoded_image.shape[0]
    height_map = np.full((image_size, image_size), z_range[0], dtype=np.float32)
    
    # Pega apenas os pixels que não são pretos (que contêm dados)
    valid_pixels = np.where(encoded_image.any(axis=2))
    R = encoded_image[valid_pixels][:, 0]
    G = encoded_image[valid_pixels][:, 1]

    # Decodifica os canais R e G para obter o índice de altura
    height_indices = G.astype(int) * 256 + R.astype(int)
    
    # Converte os índices de volta para altura em metros
    heights = height_indices * height_step + z_range[0]
    height_map[valid_pixels] = heights
    
    print("Mapa de altura reconstruído com sucesso a partir da imagem.")
    return height_map

# -----------------------------------------------------------------------------
# FUNÇÃO PARA O MODO "voxels"
# -----------------------------------------------------------------------------

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
    create_encoded_bev_from_height_map(height_map, z_range, voxel_size, "bird_eye_view_voxels.png")

    # 4. --- CORREÇÃO APLICADA AQUI ---
    # Em vez de tentar colorir o VoxelGrid, criamos uma malha (um cubo) para cada voxel.
    norm = plt.Normalize(vmin=z_range[0], vmax=z_range[1])
    cmap = plt.cm.inferno
    
    # Lista para guardar todos os cubos coloridos
    meshes = []
    for voxel in voxels:
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        color = cmap(norm(center[2]))[:3] # Cor baseada na altura Z

        # Criar um cubo (malha) para o voxel
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        
        # O box é criado na origem, então o transladamos para o centro do voxel
        mesh_box.translate(center, relative=False)
        
        # Pintamos o cubo com a cor da altura
        mesh_box.paint_uniform_color(color)
        
        meshes.append(mesh_box)
    
    print("\nMostrando visualização 3D dos voxels (como malhas de cubos)...")
    o3d.visualization.draw_geometries(meshes)
# -----------------------------------------------------------------------------
# FUNÇÃO PARA O MODO "image"
# -----------------------------------------------------------------------------
def load_and_visualize_from_image(bev_image_path, x_range, y_range, z_range):
    """Carrega uma imagem BEV e chama a função de visualização de pilares."""
    print(f"\n--- Carregando imagem BEV de {bev_image_path} ---")
    try:
        bev_image = plt.imread(bev_image_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo de imagem não encontrado em '{bev_image_path}'"); return
    
    height_map = load_and_decode_bev_image(bev_image_path, z_range, 0.2) # Supondo voxel_size=0.2 para decodificar
    visualize_pillars_from_map(height_map, x_range, y_range, z_range)

def create_encoded_bev_from_height_map(height_map, z_range, height_step, filename):
    """Converte um mapa de altura float para uma imagem RGB codificada e a salva."""
    image_size = height_map.shape[0]
    encoded_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Pega apenas os pixels que contêm altura (ignora o fundo)
    valid_pixels = np.where(height_map > z_range[0])
    
    # Converte alturas para índices inteiros
    heights = height_map[valid_pixels]
    height_indices = np.round((heights - z_range[0]) / height_step).astype(int)
    
    # Codifica os índices nos canais R e G
    R = height_indices % 256
    G = height_indices // 256

    # Atribui os valores RGB à imagem
    encoded_image[valid_pixels] = np.stack([R, G, np.zeros_like(R)], axis=1)
    
    # --- ALTERAÇÃO AQUI: Inverte toda a imagem (preto vira branco, etc.) ---
    # encoded_image = 255 - encoded_image

    # Salva a imagem codificada (sem perdas)
    rotated_image = np.rot90(encoded_image, k=1)
    plt.imsave(filename, np.ascontiguousarray(rotated_image))
    print(f"Imagem invertida e codificada com altura salva em '{filename}'")

# -----------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL com seleção de modo
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Processa nuvens de pontos LiDAR e imagens Bird's-Eye View.")
    parser.add_argument(
        '--mode', type=str, default='voxels', choices=['points', 'voxels', 'image'],
        help="Modo de operação: 'points' (BEV de pontos), 'voxels' (BEV de voxels), ou 'image' (pilares de imagem)."
    )
    parser.add_argument(
        '--input_file', type=str, default="/home/lume/Desktop/velodyne_0/0119.ply",
        help="Caminho para o arquivo de entrada (.ply para modos 'points'/'voxels', .png/.jpg para modo 'image')."
    )
    args = parser.parse_args()

    # --- Parâmetros de configuração ---
    x_filter_range = (-30, 30)
    y_filter_range = (-30, 30)
    z_filter_range = (0.0, 3.0) 
    image_resolution = 1024
    
    print(f"--- MODO SELECIONADO: {args.mode.upper()} ---")

    if args.mode == 'image':
        load_and_visualize_from_image(
            args.input_file, x_range=x_filter_range, y_range=y_filter_range, z_range=z_filter_range
        )
    else:
        try:
            pcd = o3d.io.read_point_cloud(args.input_file)
        except Exception as e:
            print(f"Erro ao ler o arquivo {args.input_file}: {e}"); return

        pcd_corrected, angle = find_and_correct_ground_plane(pcd)
        if pcd_corrected is None: return

        if args.mode == 'points':
            points = np.asarray(pcd_corrected.points)
            height_map = generate_bev_from_points(
                points, x_range=x_filter_range, y_range=y_filter_range, z_range=z_filter_range, image_size=image_resolution
            )
            visualize_pillars_from_map(
                height_map, x_range=x_filter_range, y_range=y_filter_range, z_range=z_filter_range
            )
        elif args.mode == 'voxels':
            voxel_s = 0.2
            generate_bev_and_visualize_voxels(
                pcd_corrected, x_range=x_filter_range, y_range=y_filter_range, z_range=z_filter_range,
                voxel_size=voxel_s, image_size=image_resolution
            )

if __name__ == "__main__":
    main()