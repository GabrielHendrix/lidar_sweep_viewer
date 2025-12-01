import numpy as np
import open3d as o3d

# Load binary point cloud
bin_pcd = np.fromfile("/home/lume/astro/data/lidar_sweep_viewer/bin_files/edb310506f85823b/1.bin", dtype=np.float32)
# Reshape and drop reflection values
# Remodela para (N, 4) e remove os valores de reflexão para obter (N, 3)
# points_2d = bin_pcd.reshape((-1, 4))[:, 0:3]
points_2d = bin_pcd.reshape((-1, 4))[:, 3]

# --- AQUI ESTÁ A MUDANÇA ---
# Tenta remodelar o array 2D para o shape 3D desejado
try:
    points_3d = points_2d.reshape((2650, 64, 1))
    print("Shape do array 2D original:", points_2d.shape)
    print("Shape do novo array 3D:", points_3d.shape)

    # Se você quiser salvar este array 3D, use o formato .npy do NumPy
    # np.save("/home/lume/Desktop/meu_array_3d.npy", points_3d)

except ValueError as e:
    print(f"Erro no reshape: {e}")
    print(f"O número total de elementos no array ({points_2d.size}) não é compatível com o shape desejado (2650, 64, 3).")

# # Convert to Open3D point cloud
# o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_2d))

# # --- LINHA PARA VISUALIZAR A NUVEM DE PONTOS ---
# # Esta função abrirá uma nova janela com a visualização 3D.
# print("Abrindo a janela de visualização. Feche-a para continuar o script...")
# o3d.visualization.draw_geometries([o3d_pcd])

# # O script continua após você fechar a janela de visualização
# print("Janela fechada. Salvando o arquivo...")

# # Save to whatever format you like
# o3d.io.write_point_cloud("/home/lume/Desktop/0119.ply", o3d_pcd)