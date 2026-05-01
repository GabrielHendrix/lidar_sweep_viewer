import os
import tensorflow as tf
import numpy as np
import cv2  # Usaremos OpenCV para renderização rápida
from datetime import datetime

# Import waymo
try:
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils
except ImportError:
    print("Erro no import do Waymo.")

# CAMINHO DO ARQUIVO
FILENAME = '/dados/waymo/training/segment-183829460855609442_430_000_450_000.tfrecord'

def extract_data(obj):
    """
    Função robusta para extrair dados (Mantida da versão anterior).
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
    
    return np.array([]) # Retorno vazio em caso de erro

def process_frame_for_display(lidar_object):
    """
    Converte o dado bruto do LiDAR em uma imagem colorida para o OpenCV.
    """
    # 1. Extrair dados
    data_np = extract_data(lidar_object)
    
    if data_np.size == 0:
        return None
    # --- CANAL 0: RANGE ---
    raw_data = data_np[..., 0]

    # 1. Cria máscara: Onde tem dados válidos? (Maior que 0)
    # Pontos sem retorno geralmente são 0 ou -1.
    mask_valid = raw_data >= 0

    # 2. Processamento normal
    max_dist = 75.0
    clipped = np.clip(raw_data, 0, max_dist)

    # Normaliza (0 a 255)
    norm_img = (clipped / max_dist * 255).astype(np.uint8)

    # Inverte: Perto=255, Longe=0
    # Mas isso faria o Fundo (que era 0) virar 255 (Branco). Vamos corrigir com a máscara.
    norm_img = 255 - norm_img

    # 3. APLICA A MÁSCARA: Tudo que não for válido vira PRETO (0)
    # Onde a máscara for False (sem dados), pintamos de 0.
    norm_img[~mask_valid] = 0 

    return cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

# def process_frame_for_display(lidar_object):
#     """
#     Converte o dado bruto do LiDAR em uma imagem colorida para o OpenCV.
#     """
#     # 1. Extrair dados
#     data_np = extract_data(lidar_object)
    
#     if data_np.size == 0:
#         return None

#     # Canal 0 é Range
#     range_data = data_np[..., 0]
    
#     # 2. Normalização para visualização (0 a 255)
#     # Definimos um alcance máximo de 75 metros para visualização. 
#     # Tudo além disso fica na cor máxima.
#     max_range = 75.0
#     range_data = np.clip(range_data, 0, max_range)
    
#     # Normaliza de 0.0-75.0 para 0-255 (uint8)
#     norm_image = (range_data / max_range * 255).astype(np.uint8)
    
#     # Inverte para que perto seja 'quente' e longe 'frio' (ou vice-versa, opcional)
#     # norm_image = 255 - norm_image 

#     # 3. Aplica Colormap (JET ou TURBO deixam parecido com o Matplotlib)
#     colored_image = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)

#     return colored_image

def main():
    if not os.path.exists(FILENAME):
        print(f"Erro: Arquivo {FILENAME} não encontrado.")
        return

    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    print(f"Reproduzindo arquivo: {FILENAME}")
    print("Pressione 'q' na janela do vídeo para sair.")

    for i, data in enumerate(dataset):
        # ... dentro do loop for i, data in enumerate(dataset): ...
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        print(f"--- Frame {i} ---")
        print(f"Timestamp (us): {frame.timestamp_micros}")

        # Converte microssegundos para segundos e cria o objeto de data
        dt_object = datetime.fromtimestamp(frame.timestamp_micros / 1e6)

        # Formata para string legível: Ano-Mês-Dia Hora:Min:Seg.Microssegundos
        print(f"Data Detalhada: {dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')}")

        # 1. Ver dados do Veículo (Pose)
        # A pose é uma matriz 4x4 achatada (16 floats)
        pose = np.array(frame.pose.transform).reshape(4, 4)
        print(f"Posição do Carro (X, Y, Z): {pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}")

        # 2. Ver quantos objetos foram detectados (Labels)
        print(f"Objetos detectados (LiDAR): {len(frame.laser_labels)}")
        if len(frame.laser_labels) > 0:
            obj = frame.laser_labels[0]
            # Tipos: 1=Vehicle, 2=Pedestrian, 3=Sign, 4=Cyclist
            print(f"  Exemplo: Objeto ID {obj.id} do tipo {obj.type}")
            print(f"  Velocidade do obj (lx, ly): {obj.metadata.speed_x:.2f}, {obj.metadata.speed_y:.2f}")

        # 3. Ver disponibilidade de Câmeras
        print(f"Imagens de câmera disponíveis: {len(frame.images)}") # Geralmente 5

        print("-" * 30)

        (range_images, _, _, _) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        # LiDAR TOP, retorno 0
        lidar_obj = range_images[open_dataset.LaserName.TOP][0]
        
        # Processa imagem
        img = process_frame_for_display(lidar_obj)

        if img is not None:
            # O Waymo gera imagens muito largas (ex: 2650x64). 
            # Vamos redimensionar verticalmente (ex: 5x) para ficar visível na tela.
            # h, w, _ = img.shape
            # for gray scale
            h, w = img.shape[:2]
            scale_y = 4 # Aumenta a altura em 6x para ver melhor as linhas
            img_resized = cv2.resize(img, (w, h * scale_y), interpolation=cv2.INTER_NEAREST)

            # Adiciona texto com número do frame
            cv2.putText(img_resized, f"Frame: {i}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Mostra janela
            cv2.imshow("Waymo LiDAR Sequence", img_resized)

            # Controla o FPS
            # waitKey(50) espera 50ms (aprox 20 FPS). Aumente para ficar mais lento.
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        else:
            print(f"Frame {i} vazio ou inválido.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    main()