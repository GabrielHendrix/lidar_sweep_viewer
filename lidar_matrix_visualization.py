from datetime import datetime
import pandas as pd
import numpy as np
import json
import cv2
import os
import time

def normalize(image, meters):
    lines = (len(image))
    columns = (len(image[0]))
    resolution = ((meters * 100) / 256)
    for x in range(lines):
        for y in range(columns):
            pixel = image[x, y] 
            if (pixel == -1):
                image[x, y] = 0
            else:
                image[x, y] = (255 - ((pixel * 100) / resolution))

    return np.clip(image, 0, 255).astype(np.uint8)

def process_parquet(path_dir):
    # try:
    # Lê o arquivo .parquet
    df = pd.read_parquet(path_dir)
    images = []
    # Extrai os valores da range image de retorno 1 e 2
    range_image_return_1 = df['[LiDARComponent].range_image_return1.values'].iloc[0]
    shape_1 = df['[LiDARComponent].range_image_return1.shape'].iloc[0]
    range_image_return_2 = df['[LiDARComponent].range_image_return2.values'].iloc[0]
    shape_2 = df['[LiDARComponent].range_image_return2.shape'].iloc[0]
    # Converte os dados da range image em arrays numpy
    range_image_1 = np.array(range_image_return_1).reshape(shape_1)
    range_image_2 = np.array(range_image_return_2).reshape(shape_2)

    lidar_image_1 = range_image_1[:, :, 0]  # Seleciona o canal atual
    lidar_image_2 = range_image_2[:, :, 0]  # Seleciona o canal atual
    imagem_normalizada_1 = normalize(lidar_image_1, 75)  # Normaliza a imagem
    imagem_normalizada_2 = normalize(lidar_image_2, 20)  # Normaliza a imagem

    imagem_redimensionada_1 = cv2.resize(imagem_normalizada_1, (1325, 128), 
               interpolation = cv2.INTER_CUBIC)
    imagem_redimensionada_2 = cv2.resize(imagem_normalizada_2, (1325, 128), 
               interpolation = cv2.INTER_CUBIC)
    images.append(imagem_normalizada_1)
    images.append(imagem_normalizada_2)

 # Adicionando espaçamento entre as imagens
    spacer_height = 10  # Tamanho do espaçamento entre as imagens
    spacer = np.ones((spacer_height, shape_1[1]), dtype=np.uint8) * 255  # Um espaço branco (255)
    
    # Concatenar todas as imagens verticalmente
    final_image = images[0]
    for img in images[1:]:
        final_image = np.vstack((final_image, spacer, img))

    # imagem_final = cv2.addWeighted(imagem_redimensionada_2, 1.0, imagem_redimensionada_1, 1.0, 0)

    
    timestamp_segundos = int(df['key.frame_timestamp_micros'].iloc[0]) / 1000000
    # Converter o timestamp para um objeto datetime
    data = datetime.fromtimestamp(timestamp_segundos)

    # Formatar a data em um formato legível
    data_formatada = data.strftime('%Y-%m-%d %H:%M:%S')

    # Exibir a data formatada
    print(data_formatada)
    # Exibir a imagem usando OpenCV
    cv2.imshow('LiDAR', final_image)
    cv2.waitKey(1)  # Aguarda até que uma tecla seja pressionada para fechar a janela
    # time.sleep(1)


def dir_process(path_dir):
    # Busca todos os arquivos .parquet no diretório
    files = sorted([os.path.join(path_dir, f) for f in os.listdir(path_dir) if f.endswith('.parquet')])

    for file in files:
        print(f"Processando {file}...")
        process_parquet(file)

if __name__ == "__main__":
    path_dir = '/media/lume/hendrix/datasets/waymo_open_dataset_v_2_0_1/training/lidar/'

    # Carregar os dados existentes do arquivo JSON
    with open('data.json', 'r', encoding='utf-8') as arquivo:
        json_data = json.load(arquivo)

    # Ordenar as chaves pelo valor de 'timestamp'
    sorted_keys = sorted(json_data, key=lambda x: json_data[x]['timestamp'])

    # Imprimir as chaves ordenadas
    for key in sorted_keys:
        process_parquet(os.path.join(path_dir, key + '.parquet'))
    # # Defina o diretório contendo os 
    # # arquivos .parquet
    # path_dir = '/media/lume/hendrix/datasets/waymo_open_dataset_v_2_0_1/training/lidar_renamed_files/'
    
    # # Processa todos os arquivos no diretório
    # dir_process(path_dir) 

