import pandas as pd
import shutil
import json
import os


def rename_files(path_dir, new_path_dir):
    # Busca todos os arquivos .parquet no diretório
    files = [os.path.join(path_dir, f) for f in os.listdir(path_dir) if f.endswith('.parquet')]
    for file in files:
        if (int(str(file.split('/')[-1].split('.')[0])[:1]) >= 4):
            print(f"Processando {file}...")
            df = pd.read_parquet(file)
            new_file_name = os.path.join(new_path_dir, str(df['key.frame_timestamp_micros'].iloc[0]) + '.parquet')
            caminho_arquivo = "data.json"
            # with open("data.json", "w") as outfile: 
            #     json_string = json.dumps(string, indent=4)
            #     print(json_string)
            #     json.dump(string, outfile)
            dictionary = {str(file.split('/')[-1].split('.')[0]): {'timestamp':  str(df['key.frame_timestamp_micros'].iloc[0])}}

            # Carregar os dados existentes do arquivo JSON
            with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
                dados = json.load(arquivo)

            # Atualizar o dicionário existente com as novas chaves
            if isinstance(dados, dict):
                dados.update(dictionary)
            else:
                print("Erro: O conteúdo do JSON não é um dicionário.")

            # Salvar os dados atualizados de volta ao arquivo JSON
            with open(caminho_arquivo, 'w', encoding='utf-8') as arquivo:
                json.dump(dados, arquivo, indent=4, ensure_ascii=False)
            # if not (os.path.isfile(new_file_name)):
                # shutil.copy(file, new_file_name)
                # os.rename(file, new_file_name)

if __name__ == "__main__":
    # Defina o diretório contendo os arquivos .parquet e caminho para o novo diretorio onde serao salvos os arquivos renomeados.
    new_path_dir = '/media/lume/hendrix/datasets/waymo_open_dataset_v_2_0_1/training/lidar_renamed_files/'
    path_dir = '/media/lume/hendrix/datasets/waymo_open_dataset_v_2_0_1/training/lidar/'


    # Processar todos os arquivos no diretório
    rename_files(path_dir, new_path_dir) 