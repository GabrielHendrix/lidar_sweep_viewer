import argparse
import shutil
from pathlib import Path

def remove_scenes(scenes_file, data_path):
    """
    Remove scene directories listed in a file from the specified data subdirectories.

    Args:
        scenes_file (str): Path to the .txt file containing scene names, one per line.
        data_path (str): Path to the root data directory.
    """
    base_path = Path(data_path)
    scenes_path = Path(scenes_file)

    if not scenes_path.is_file():
        print(f"Erro: O arquivo de cenas '{scenes_file}' não foi encontrado.")
        return

    if not base_path.is_dir():
        print(f"Erro: O diretório de dados '{data_path}' não foi encontrado.")
        return

    # Subdiretórios onde as cenas precisam ser removidas
    subdirs_to_clean = ["bin_files", "poses", "objs_bbox", "images"]

    try:
        # Usar um set para lidar com nomes de cenas duplicados automaticamente
        with open(scenes_path, 'r') as f:
            scenes_to_remove = {line.strip() for line in f if line.strip()}
    except IOError as e:
        print(f"Erro ao ler o arquivo de cenas '{scenes_file}': {e}")
        return

    if not scenes_to_remove:
        print("Nenhum nome de cena encontrado no arquivo.")
        return

    print(f"As seguintes cenas serão removidas: {', '.join(scenes_to_remove)}")

    # Iterar sobre cada cena que precisa ser removida
    for scene_name in scenes_to_remove:
        print(f"\nProcessando cena: '{scene_name}'")
        # Iterar sobre os subdiretórios (bin_files, poses, etc.)
        for subdir in subdirs_to_clean:
            scene_dir_path = base_path / subdir / scene_name
            
            if scene_dir_path.is_dir():
                try:
                    shutil.rmtree(scene_dir_path)
                    print(f"  - Removido: '{scene_dir_path}'")
                except OSError as e:
                    print(f"  - Erro ao remover '{scene_dir_path}': {e}")
            else:
                print(f"  - Não encontrado (ou não é um diretório): '{scene_dir_path}'")

def main():
    parser = argparse.ArgumentParser(
        description="Remove diretórios de cenas especificados em um arquivo de texto de várias pastas de dados."
    )
    parser.add_argument(
        "scenes_file",
        help="Caminho para o arquivo .txt contendo os nomes das cenas a serem removidas (um por linha)."
    )
    parser.add_argument(
        "--data_path",
        default=".",
        help="Caminho para o diretório de dados principal que contém as pastas 'bin_files', 'poses', etc. O padrão é o diretório atual."
    )
    
    args = parser.parse_args()
    
    remove_scenes(args.scenes_file, args.data_path)

if __name__ == "__main__":
    main()
