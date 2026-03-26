import os
import re
import argparse
from PIL import Image

def natural_sort_key(s):
    """
    Ordena corretamente arquivos com números (ex: img2 vem antes de img10).
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def processar_diretorios(root_dir):
    """
    Processa subdiretórios para criar imagens compostas.
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
     
    # --- CONFIGURAÇÕES FIXAS ---
    IMAGES_PER_BATCH = 41
    FINAL_WIDTH = 2650
    FINAL_HEIGHT = 2650

    # Caminha por todos os subdiretórios
    for subdir, dirs, files in os.walk(root_dir):
        
        # Pula o próprio diretório raiz
        if os.path.abspath(subdir) == os.path.abspath(root_dir):
            continue

        # Filtra apenas imagens
        images = [f for f in files if f.lower().endswith(valid_extensions)]
        
        if not images:
            continue

        # Ordena: garante a sequência correta
        images.sort(key=natural_sort_key)
        
        print(f"Processando subdiretório: {os.path.basename(subdir)} ({len(images)} imagens)...")

        try:
            # Processa em lotes de 41 imagens
            for i in range(0, len(images), IMAGES_PER_BATCH):
                batch_files = images[i : i + IMAGES_PER_BATCH]

                # Pula lotes que não têm o número exato de imagens
                if len(batch_files) != IMAGES_PER_BATCH:
                    print(f" -> Lote ignorado: número de imagens ({len(batch_files)}) é menor que {IMAGES_PER_BATCH}.")
                    continue
                
                # Índice atual para o nome do arquivo (0, 1, 2...)
                batch_index = i // IMAGES_PER_BATCH

                opened_images = []
                for img_name in batch_files:
                    img_path = os.path.join(subdir, img_name)
                    opened_images.append(Image.open(img_path))

                if not opened_images:
                    continue

                # Pega a altura da primeira imagem para o cálculo do offset
                img_height = opened_images[0].size[1]
                
                # Cria a imagem final com padding (fundo preto)
                new_im = Image.new('RGB', (FINAL_WIDTH, FINAL_HEIGHT), 'black')

                y_offset = 0
                for img in opened_images:
                    new_im.paste(img, (0, y_offset))
                    y_offset += img_height
                    img.close()

                # --- MUDANÇA: Nome do arquivo com índice ---
                folder_name = os.path.basename(subdir)
                
                # Exemplo de saída: subfolder_01_0.png, subfolder_01_1.png
                output_filename = f"{folder_name}_{batch_index}.png"
                output_path = os.path.join(root_dir, output_filename)
                
                new_im.save(output_path)
                print(f" -> Arquivo criado na raiz: {output_filename} ({len(batch_files)} imagens, {FINAL_WIDTH}x{FINAL_HEIGHT})")

        except Exception as e:
            print(f"Erro ao processar {subdir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Waymo Range files concatenator')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Pasta com os subdiretórios de imagens')
    args = parser.parse_args()

    if os.path.isdir(args.input_dir):
        print(f"Iniciando processamento com lotes de 41 imagens e padding...\n")
        processar_diretorios(args.input_dir)
        print("\nConcluído!")
    else:
        print("O diretório informado não existe.")