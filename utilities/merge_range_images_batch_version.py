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

def processar_diretorios(root_dir, batch_size):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
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
            # --- MUDANÇA: Loop para processar em lotes (chunks) de 16 em 16 ---
            for i in range(0, len(images), batch_size):
                # Pega o fatiamento (slice) da lista. Ex: 0 a 16, 16 a 32...
                batch_files = images[i : i + batch_size]
                
                # Índice atual para o nome do arquivo (0, 1, 2...)
                batch_index = i // batch_size

                opened_images = []
                for img_name in batch_files:
                    img_path = os.path.join(subdir, img_name)
                    opened_images.append(Image.open(img_path))

                if not opened_images:
                    continue

                # Cálculos de dimensão baseados na primeira imagem do lote
                width, height = opened_images[0].size
                total_height = height * len(opened_images)

                # Cria a imagem vertical
                new_im = Image.new('RGB', (width, total_height))

                y_offset = 0
                for img in opened_images:
                    new_im.paste(img, (0, y_offset))
                    y_offset += img.size[1]
                    img.close() # Fecha a imagem individual para liberar memória

                # --- MUDANÇA: Nome do arquivo com índice ---
                folder_name = os.path.basename(subdir)
                
                # Exemplo de saída: subfolder_01_0.png, subfolder_01_1.png
                output_filename = f"{folder_name}_{batch_index}.png"
                output_path = os.path.join(root_dir, output_filename)
                
                new_im.save(output_path)
                print(f" -> Arquivo criado na raiz: {output_filename} ({len(batch_files)} imagens unidas)")

        except Exception as e:
            print(f"Erro ao processar {subdir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Waymo Range files concatenator')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Pasta com .tfrecords')
    parser.add_argument('-b', '--batch_size', type=str, required=True, help='Configuração: Quantas imagens fundir por vez')
    args = parser.parse_args()

    if os.path.isdir(args.input_dir):
        print(f"Iniciando processamento em lotes de {args.batch_size} imagens...\n")
        processar_diretorios(args.input_dir, int(args.batch_size))
        print("\nConcluído!")
    else:
        print("O diretório informado não existe.")