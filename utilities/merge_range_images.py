import os
import re
from PIL import Image

def natural_sort_key(s):
    """
    Ordena corretamente arquivos com números (ex: img2 vem antes de img10).
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def processar_diretorios(root_dir):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    # Caminha por todos os subdiretórios
    for subdir, dirs, files in os.walk(root_dir):
        
        # Pula o próprio diretório raiz para não processar imagens soltas nele
        # (Focamos apenas nos subdiretórios conforme seu pedido)
        if os.path.abspath(subdir) == os.path.abspath(root_dir):
            continue

        # Filtra apenas imagens
        images = [f for f in files if f.lower().endswith(valid_extensions)]
        
        if not images:
            continue

        # Ordena: garante que 0 fique no topo e 11 no fundo
        images.sort(key=natural_sort_key)
        
        print(f"Processando subdiretório: {os.path.basename(subdir)}...")

        try:
            opened_images = []
            for img_name in images:
                img_path = os.path.join(subdir, img_name)
                opened_images.append(Image.open(img_path))

            if not opened_images:
                continue

            # Cálculos de dimensão
            width, height = opened_images[0].size
            total_height = height * len(opened_images)

            # Cria a imagem vertical
            new_im = Image.new('RGB', (width, total_height))

            y_offset = 0
            for img in opened_images:
                new_im.paste(img, (0, y_offset))
                y_offset += img.size[1]
                img.close()

            # --- MUDANÇA AQUI ---
            # Pega o nome da pasta atual (ex: "subfolder_01")
            folder_name = os.path.basename(subdir)
            
            # Define o caminho de salvamento na RAIZ (root_dir) com o nome da pasta
            output_filename = f"{folder_name}.png"
            output_path = os.path.join(root_dir, output_filename)
            
            new_im.save(output_path)
            print(f" -> Arquivo criado na raiz: {output_filename}")

        except Exception as e:
            print(f"Erro ao processar {subdir}: {e}")

if __name__ == "__main__":
    diretorio_raiz = input("Digite o caminho do diretório principal: ").strip()
    diretorio_raiz = diretorio_raiz.replace('"', '').replace("'", "")

    if os.path.isdir(diretorio_raiz):
        print(f"Iniciando processamento em: {diretorio_raiz}\n")
        processar_diretorios(diretorio_raiz)
        print("\nConcluído!")
    else:
        print("O diretório informado não existe.")