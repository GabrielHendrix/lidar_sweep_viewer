import cv2
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 replace_white_to_black.py <imagem_entrada>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Lê a imagem
    img = cv2.imread(input_path)

    if img is None:
        print(f"Erro: não foi possível abrir {input_path}")
        sys.exit(1)

    # Cria uma máscara onde os pixels são totalmente brancos (255,255,255)
    mask = np.all(img == [255, 255, 255], axis=-1)

    # Altera os pixels brancos para preto
    img[mask] = [0, 0, 0]

    # Gera nome de saída automaticamente
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_no_white{ext}"

    # Salva a imagem modificada
    cv2.imwrite(output_path, img)

    print(f"Imagem processada salva em: {output_path}")

if __name__ == "__main__":
    main()
