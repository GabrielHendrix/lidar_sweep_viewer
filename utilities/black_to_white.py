import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 replace_white_to_black_plt_stats.py <imagem_entrada>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Lê a imagem (float no intervalo [0,1])
    img = mpimg.imread(input_path)

    # Se tiver canal alfa, separa só RGB
    if img.shape[-1] == 4:
        rgb = img[..., :3]
    else:
        rgb = img

    # Máscara de pixels pretos puros
    mask_black = np.all(rgb == 0.0, axis=-1)

    # Substitui preto por branco
    rgb[mask_black] = 1.0

    # Atualiza imagem original (mantendo alfa, se houver)
    if img.shape[-1] == 4:
        img[..., :3] = rgb
    else:
        img = rgb

    # Máscara para ignorar branco (1,1,1)
    mask_non_white = np.any(rgb < 1.0, axis=-1)

    # Seleciona apenas pixels válidos (não pretos)
    valid_pixels = rgb[mask_non_white]

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

    # Salva imagem de saída
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_no_black{ext}"
    plt.imsave(output_path, img)

    print(f"\nImagem processada salva em: {output_path}")

if __name__ == "__main__":
    main()
