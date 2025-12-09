#!/usr/bin/env python3
import argparse
import cv2
from mmpretrain import MAEInferencer
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Rodar reconstrução de imagem com MAE")
    # Renomeado para --config para maior clareza
    parser.add_argument(
        "--config", type=str, required=True,
        help="Caminho para o arquivo de configuração do modelo (.py)"
    )
    # Novo argumento para o checkpoint
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Caminho para o arquivo de checkpoint do modelo (.pth)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Imagem de entrada"
    )
    parser.add_argument(
        "--output", type=str, default="mae_result.png",
        help="Arquivo de saída para salvar a imagem reconstruída"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Dispositivo (cpu ou cuda:0)"
    )
    args = parser.parse_args()

    # --- ALTERAÇÃO PRINCIPAL AQUI ---
    # Inicializa inferencer com a config E o checkpoint
    inferencer = MAEInferencer(
        model=args.config,
        pretrained=args.checkpoint, # Use o argumento 'pretrained'
        device=args.device
    )

    # O resto do seu script continua igual...
    
    # Executa reconstrução
    result_images = inferencer(args.input, copy_inputs=False)
    result_images_with_visible = inferencer(args.input, copy_inputs=True)
    print("[DEBUG] result_images shapes:", [img.shape for img in result_images])
    print("[DEBUG] result_images_with_visible shapes:", [img.shape for img in result_images_with_visible])

    # ... (o restante do seu código para salvar as imagens)
    vis_image = result_images[0]
    vis_image_with_visible = result_images_with_visible[0]
    
    w = vis_image.shape[1] // 3
    original = vis_image[:, :w, :]
    masked = vis_image[:, w:2*w, :]
    recon = vis_image[:, 2*w:3*w, :]
    recon_plus_visible = vis_image_with_visible[:, 2*w:3*w, :]

    concat4 = np.concatenate((original, masked, recon, recon_plus_visible), axis=1)

    cv2.imwrite(args.output.replace('.png', '_original.png'), original)
    cv2.imwrite(args.output.replace('.png', '_masked.png'), masked)
    cv2.imwrite(args.output.replace('.png', '_recon.png'), recon)
    cv2.imwrite(args.output.replace('.png', '_recon_plus_visible.png'), recon_plus_visible)

    cv2.imwrite(args.output, concat4)
    print(f"[OK] Resultado salvo em {args.output} (original, mascarada, reconstruída, reconstrução+visível)")

if __name__ == "__main__":
    main()
