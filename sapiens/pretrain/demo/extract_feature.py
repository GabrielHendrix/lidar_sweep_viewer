# import cv2
# import mmcv
# import mmengine
# import numpy as np
# from tqdm import tqdm
# import warnings
# from argparse import ArgumentParser
# from mmpretrain import FeatureExtractor
# import os

# def main():
#     parser = ArgumentParser()
#     parser.add_argument('config', help='Config file for pose')
#     parser.add_argument('checkpoint', help='Checkpoint file for pose')
#     parser.add_argument(
#         '--input', type=str, default='', help='Image/Video file')
#     parser.add_argument(
#         '--output-root',
#         type=str,
#         default='',
#         help='root of the output img file. '
#         'Default not saving the visualization images.')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')

#     args = parser.parse_args()

#     assert args.output_root != ''
#     assert args.input != ''

#     if args.output_root:
#         mmengine.mkdir_or_exist(args.output_root)

#     inferencer = FeatureExtractor(model=args.config, pretrained=args.checkpoint, device=args.device)
#     inferencer.model.backbone.out_type = 'featmap'

#     input_path = args.input
#     image_names = []
#     input_dir = ''

#     if os.path.isdir(input_path):
#         input_dir = input_path
#         image_names = [image_name for image_name in sorted(os.listdir(input_dir))
#                     if image_name.endswith(('.jpg', '.jpeg', '.png'))]
#     elif os.path.isfile(input_path) and input_path.endswith('.txt'):
#         with open(input_path, 'r') as file:
#             image_paths = [line.strip() for line in file if line.strip()]
#         image_names = [os.path.basename(path) for path in image_paths]
#         input_dir = os.path.dirname(image_paths[0]) if image_paths else ''

#     TARGET_HEIGHT = 1024
#     WHITE_COLOR = [255, 255, 255] # Cor branca em BGR (padrão do OpenCV)

#     for image_name in tqdm(image_names, total=len(image_names)):
#         image_path = os.path.join(input_dir, image_name)
        
#         try:
#             full_image = mmcv.imread(image_path)
#             if full_image is None:
#                 warnings.warn(f'Could not read image {image_path}. Skipping.')
#                 continue
            
#             # ==================== INÍCIO DAS MODIFICAÇÕES ====================

#             height, width, _ = full_image.shape
            
#             # 1. Calcular o padding necessário
#             if height < TARGET_HEIGHT:
#                 pad_total = TARGET_HEIGHT - height
#                 # Divisão inteira para garantir pixels inteiros
#                 pad_top = pad_total // 2
#                 # O restante vai para baixo para lidar com casos ímpares
#                 pad_bottom = pad_total - pad_top
                
#                 # 2. Aplicar o padding com a cor branca
#                 padded_image = cv2.copyMakeBorder(
#                     full_image, 
#                     pad_top, 
#                     pad_bottom, 
#                     0, # Sem padding na esquerda
#                     0, # Sem padding na direita
#                     cv2.BORDER_CONSTANT, 
#                     value=WHITE_COLOR
#                 )
#             else:
#                 # Se a imagem já for maior ou igual, usa a original
#                 padded_image = full_image

#             # 3. Executar a inferência na imagem inteira com padding
#             feature = inferencer(padded_image)[0][0]
#             feature_np = feature.cpu().numpy()

#             # 4. Salvar o mapa de features 3D resultante
#             output_file_path = os.path.join(args.output_root, os.path.basename(image_path))
#             pred_save_path = output_file_path.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')
            
#             np.save(pred_save_path, feature_np)
            
#             # ==================== FIM DAS MODIFICAÇÕES ====================

#         except Exception as e:
#             warnings.warn(f'Error processing image {image_path}: {e}. Skipping.')
#             continue

# if __name__ == '__main__':
#     main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmpretrain import FeatureExtractor
import os
import time
from argparse import ArgumentParser

import cv2
import mmcv
import mmengine
import numpy as np
from tqdm import tqdm
import warnings

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file for pose')
    parser.add_argument('checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    assert args.output_root != ''
    assert args.input != ''

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))

    inferencer = FeatureExtractor(model=args.config, pretrained=args.checkpoint, device=args.device)
    inferencer.model.backbone.out_type = 'featmap' ## removes cls_token and returns spatial feature maps.
    input = args.input
    image_names = []
    
    # # Lista os arquivos .bin
    # bin_files = [os.path.join(input, f) for f in os.listdir(input) if f.endswith('.bin')]
    
    # for bin_path in tqdm(bin_files):
    #     # O __call__ da classe vai invocar nosso método 'preprocess' modificado
    #     # A extração retorna uma lista, pegamos o primeiro (e único) elemento
    #     # feature = inferencer(bin_path)[0]
        
    #     # Converte para numpy se for um tensor
    #     # if hasattr(feature, 'cpu'):
    #     feature = inferencer(bin_path)[0][0] ## embed_dim x H x W. For sapien_1b: 1536 x 64 x 64
    #     feature = feature.cpu().numpy()
            
    #     # Salva o resultado
    #     output_filename = os.path.basename(bin_path).replace('.bin', '.npy')
    #     save_path = os.path.join(output_file, output_filename)
    #     np.save(save_path, feature)
        
    # print("Extração de features concluída!")

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [image_name for image_name in sorted(os.listdir(input_dir))
                    if image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png')]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]  # Extract base names for image processing
        input_dir = os.path.dirname(image_paths[0]) if image_paths else ''  # Use the directory of the first image path

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)  # Join the directory path with the image file name 
        feature = inferencer(image_path)[0][0] ## embed_dim x H x W. For sapien_1b: 1536 x 64 x 64
        feature = feature.cpu().numpy()

        output_file = os.path.join(args.output_root, os.path.basename(image_path))
        pred_save_path = os.path.join(output_file.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy'))
        np.save(pred_save_path, feature)


if __name__ == '__main__':
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from mmpretrain import FeatureExtractor
# import os
# import time
# from argparse import ArgumentParser

# import cv2
# import mmcv
# import mmengine
# import numpy as np
# from tqdm import tqdm
# import warnings

# def main():
#     parser = ArgumentParser()
#     parser.add_argument('config', help='Config file for pose')
#     parser.add_argument('checkpoint', help='Checkpoint file for pose')
#     parser.add_argument(
#         '--input', type=str, default='', help='Image/Video file')
#     parser.add_argument(
#         '--output-root',
#         type=str,
#         default='',
#         help='root of the output img file. '
#         'Default not saving the visualization images.')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')

#     args = parser.parse_args()

#     assert args.output_root != ''
#     assert args.input != ''

#     output_file = None
#     if args.output_root:
#         mmengine.mkdir_or_exist(args.output_root)
#         output_file = os.path.join(args.output_root,
#                                    os.path.basename(args.input))

#     inferencer = FeatureExtractor(model=args.config, pretrained=args.checkpoint, device=args.device)
#     inferencer.model.backbone.out_type = 'featmap' ## removes cls_token and returns spatial feature maps.

#     input_path = args.input
#     image_names = []
#     input_dir = ''

#     # Check if the input is a directory or a text file
#     if os.path.isdir(input_path):
#         input_dir = input_path
#         image_names = [image_name for image_name in sorted(os.listdir(input_dir))
#                     if image_name.endswith(('.jpg', '.jpeg', '.png'))]
#     elif os.path.isfile(input_path) and input_path.endswith('.txt'):
#         # If the input is a text file, read the paths from it
#         with open(input_path, 'r') as file:
#             image_paths = [line.strip() for line in file if line.strip()]
#         image_names = [os.path.basename(path) for path in image_paths]
#         input_dir = os.path.dirname(image_paths[0]) if image_paths else ''

#     # ==================== INÍCIO DAS MODIFICAÇÕES ====================

#     CROP_SIZE = 32  # Definimos o tamanho do crop

#     for image_name in tqdm(image_names, total=len(image_names)):
#         image_path = os.path.join(input_dir, image_name)
        
#         # 1. Carregar a imagem inteira usando mmcv
#         try:
#             full_image = mmcv.imread(image_path)
#             if full_image is None:
#                 warnings.warn(f'Could not read image {image_path}. Skipping.')
#                 continue
#             height, width, _ = full_image.shape
#         except Exception as e:
#             warnings.warn(f'Error loading image {image_path}: {e}. Skipping.')
#             continue

#         # 2. Lista para armazenar as features de todos os crops desta imagem
#         features_from_crops = []

#         # 3. Loop para extrair e processar cada crop de 32x32
#         for x_start in range(0, width, CROP_SIZE):
#             x_end = x_start + CROP_SIZE
            
#             # Garante que não passaremos do limite da imagem
#             if x_end > width:
#                 continue

#             # Extrai o crop
#             image_crop = full_image[0:CROP_SIZE, x_start:x_end]

#             # 4. Executa a inferência no crop
#             # O inferencer aceita um array numpy diretamente
#             feature_crop = inferencer(image_crop)[0][0]
            
#             # Adiciona a feature extraída (convertida para numpy) na lista
#             features_from_crops.append(feature_crop.cpu().numpy())

#         # 5. Se alguma feature foi extraída, empilha todas em um único array
#         if features_from_crops:
#             # final_feature_stack = np.stack(features_from_crops, axis=0)
#             # Linha nova
#             final_feature_map_stitched = np.concatenate(features_from_crops, axis=2)
#             # 6. Define o caminho de saída e salva o array .npy final
#             output_file_path = os.path.join(args.output_root, os.path.basename(image_path))
#             pred_save_path = output_file_path.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')
#             np.save(pred_save_path, final_feature_map_stitched)
            
#     # ==================== FIM DAS MODIFICAÇÕES ====================

# if __name__ == '__main__':
#     main()