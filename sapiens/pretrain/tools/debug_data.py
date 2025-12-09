import numpy as np
from mmengine.config import Config
from mmpretrain.datasets import CustomDataset
from PIL import Image
import os


print(">>> Carregando configuração...")
# --- SUBSTITUA AQUI PELO NOME DO SEU ARQUIVO DE CONFIG ---
cfg = Config.fromfile('../configs/sapiens_mae/imagenet/mae_sapiens_0.3b-p16_8xb512-coslr-1600e_imagenet.py') # <--- Coloque o nome do seu arquivo aqui
# ---------------------------------------------------------

# --- CONFIGURAR PIPELINE "CRU" ---
# Removemos o Resize para pegar a resolução real
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs')
]

# --- CARREGAR DATASET ---
dataset = CustomDataset(
    data_root=cfg.train_dataloader.dataset.data_root,
    pipeline=cfg.train_pipeline
)

# --- PEGAR IMAGEM E SALVAR ---
if len(dataset) > 0:
    # Pega o tensor [C, H, W]
    inputs = dataset[0]['inputs'] 
    
    # Converte para Numpy [H, W, C]
    img_array = inputs.permute(1, 2, 0).numpy()
    
    # MMCV carrega em BGR, converter para RGB
    img_array = img_array[..., ::-1]
    
    # Garante uint8
    if img_array.dtype != 'uint8':
        img_array = img_array.astype('uint8')

    # Salva na resolução nativa usando PIL
    im = Image.fromarray(img_array)
    output_path = "debug_real_resolution.png"
    im.save(output_path)
    
    print(f"Salvo: {output_path} | Resolução: {im.size}")
else:
    print("Dataset vazio.")