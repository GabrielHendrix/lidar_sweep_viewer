# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

_base_ = [
    '../../_base_/models/mae_vit-base-p16.py',
    '../../_base_/default_runtime.py',
]

# --- CONFIGURAÇÕES CRÍTICAS ---
image_size = 224  # O OBJETIVO FINAL
patch_size = 16
embed_dim = 1024
num_layers = 24
model_name = 'sapiens_0.3b'

# Visualizar e Salvar com frequência
vis_every_iters = 1000 # Como vamos usar acumulação, 1 iteração demora mais
save_every_epochs = 200

num_patches = (image_size // patch_size) ** 2

custom_imports = dict(
    imports=['mmpretrain.datasets', 'mmpretrain.visualization'],
    allow_failed_imports=False
)

# --- PREPROCESSAMENTO ---
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # Resize Fixo para 1024 (Sem Random Crop para garantir Overfit)
    dict(
        type='Resize',
        scale=(image_size, image_size), 
        backend='pillow',
        interpolation='bicubic'),
    dict(type='PackInputs')
]

# --- DATALOADER (OTIMIZADO PARA MEMÓRIA) ---
train_dataloader = dict(
    # [IMPORTANTE] Batch Size 1 para não estourar a VRAM com imagem 1024
    batch_size=2, 
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset',
        data_root='/home/hendrix/Desktop/indir/',
        pipeline=train_pipeline
    )
)

# --- MODELO ---
model = dict(
    backbone=dict(
        type='MAEViT', 
        arch=model_name, 
        patch_size=patch_size, 
        img_size=image_size, 
        final_norm=True, 
        # [ESTRATÉGIA] Comece com 0.25 (25%) mascarado. 
        # É difícil o suficiente para provar que aprendeu, mas fácil o suficiente para não travar.
        mask_ratio=0.75
    ),
    neck=dict(
        type='MAEPretrainDecoder',
        embed_dim=embed_dim,
        patch_size=patch_size,
        num_patches=num_patches),
    head=dict(
        type='MAEPretrainHead',
        patch_size=patch_size,
        # Mantemos FALSE pois funcionou bem no seu teste anterior
        norm_pix=False 
    ))

# --- OTIMIZADOR (A MÁGICA ACONTECE AQUI) ---
optim_wrapper = dict(
    # [IMPORTANTE] AmpOptimWrapper economiza VRAM usando FP16
    type='AmpOptimWrapper', 
    loss_scale='dynamic',
    
    # [IMPORTANTE] Gradient Accumulation:
    # Batch Size real (1) * Accumulative (10) = Batch Virtual (10).
    # Isso estabiliza o treino igual ao teste de 224px, mas com imagem gigante.
    accumulative_counts=10, 
    
    optimizer=dict(
        type='AdamW', 
        lr=1e-4, # Seguro
        weight_decay=0.05, 
    ),
    clip_grad=dict(max_norm=3.0, norm_type=2), 
    
    paramwise_cfg=dict(
        custom_keys={
            'bias': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0)
        }
    )
)

# --- RUNTIME ---
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=save_every_epochs, max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=1), 
    visualization=dict(type='VisualizationHook', enable=True),
)

randomness = dict(seed=0, diff_rank_seed=True)
resume = True
auto_scale_lr = dict(enable=False)

custom_hooks = [
    dict(
        type='PretrainVisualizationHook',
        enable=True,
        vis_every_iters=vis_every_iters,
        vis_max_samples=1, 
    )
]

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

## Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(image_size, image_size),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs'),
]

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     dataset=dict(
#         type='CustomDataset',
#         data_root='/home/hendrix/Desktop/indir/',
#         pipeline=test_pipeline
#     ),
#     persistent_workers=True,
# )