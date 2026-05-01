# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

_base_ = [
    '../../_base_/models/mae_vit-base-p16.py',
    '../../_base_/default_runtime.py',
]

# --- CONFIGURAÇÕES GERAIS ---
patch_height = 16
patch_width =  25 # ou o valor desejado
image_height = 1024
image_width = 2650

# Visualizar e Salvar com frequência para monitorar o teste
vis_every_iters = 1000
save_every_epochs = 200

# Modelo Escolhido
model_name = 'sapiens_0.3b'; embed_dim=1024; num_layers=24 # <--- CORRIGIDO: Garante o uso do modelo 0.3b
# model_name = 'sapiens_0.6b'; embed_dim=1280; num_layers=32
# model_name = 'sapiens_1b'; embed_dim=1536; num_layers=40
# model_name = 'sapiens_2b'; embed_dim=1920; num_layers=48 # <--- Comentado para desativar

num_patches = (image_height // patch_height) * (image_width // patch_width)

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
    # SUBSTITUIR RandomResizedCrop por Resize fixo
    dict(
        type='Resize',
        scale=(image_width, image_height), 
        # scale=(image_size, image_size), # Força tamanho exato 1024x1024
        backend='pillow',
        interpolation='bicubic'),
    dict(type='PackInputs')
]

# [ALTERADO] Dataloader para ler pasta local e Batch pequeno
train_dataloader = dict(
    batch_size=10, # <--- Ideal: tamanho do batch igual ao do dataset para overfitting
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True), # Shuffle False ajuda no debug visual
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset', # <--- Usa pasta genérica de imagens
        data_root='/dados/hendrix/customDataRet1000', # <--- Certifique-se que suas 10 imagens estão aqui
        pipeline=train_pipeline
    )
)

# --- MODELO ---
model = dict(
    backbone=dict(
        type='MAEViT', 
        arch=model_name, 
        patch_height=patch_height,
        patch_width=patch_width,
        img_width=image_width, 
        img_height=image_height, 
        final_norm=True, 
        mask_ratio=0.5 # Baixo para facilitar, mas > 0 para evitar erro de cálculo
    ),
    neck=dict(
        type='MAEPretrainDecoder',
        embed_dim=embed_dim,
        img_width=image_width, 
        img_height=image_height,
        patch_height=patch_height,
        patch_width=patch_width,
        num_patches=num_patches),
    head=dict(
        type='MAEPretrainHead',
        img_width=image_width, 
        img_height=image_height,
        patch_height=patch_height,
        patch_width=patch_width,
        norm_pix=False # Desligado para evitar divisão por zero em patches lisos
    ))

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4, # Valor estável para batch pequeno
        betas=(0.9, 0.95),
        weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, error_if_nonfinite=True), ## clip gradients
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# --- AJUSTE DO SCHEDULER PARA OVERFIT ---
param_scheduler = [
    # 1. WARMUP: 50 épocas de subida gradual
    # Como você tem 2 iterações por época, o warmup dura 100 iterações.
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True),
    
    # 2. DECAIMENTO: Do fim do warmup até o final do treino
    dict(
        type='CosineAnnealingLR',
        # T_max deve ser o TOTAL de iterações restantes: (2500 - 50) * 2 = 4900
        T_max=17970, 
        by_epoch=True,
        begin=30,    # Começa logo após o warmup
        end=18000,   # Vai até a última época
        convert_to_iter_based=True)
]

# --- RUNTIME ---
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=18000) # <--- AUMENTADO para garantir a convergência

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=save_every_epochs, max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=1), # Log a cada iteração para ver o loss caindo
    visualization=dict(type='VisualizationHook', enable=True),
)

randomness = dict(seed=0, diff_rank_seed=True)
resume = True

# [ALTERADO] Desligar auto-scale para respeitar nosso LR de 1e-4
auto_scale_lr = dict(enable=False)

custom_hooks = [
    dict(
        type='PretrainVisualizationHook',
        enable=True,
        vis_every_iters=vis_every_iters,
        vis_max_samples=4, # Visualiza o batch todo (se for 4)
    )
]

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# -- CONFIGURAÇÃO DO VISUALIZER (WANDB) --
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),  # Mantém os logs locais
        dict(
            type='WandbVisBackend',    # Ativa o backend do Weights & Biases
            init_kwargs=dict(
                project='SapiensOverfit',  # Nome do seu projeto no W&B
                entity='hendrix_research',  # <-- ADICIONADO: Troque pelo seu time/usuário
                # Opcional: nome específico para esta execução. Pode ser baseado na data/hora.
                # name='1-2650x2650_pixels_patch_50x50_lr1e-4_batch10' 
                name='0-1000files_2650x1024_pixels_patch_25x16_lr1e-4_batch10' 
            )
        )
    ]
)

# ## for dummy testing
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='Resize',
#         scale=image_size,
#         interpolation='bicubic',
#         backend='pillow'),
#     dict(type='PackInputs'),
# ]
