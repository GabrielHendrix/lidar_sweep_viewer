# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Union

import torch
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate

from mmpretrain.registry import TRANSFORMS
from .base import BaseInferencer, InputType
from .model import list_models
import numpy as np

class FeatureExtractor(BaseInferencer):
    """The inferencer for extract features.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``FeatureExtractor.list_models()`` and you can also query it in
            :doc:`/modelzoo_statistics`.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).

    Example:
        >>> from mmpretrain import FeatureExtractor
        >>> inferencer = FeatureExtractor('resnet50_8xb32_in1k', backbone=dict(out_indices=(0, 1, 2, 3)))
        >>> feats = inferencer('demo/demo.JPEG', stage='backbone')[0]
        >>> for feat in feats:
        >>>     print(feat.shape)
        torch.Size([256, 56, 56])
        torch.Size([512, 28, 28])
        torch.Size([1024, 14, 14])
        torch.Size([2048, 7, 7])
    """  # noqa: E501

    def __call__(self,
                 inputs: InputType,
                 batch_size: int = 1,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (str | array | list): The image path or array, or a list of
                images.
            batch_size (int): Batch size. Defaults to 1.
            **kwargs: Other keyword arguments accepted by the `extract_feat`
                method of the model.

        Returns:
            tensor | Tuple[tensor]: The extracted features.
        """
        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(ori_inputs, batch_size=batch_size)
        preds = []
        for data in inputs:
            preds.extend(self.forward(data, **kwargs))

        return preds

    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], **kwargs):
        inputs = self.model.data_preprocessor(inputs, False)['inputs']
        outputs = self.model.extract_feat(inputs, **kwargs)

        def scatter(feats, index):
            if isinstance(feats, torch.Tensor):
                return feats[index]
            else:
                # Sequence of tensor
                return type(feats)([scatter(item, index) for item in feats])

        results = []
        for i in range(inputs.shape[0]):
            results.append(scatter(outputs, i))

        return results

    def _init_pipeline(self, cfg: Config) -> Callable:
        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        from mmpretrain.datasets import remove_transform

        # Image loading is finished in `self.preprocess`.
        test_pipeline_cfg = remove_transform(test_pipeline_cfg,
                                             'LoadImageFromFile')
        test_pipeline = Compose(
            [TRANSFORMS.build(t) for t in test_pipeline_cfg])
        return test_pipeline

    # def preprocess(self, inputs: List[InputType], batch_size: int = 1):
    #     """
    #     Sobrescreve o método original para carregar dados de arquivos .bin.
    #     """
    #     # A função interna que define como carregar um único arquivo.
    #     # Nós a substituímos para usar nossa função 'load_bin_file_as_array'.
    #     def load_bin_logic(input_):
    #         # Carrega o array 3D do arquivo .bin
    #         """
    #         Lê um arquivo binário e o converte para um array NumPy 3D.

    #         !!! ATENÇÃO !!!
    #         Você DEVE alterar os valores de 'shape' e 'dtype' para corresponderem
    #         aos seus dados.
    #         """
    #         # ----- CONFIGURE AQUI -----
    #         # Substitua pelas dimensões corretas do seu dado (ex: Altura, Largura, Canais)
    #         YOUR_SHAPE = (2650, 64, 3) 
    #         # Substitua pelo tipo de dado correto (ex: np.uint8, np.float32, etc.)
    #         YOUR_DTYPE = np.float32    
    #         # --------------------------

    #         # Lê o arquivo binário como um array 1D
    #         data_1d = np.fromfile(input_, dtype=YOUR_DTYPE)

    #         # Remodela para (N, 4) e remove os valores de reflexão para obter (N, 3)
    #         points_2d = data_1d.reshape((-1, 4))[:, 0:3]

    #         # --- AQUI ESTÁ A MUDANÇA ---
    #         # Tenta remodelar o array 2D para o shape 3D desejado
    #         points_3d = points_2d.reshape((2650, 64, 3))
    #         print("Shape do array 2D original:", points_2d.shape)
    #         print("Shape do novo array 3D:", points_3d.shape)


    #         print(points_3d)
    #         # Verifica se a quantidade de dados lidos corresponde ao shape esperado
    #         expected_elements = np.prod(YOUR_SHAPE)
    #         if points_3d.size != expected_elements:
    #             raise ValueError(
    #                 f"Erro ao ler {input_}. O número de elementos no arquivo ({points_3d.size}) "
    #                 f"não corresponde ao shape esperado {YOUR_SHAPE} (total: {expected_elements} elementos)."
    #             )

    #         # Remodela o array para o shape 3D
    #         array_float32 = points_2d.reshape(YOUR_SHAPE)
            
    #         # Multiplicamos todos os valores por 255.
    #         array_escalonado = array_float32 * 255.0

    #         print("--- Array Após Escalonamento ---")
    #         print(array_escalonado)
    #         print("Tipo de dado ainda é:", array_escalonado.dtype)
    #         print("\n")

    #         # 3. PASSO 2: Converta o tipo para uint8
    #         # O .astype() irá truncar a parte decimal (ex: 63.75 vira 63)
    #         array_3d = array_escalonado.astype(np.uint8)
    #         if array_3d is None:
    #             raise ValueError(f'Falha ao ler o arquivo binário {input_}.')
            
    #         # O resto do pipeline espera um dicionário neste formato.
    #         # O 'img' agora é o nosso array vindo do .bin.
    #         return dict(
    #             img=array_3d,
    #             img_shape=array_3d.shape[:2],
    #             ori_shape=array_3d.shape[:2],
    #         )

    #     # O 'self.pipeline' contém as outras transformações (redimensionar, normalizar, etc.)
    #     # que foram definidas no arquivo de configuração do modelo.
    #     pipeline = Compose([load_bin_logic, self.pipeline])

    #     # O resto da função continua como o original
    #     chunked_data = self._get_chunk_data(map(pipeline, inputs), batch_size)
    #     yield from map(default_collate, chunked_data)
    def preprocess(self, inputs: List[InputType], batch_size: int = 1):

        def load_image(input_):
            img = imread(input_)
            if img is None:
                raise ValueError(f'Failed to read image {input_}.')
            return dict(
                img=img,
                img_shape=img.shape[:2],
                ori_shape=img.shape[:2],
            )

        pipeline = Compose([load_image, self.pipeline])

        chunked_data = self._get_chunk_data(map(pipeline, inputs), batch_size)
        yield from map(default_collate, chunked_data)

    def visualize(self):
        raise NotImplementedError(
            "The FeatureExtractor doesn't support visualization.")

    def postprocess(self):
        raise NotImplementedError(
            "The FeatureExtractor doesn't need postprocessing.")

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List all available model names.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern)
