# Lidar Sweep Viewer

> Technical Responsible: Gabriel Hendrix

> Module Classification: <mark style="background-color: green;color: white;">Interface</mark>

## Functional Specification

This module captures the lidar messages to create a runtime visualization of the distance matrix of each point in the message.


## Como usar

### 🔧 Preparando o dataset:

<!-- Primeiro baixe um dos exemplos através dos links abaixo e salve no diretório de sua preferência. -->

Primeiro baixe um dos exemplos através desses links ([waymo_10](https://drive.google.com/file/d/1aWKEYJjrxpPB-2yiydCXrWggNagqJalC/view?usp=sharing) | [waymo_100](https://drive.google.com/file/d/1xc-psc0x9XwQkRCSAYMdBdzn4T5E6mqR/view?usp=sharing) | [waymo_1000](https://drive.google.com/file/d/116SWcZFknjDkGp69xnYtmc0SkM_w_7-4/view?usp=sharing)) e salve no diretório de preferência em sua máquina.

Agora descompacte o arquivo baixado no mesmo diretório:

```
tar -xf waymo_x.tar.gz
```

### 📂 Esturura de diretórios e arquivos:

```
/path/to/data/

├── waymo_x/
|   ├── bin_files/           # Contém arquivos binários (.bin) com dados LiDAR
│   │   ├── cena1/
│   │   │   ├── 0.bin
│   │   │   ├── 1.bin
│   │   │   └── ...
│   │   └── cena2/
│   │       └── ...
|   ├── poses/               # Contém arquivos de pose (.txt) por cena
│   │   ├── cena1/
│   │   │   ├── 0.txt
│   │   │   ├── 1.txt
│   │   │   └── ...
│   │   └── cena2/
│   │       └── ...
|   ├── objs_bbox/           # Contém arquivos com bounding boxes
│   │   ├── cena1/
│   │   │   ├── 0.txt
│   │   │   ├── 1.txt
│   │   │   └── ...
│   │   └── cena2/
│   │       └── ...
```

### 📥 Estrutura dos Arquivos de Entrada

#### .bin (LiDAR)

Cada arquivo .bin contém pontos LiDAR sequenciais. Cada ponto está estruturado da seguinte forma:

- float x
- float y
- float z
- float intensidade ou range

Cada arquivo representa uma lambida do sensor onde são registrados seus 2650 tiros de cada um dos seus 64 raios. Os pontos são carregados em uma estrutura std::vector<lidar_point>.

#### .txt (Pose)

Cada arquivo de pose contém uma matriz 4x4 (em formato de texto), representando a pose da câmera/sensor (transformação 3D):

#### .txt (Bounding Boxes)

Cada arquivo de bounding boxes contém uma ou mais caixas 3D. A estrutura pode variar, mas geralmente consiste em:

- Lista de vértices de bounding boxes
- Cada linha: 8 vértices ou parâmetros de centro/dimensão/orientação

Esses dados são transformados conforme a pose e renderizados sobre a imagem birdview.

### 🔧 Executando show_point_cloud:

Para visualizar os dados basta executar os seguintes comandos:

```
cd /path/to/repo/lidar_sweep_viewer
make
./show_point_cloud --input /path/to/data/lidar_sweep_viewer/waymo_1000/ -v 100 (velocidade em ms, vocẽ deve aumentar para poucos dados. Padrão é 1)
```

É possível desativar o calculo, na range_image, dos pontos que estão dentro dos bbox3D de objetos detectados com a seguinte flag:

```
./show_point_cloud --input /path/to/data/lidar_sweep_viewer/waymo_1000/ -no_red
```

E tambem podemos desativar o desenho das imagens, mantendo apenas o carregamento dos dados:

```
./show_point_cloud --input /path/to/data/lidar_sweep_viewer/waymo_1000/ -no_show
```

### ⌨ Controles Interativos

Durante a visualização:
- Espaço: Pausa ou continua.
- ESC: Encerra o programa.


### 🕒 Medição de Desempenho

O código utiliza std::chrono para medir:
<!-- - O tempo de processamento de cada sweep (par de arquivos bin e pose). -->
- Um tempo global médio por sweep.

O tempo é exibido ao final de cada cena.

## 큐 Voxel Representation

O submódulo `voxel_representation` oferece ferramentas para converter nuvens de pontos LiDAR em uma representação baseada em voxels e gerar imagens Bird's-Eye View (BEV) a partir delas. Este processo é útil para criar uma visão 2D de cima para baixo do ambiente 3D capturado pelo LiDAR.
<!--
Existem duas implementações disponíveis: uma em C++ e outra em Python.

#### Implementação em Python

A versão em Python (`voxel_representation.py`) utiliza as bibliotecas `open3d` e `numpy` para realizar as seguintes operações:

1.  **Carregar Nuvem de Pontos**: Carrega os dados de um arquivo `.bin`.
2.  **Correção do Plano do Solo**: Usa RANSAC para identificar e nivelar o solo na nuvem de pontos.
3.  **Voxelização**: Converte a nuvem de pontos corrigida em uma grade de voxels.
4.  **Geração de BEV**: Cria uma imagem BEV (`bird_eye_view_voxels.png`) a partir da grade de voxels.
5.  **Visualização 3D**: Mostra uma visualização 3D dos voxels.

**Como usar (Python):**

O script `voxel_representation.py` é projetado para ser executado diretamente. Ele contém uma função `main` que demonstra o fluxo de ponta a ponta.

```bash
python3 voxel_representation.py
```

Você pode modificar os parâmetros dentro da função `main`, como o caminho para o arquivo `.bin`, o tamanho do voxel e os intervalos de filtro, para se adequar aos seus dados.
#### Implementação em C++

A versão em C++ (`voxel_representation.hpp`, `voxel_representation.cpp`) fornece funções para integrar a geração de BEV em outras aplicações C++. As principais funcionalidades incluem:

-   `find_and_correct_ground_plane`: Corrige a inclinação da nuvem de pontos.
-   `generate_bev_from_points`: Gera um mapa de altura BEV a partir dos pontos.
-   `load_and_decode_bev_image`: Carrega e decodifica uma imagem BEV.
-   `create_encoded_bev_from_height_map`: Codifica um mapa de altura em uma imagem.
-   `load_bin_file`: Carrega uma nuvem de pontos de um arquivo `.bin`.


## Funcionalidades do `voxel_representation.cpp`

O executável `voxel_representation` (localizado em `src/lidar_sweep_viewer/`) é uma ferramenta de linha de comando para processar e visualizar nuvens de pontos 3D, tipicamente obtidas de sensores LiDAR.
-->

### Objetivo Principal

A funcionalidade central é converter dados de nuvens de pontos 3D (de arquivos `.bin`) em uma representação 2D vista de topo, conhecida como **Bird's-Eye View (BEV)** ou "mapa de altura". Subsequentemente, ele renderiza esta representação em um visualizador 3D interativo usando OpenGL.

### Modos de Operação e Recursos

O programa pode ser executado com diferentes modos (`--mode`):

1.  **Modo `pillars` (Padrão):**
    *   Este é o modo principal de visualização.
    *   Ele gera um mapa de altura 2D a partir da nuvem de pontos e depois o "extruda" verticalmente para criar uma cena 3D composta por "pilares" ou colunas. A altura de cada pilar corresponde à altura máxima detectada naquela posição.
    *   Esta é uma técnica de representação de obstáculos muito comum em robótica e sistemas de direção autônoma.

2.  **Modo `points`:**
    *   Neste modo, a nuvem de pontos é primeiramente simplificada em uma grade de "voxels" (pixels 3D).
    *   A cena 3D é então renderizada mostrando os centroides de cada voxel como cubos coloridos, onde a cor representa a altura.

3.  **Modo `image`:**
    *   Carrega um mapa de altura que foi previamente salvo como uma imagem PNG codificada.
    *   Reconstrói a cena 3D no formato de pilares a partir desta imagem, permitindo uma recriação rápida de visualizações salvas.

### Etapas do Processamento

O fluxo de trabalho do programa geralmente segue estes passos:

1.  **Carregamento de Dados:** Carrega uma nuvem de pontos de um arquivo `.bin`. O sistema é capaz de processar sequências de arquivos para visualizar cenas dinâmicas.
2.  **Correção do Plano de Solo:** Utiliza o algoritmo RANSAC para identificar o plano do solo na cena. Em seguida, aplica uma transformação (rotação e translação) para nivelar a nuvem de pontos, garantindo que o solo esteja em Z=0.
3.  **Geração do BEV (Mapa de Altura):** Projeta a nuvem de pontos corrigida em uma grade 2D. Cada célula da grade armazena o valor máximo de altura (Z) dos pontos que caem sobre ela.
4.  **Codificação e Coloração:** O mapa de altura é colorido usando diferentes paletas de cores (ex: 'jet', 'turbo', 'viridis') para facilitar a visualização. O programa também pode codificar este mapa de altura em uma imagem PNG para uso posterior.
5.  **Visualização 3D:** Renderiza a representação final (pilares ou voxels) em uma janela interativa com OpenGL, onde o usuário pode navegar (zoom, rotação, pan), pausar e continuar a visualização.

### Dependências

- **PCL (Point Cloud Library):** Para processamento de nuvens de pontos, filtragem e segmentação RANSAC.
- **OpenCV:** Para manipulação de imagens, criação e coloração de mapas de altura.
- **Eigen:** Para operações de álgebra linear.
- **OpenGL / GLEW / GLFW:** Para a renderização e visualização 3D.

**Como usar (C++):**

As funções no arquivo de cabeçalho `voxel_representation.hpp` podem ser incluídas e utilizadas em seu próprio código C++. Você precisará apenas compilar e rodar ou vincular aos seus projetos, garantindo que as dependências (como PCL e OpenCV) estejam configuradas corretamente. Consulte o código-fonte para obter detalhes sobre os parâmetros da função.


```bash
./voxel_representation --mode [pillars/voxels] --input /path/to/data/lidar_sweep_viewer/waymo_1000/ --color [viridis/plasma/inferno/jet/hot/turbo] -v 100 (velocidade em ms, vocẽ deve aumentar para poucos dados. Padrão é 1)
```

Execução default (pillars):

```bash
./voxel_representation --input /path/to/data/lidar_sweep_viewer/waymo_1000/ 
```

<!-- ### 🔧 Running lidar_sweep_visualization:

Make sure lidar messages are available, Lidar Sweep Viewer uses these messages.

Usage: ./lidar_sweep_visualization -lidar <message_number>

Example:
```
./lidar_sweep_visualization -lidar 1
``` -->

<!--     
### 🔧 Running tfrecord_matrix_viewer (Waymo Motion Dataset):

First, it will be necessary to create and configure a virtual environment.

If you don't have virtualenv yet, install it with:

```
sudo apt install virtualenv
``` 

We will also need Python 3.9:

```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9
```

Now, you need to create the virtual environment (venv) inside the directory related to the module:

```
cd /home/lume/astro/src/lidar_sweep_viewer
virtualenv --python=/usr/bin/python3.9 venv
source venv/bin/activate
venv/bin/python3 -m pip install --upgrade pip
```

Install the necessary libraries to run the .py script:

```
venv/bin/python3 -m pip install -r requirements.txt
sed -i 's/range_image\[..., 0\] > 0/range_image[..., 0]/' venv/lib/python3.9/site-packages/waymo_open_dataset/utils/womd_lidar_utils.py
```

Download the dataset tfrecords with 100 files by clicking [here](https://drive.google.com/file/d/1ATkr_jx8OwFPnwHCFExWxYvFMb6wX3UN/view?usp=sharing):

Download the dataset a example scenario by clicking [here](https://drive.google.com/file/d/1H92_tW1_cVdDwcgxyHVnCYCyUZzZxcBN/view?usp=drive_link):

And now use the script tfrecord_matrix_viewer to visualize the examples:

```
source venv/bin/activate
python3 show_rangeview_and_birdview.py -tf /path/to/tfrecords -s /path/to/scenario_file
```

Example:

```
source venv/bin/activate
python3 show_rangeview_and_birdview.py -tf /dados/lidar_sweep_viewer/examples100 -s /dados/lidar_sweep_viewer/testing.tfrecord-00000-of-00150
```
 -->
