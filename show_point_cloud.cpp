#include <vector>
#include <string>
#include <array>
#include <stdio.h>
#include <float.h>  // Para FLT_MAX e FLT_MIN
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <opencv/highgui.h>
#include <cmath>
#include <chrono>

using namespace std;

// Definição das dimensões
#define DIM1 64
#define DIM2 2650
#define DIM3_INTENSITY 1
#define DIM3_COLOR 3
#define POINTS_PER_RECORD 4  // 3 para (x, y, z) e 1 para reflexão
#define POSE_VALUES 4        // 4 valores em cada linha de pose
#define BBOX_POINTS 8        // 8 pontos por bbox (cada um com 3 coordenadas x, y, z)

typedef struct {
    double x;
    double y;
    double z;
} astro_vector_3D_t;

typedef struct {
    astro_vector_3D_t p0, p1, p2, p3, p4, p5, p6, p7;
} t_3d_bbox_struct;

typedef struct {
    int b;
    int g;
    int r;
    double cartesian_x;
    double cartesian_y;
    double cartesian_z;
    double range;
} lidar_point;


// Função para calcular o valor máximo de uma coordenada Z
float max(float arr[], int size) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

// Função para calcular o valor mínimo de uma coordenada Z
float min(float arr[], int size) {
    float min_val = FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

// Função para verificar se um ponto está dentro de um polígono (algoritmo de ray-casting)
int point_in_polygon(float px, float py, float base_points[4][2]) {
    int i, j, c = 0;
    for (i = 0, j = 3; i < 4; j = i++) {
        if (((base_points[i][1] > py) != (base_points[j][1] > py)) &&
            (px < (base_points[j][0] - base_points[i][0]) * (py - base_points[i][1]) / (base_points[j][1] - base_points[i][1]) + base_points[i][0])) {
            c = !c;
        }
    }
    return c;
}

/// Função para processar os vértices das caixas delimitadoras e colorir os pontos LIDAR
void color_points_within_bbox(vector<lidar_point> &lidar_points, vector<vector<float>> all_transformed_bbox_for_rangeview) {
    // Arrays para armazenar a base e as alturas mínima e máxima
    float* bbox_max_height = (float*)malloc(all_transformed_bbox_for_rangeview.size() * sizeof(float));
    float* bbox_min_height = (float*)malloc(all_transformed_bbox_for_rangeview.size() * sizeof(float));

    // Processar cada caixa delimitadora
    for (int i = 0; i < (int) all_transformed_bbox_for_rangeview.size(); i++) {
      // Extrair os vértices da base inferior (4 primeiros vértices)
        float base_points[4][2];  // Para armazenar x e y dos 4 vértices da base inferior
        for (int j = 0; j < 4; j++) {
            base_points[j][0] = all_transformed_bbox_for_rangeview[i][j * 3];     // x
            base_points[j][1] = all_transformed_bbox_for_rangeview[i][j * 3 + 1]; // y
        }

        // Calcular a altura máxima e mínima do bounding box
        float z_values[8];  // Para armazenar as coordenadas Z dos 4 vértices
        for (int j = 0; j < 8; j++) {
            z_values[j] = all_transformed_bbox_for_rangeview[i][j * 3 + 2];  // z
        }

        bbox_max_height[i] = max(z_values, 8);
        bbox_min_height[i] = min(z_values, 8);

        // printf("bbox_min_height: %f\n", bbox_min_height[i]);
        // printf("bbox_max_height: %f\n\n", bbox_max_height[i]);

        // Iterar sobre os pontos LIDAR e verificar se estão dentro da base inferior e da altura
        for (int k = 0; k < (int) lidar_points.size(); k++) {
            float px = lidar_points[k].cartesian_x;
            float py = lidar_points[k].cartesian_y;
            float pz = lidar_points[k].cartesian_z;
            // printf("pz: %f\n", pz);

            // Verificar se o ponto está dentro do polígono formado pela base inferior e se está na altura correta
            // bbox_min_height[j] <= pz <= bbox_max_height[j]:
            if (point_in_polygon(px, py, base_points) && pz <= bbox_max_height[i]) {
                lidar_points[k].r = 1.0f;  // Red
                lidar_points[k].g = 0.0f;  // Green
                lidar_points[k].b = 0.0f;  // Blue
            }
        }
    }

    // Liberar a memória alocada
    free(bbox_max_height);
    free(bbox_min_height);
}


// Função de normalização baseada na lógica do Python fornecido
cv::Mat normalizar(const cv::Mat& range_image, const cv::Mat& colors_image, float meters) {
    // Verificar se as dimensões das imagens correspondem
    if (range_image.rows != colors_image.rows || range_image.cols != colors_image.cols) {
        std::cerr << "As dimensões de range_image e colors_image não correspondem." << std::endl;
        return cv::Mat();
    }

    int lines = range_image.rows;
    int columns = range_image.cols;

    // Calcular a resolução
    float resolution = (meters * 100.0f) / 256.0f;

    // Inicializar a imagem normalizada como uma imagem de 8 bits em escala de cinza
    cv::Mat image_normalized = cv::Mat::zeros(lines, columns, CV_8UC1);

    // Normalizar os valores de intensidade
    for(int x = 0; x < lines; ++x) {
        for(int y = 0; y < columns; ++y) {
            float pixel = range_image.at<float>(x, y);
            if(pixel < 0.0f) {
                image_normalized.at<uchar>(x, y) = 0;
            }
            else {
                int value = static_cast<int>(255.0f - ((pixel * 100.0f) / resolution));
                value = std::min(std::max(value, 0), 255); // Garantir que esteja entre 0 e 255
                image_normalized.at<uchar>(x, y) = static_cast<uchar>(value);
            }
        }
    }

    // Converter a imagem normalizada para BGR
    cv::Mat image_color;
    cv::cvtColor(image_normalized, image_color, cv::COLOR_GRAY2BGR);

    // Criar uma máscara onde os pixels são <= -2.0
    cv::Mat mask = range_image <= -2.0f;

    // Aplicar a composição de cores baseada nas condições
    for(int x = 0; x < lines; ++x) {
        for(int y = 0; y < columns; ++y) {
            // Verificar se o pixel corresponde a vermelho [1.0, 0.0, 0.0] no colors_image com tolerância
            cv::Vec3f color = colors_image.at<cv::Vec3f>(x, y);
            bool isRed = (std::abs(color[0] - 1.0f) < 1e-3) && 
                         (std::abs(color[1] - 0.0f) < 1e-3) && 
                         (std::abs(color[2] - 0.0f) < 1e-3);

            if(isRed || mask.at<uchar>(x, y)) {
                image_color.at<cv::Vec3b>(x, y) = cv::Vec3b(0, 0, 255); // Vermelho em BGR
            }
        }
    }

    return image_color;
}

// Função para extrair o número do nome do arquivo
int extract_number_from_filename(const char *filename) {
    const char *ptr = strrchr(filename, '/');  // Localizar a última barra (se for um caminho)
    if (ptr) {
        filename = ptr + 1;  // Ajustar para o nome do arquivo
    }
    int num = 0;
    while (*filename >= '0' && *filename <= '9') {
        num = num * 10 + (*filename - '0');
        filename++;
    }
    return num;
}

// Função para comparar dois arquivos (para ordenação numérica)
int compare_files(const void *a, const void *b) {
    const char *file_a = *(const char**)a;
    const char *file_b = *(const char**)b;

    int num_a = extract_number_from_filename(file_a);
    int num_b = extract_number_from_filename(file_b);

    return num_a - num_b;
}

// Função para listar subdiretórios (scenes)
vector<string> list_subdirectories(const char *dir_path) {
    vector<string> subdirs;
    DIR *dp = opendir(dir_path);
    if (dp == NULL) {
        perror("Erro ao abrir o diretório");
        return subdirs;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        string entry_name = string(entry->d_name);
        if (entry_name == "." || entry_name == "..") continue;

        string full_path = string(dir_path) + "/" + entry_name;

        struct stat path_stat;
        if (stat(full_path.c_str(), &path_stat) != 0) {
            perror("Erro ao obter informações do subdiretório");
            continue;
        }

        if (S_ISDIR(path_stat.st_mode)) {
            subdirs.push_back(entry_name);
        }
    }

    closedir(dp);
    return subdirs;
}

// Função para listar arquivos com uma extensão específica
vector<string> list_files_with_extension(const char *dir_path, const string &extension) {
    vector<string> files;
    DIR *dp = opendir(dir_path);
    if (dp == NULL) {
        perror("Erro ao abrir o diretório");
        return files;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        string entry_name = string(entry->d_name);
        if (entry_name == "." || entry_name == "..") continue;

        string full_path = string(dir_path) + "/" + entry_name;

        struct stat path_stat;
        if (stat(full_path.c_str(), &path_stat) != 0) {
            perror("Erro ao obter informações do arquivo");
            continue;
        }

        if (S_ISREG(path_stat.st_mode)) {
            if (entry_name.size() >= extension.size()) {
                if (entry_name.compare(entry_name.size() - extension.size(), extension.size(), extension) == 0) {
                    files.push_back(entry_name);
                }
            }
        }
    }

    closedir(dp);
    return files;
}

// Função para extrair uma substring com base na posição e separador
string take_substring(int position, const string& separator, const string& name) {
    vector<string> tokens;
    size_t start = 0, end;

    while ((end = name.find(separator, start)) != string::npos) {
        tokens.push_back(name.substr(start, end - start));
        start = end + separator.length();
    }
    tokens.push_back(name.substr(start));

    if (position >= 1 && position <= (int) tokens.size()) {
        return tokens[position - 1];
    } else {
        return "";
    }
}

// Função para multiplicação de matriz 4x4 com vetor 4x1
void multiply_matrix_vector(const float mat[4][4], const float vec[4], float result[4]) {
    for (int i = 0; i < 4; i++) {
        result[i] = 0.0f;
        for (int j = 0; j < 4; j++) {
            result[i] += mat[i][j] * vec[j];
        }
    }
}

// Função para calcular a inversa de uma matriz 4x4 usando OpenCV
int invert_matrix(const float mat[4][4], float inv[4][4]) {
    // Convertir a matriz para cv::Mat
    cv::Mat mat_cv(4, 4, CV_32F);
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            mat_cv.at<float>(i,j) = mat[i][j];

    cv::Mat inv_cv;
    bool success = cv::invert(mat_cv, inv_cv, cv::DECOMP_SVD);
    if (!success) {
        return 0;  // Matriz não invertível
    }

    // Copiar a matriz inversa para o array inv
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            inv[i][j] = inv_cv.at<float>(i,j);

    return 1;  // Sucesso
}

// Função equivalente ao Python transform_vertices
void transform_vertices(const vector<array<float, 3>>& vertices, float lidar_pose[4][4], vector<float>& result) {
    float lidar_pose_inverse[4][4];
    if (!invert_matrix(lidar_pose, lidar_pose_inverse)) {
        cerr << "Erro ao inverter a matriz de pose." << endl;
        return; // Retorna vetor vazio
    }

    for (size_t i = 0; i < vertices.size(); i++) {
        float homogeneous_vertex[4] = { 
            vertices[i][0], 
            vertices[i][1], 
            vertices[i][2], 
            1.0f };
        float transformed_homogeneous[4];

        multiply_matrix_vector(lidar_pose_inverse, homogeneous_vertex, transformed_homogeneous);
        result.push_back(transformed_homogeneous[0]);
        result.push_back(transformed_homogeneous[1]);
        result.push_back(transformed_homogeneous[2]);
    }
}   

// Função para transformar as coordenadas para o sistema LIDAR
void transformar_para_sistema_lidar_topo(const vector<array<float, 3>>& bbox, float lidar_pose[4][4], vector<float>& result) {
    float lidar_pose_inversa[4][4];
    
    // Calculando a inversa da matriz de pose
    if (!invert_matrix(lidar_pose, lidar_pose_inversa)) {
        printf("Erro ao inverter a matriz de pose.\n");
        return;
    }

    // Para cada ponto do bounding box
    for (size_t i = 0; i < bbox.size(); i++) {
        // Adicionando a coordenada homogênea
        float track_coords_homogeneas[4] = {
            bbox[i][0], // x
            bbox[i][1], // y
            bbox[i][2], // z
            1.0f        // Homogeneização
        };

        float track_coords_transformadas[4];
        
        // Multiplicando a matriz de pose inversa com a coordenada homogênea
        multiply_matrix_vector(lidar_pose_inversa, track_coords_homogeneas, track_coords_transformadas);

        // Adicionando as coordenadas transformadas no vetor de resultados
        result.push_back(track_coords_transformadas[0]);  // x transformado
        result.push_back(track_coords_transformadas[1]);  // y transformado
        result.push_back(track_coords_transformadas[2]);  // z transformado
    }
}

// Função para desenhar o bounding box em uma imagem Bird's-Eye View
void draw_bounding_box_birdview(const vector<float>& result, cv::Mat& birdview_image, float meters = 75.0f, int scale = 10) {
    // Assumindo que 'result' contém as coordenadas transformadas de 8 pontos
    // Aqui, precisamos desenhar as linhas que conectam os pontos do bounding box

    // Verificar se temos 24 valores (8 pontos x 3 coordenadas)
    if (result.size() < 24) {
        printf("Bounding box incompleto para desenho.\n");
        return;
    }

    // Definir as arestas do bounding box
    int edges[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0}, // base
        {4,5}, {5,6}, {6,7}, {7,4}, // topo
        {0,4}, {1,5}, {2,6}, {3,7}  // verticais
    };

    for(int i = 0; i < 12; i++) {
        int idx1 = edges[i][0];
        int idx2 = edges[i][1];

        float x1 = result[idx1*3];
        float y1 = result[idx1*3 +1];
        float x2 = result[idx2*3];
        float y2 = result[idx2*3 +1];

        // Verificando se as coordenadas estão dentro do limite de metros
        if (x1 > -meters && x1 < meters && y1 > -meters && y1 < meters &&
            x2 > -meters && x2 < meters && y2 > -meters && y2 < meters) {
            // Convertendo as coordenadas para pixels
            int x_pixel1 = static_cast<int>((x1 + meters) * scale);
            int y_pixel1 = static_cast<int>((meters - y1) * scale);
            int x_pixel2 = static_cast<int>((x2 + meters) * scale);
            int y_pixel2 = static_cast<int>((meters - y2) * scale);

            // Desenhando a linha entre os pontos (x1, y1) e (x2, y2)
            cv::line(birdview_image, 
                        cv::Point(x_pixel1, y_pixel1), 
                        cv::Point(x_pixel2, y_pixel2), 
                        cv::Scalar(0, 255, 0), 2);  // Verde e espessura 2
        }
    }
}

// Função para calcular a imagem Birdview
void calculate_birdview_image(cv::Mat &birdview_image_color, vector<lidar_point> &lidar_points, const float *points_xyz, size_t num_points, int meters, int scale) {
    int size_in_pixels = 2 * meters * scale;
    cv::Mat birdview_image = cv::Mat::zeros(size_in_pixels, size_in_pixels, CV_8UC1);

    for (size_t i = 0; i < num_points; i++) {
        float x = points_xyz[i * POINTS_PER_RECORD];       // x
        float y = points_xyz[i * POINTS_PER_RECORD + 1];   // y
        float z = points_xyz[i * POINTS_PER_RECORD + 2];   // z
        float r = points_xyz[i * POINTS_PER_RECORD + 3];   // reflexão

        lidar_point point;
        point.cartesian_x   = x;
        point.cartesian_y   = y;
        point.cartesian_z   = z;
        point.range         = r;
        point.r             = 0.0;
        point.g             = 0.0;
        point.b             = 1.0;

        lidar_points.push_back(point);

        if (-meters < x && x < meters && -meters < y && y < meters) {
            int x_pixel = static_cast<int>((x + meters) * scale);
            int y_pixel = static_cast<int>((meters - y) * scale);

            if (x_pixel >= 0 && x_pixel < size_in_pixels && y_pixel >= 0 && y_pixel < size_in_pixels) {
                birdview_image.at<uchar>(y_pixel, x_pixel) = 255; 
            }
        }
    }

    cv::cvtColor(birdview_image, birdview_image_color, cv::COLOR_GRAY2BGR);
}

// Função para ler o arquivo binário
void read_bin_file(const char *filename, vector<lidar_point> &lidar_points, cv::Mat &birdview_image) {
    FILE *file = fopen(filename, "rb");
    // printf("Lido arquivo: %s\n", filename);

    if (file == NULL) {
        perror("Erro ao abrir o arquivo");
        return;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size % (POINTS_PER_RECORD * sizeof(float)) != 0) {
        fprintf(stderr, "O arquivo binário tem um tamanho inválido: %s\n", filename);
        fclose(file);
        return;
    }

    size_t num_points = file_size / (POINTS_PER_RECORD * sizeof(float));
    float *points = (float*) malloc(num_points * POINTS_PER_RECORD * sizeof(float));
    if (points == NULL) {
        perror("Erro ao alocar memória");
        fclose(file);
        return;
    }

    size_t read_count = fread(points, sizeof(float), num_points * POINTS_PER_RECORD, file);
    if (read_count != num_points * POINTS_PER_RECORD) {
        fprintf(stderr, "Erro ao ler o arquivo binário: %s\n", filename);
        free(points);
        fclose(file);
        return;
    }

    int meters = 75;
    int scale = 10;
    calculate_birdview_image(birdview_image, lidar_points, points, num_points, meters, scale);

    free(points);
    fclose(file);
}

// Função para ler o arquivo de pose
bool read_pose_file(const char *filename, float pose[4][4]) {
    FILE *file = fopen(filename, "r");
    // printf("Lido arquivo de pose: %s\n", filename);

    if (file == NULL) {
        perror("Erro ao abrir o arquivo de pose");
        return false;
    }

    // Ler as quatro linhas do arquivo
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (fscanf(file, "%f", &pose[i][j]) != 1) {
                fprintf(stderr, "Erro ao ler pose no arquivo: %s\n", filename);
                fclose(file);
                return false;
            }
        }
    }

    fclose(file);
    return true;
}


// Função para ler os arquivos de bounding box
void read_bbox_file(const char *objs_bbox_dir, const string& scene_name, const string& pose_name, vector<vector<array<float, 3>>> &all_bbox, float lidar_pose[4][4], cv::Mat& birdview_image, vector<vector<float>> &all_transformed_bbox_for_rangeview) {
    string pose_number = take_substring(1, ".", pose_name);  // "2.txt" -> "2"
    // printf("%s\n", pose_number.c_str());
    if (pose_number.empty()) {
        printf("Pose number extraído está vazio para pose_name: %s\n", pose_name.c_str());
        return;
    }

    // Construir o caminho para a pasta de bounding boxes: objs_bbox_dir/scene_name/pose_number/
    string bbox_dir = string(objs_bbox_dir) + "/" + scene_name + "/" + pose_number;

    // printf("Caminho para bbox: %s\n", bbox_dir.c_str());

    // Abrir o diretório de bounding boxes
    DIR *dp = opendir(bbox_dir.c_str());
    if (dp == NULL) {
        perror("Erro ao abrir o diretório de bbox");
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        string entry_name = string(entry->d_name);
        if (entry_name == "." || entry_name == "..") continue;

        // Construir o caminho completo do arquivo de bbox
        string full_path = bbox_dir + "/" + entry_name;

        struct stat path_stat;
        if (stat(full_path.c_str(), &path_stat) != 0) {
            perror("Erro ao obter informações do arquivo de bbox");
            continue;
        }

        if (S_ISREG(path_stat.st_mode) && entry_name.find(".txt") != string::npos) {
            FILE *file = fopen(full_path.c_str(), "r");
            // printf("Lido arquivo de bbox: %s\n", full_path.c_str());

            if (file == NULL) {
                perror("Erro ao abrir o arquivo de bbox");
                continue;
            }

            float bbox[8][3];  // 8 pontos (x, y, z)
            vector<array<float, 3>> bbox_aux;

            // Ler as 8 linhas com 3 valores cada
            bool read_success = true;
            for (int i = 0; i < 8; i++) {
                if (fscanf(file, "%f %f %f", &bbox[i][0], &bbox[i][1], &bbox[i][2]) != 3) {
                    fprintf(stderr, "Erro ao ler bbox no arquivo: %s\n", full_path.c_str());
                    read_success = false;
                    break;
                }
                bbox_aux.push_back({bbox[i][0], bbox[i][1], bbox[i][2]});
            }

            if (read_success) {
                all_bbox.push_back(bbox_aux);

                // Transformar as coordenadas para o sistema LIDAR
                vector<float> transformed_bbox_for_birdeyeview;
                vector<float> transformed_bbox_for_rangeview;
                transformar_para_sistema_lidar_topo(bbox_aux, lidar_pose, transformed_bbox_for_birdeyeview);
                transform_vertices(bbox_aux, lidar_pose, transformed_bbox_for_rangeview);
                all_transformed_bbox_for_rangeview.push_back(transformed_bbox_for_rangeview);
                // Desenhar o bounding box na imagem birdview
                draw_bounding_box_birdview(transformed_bbox_for_birdeyeview, birdview_image, 75, 10);
            }

            fclose(file);
        }
    }

    closedir(dp);
}

int main(int argc, char** argv) {
    bool draw_red_points = true;
    bool show = true;
    std::string input_base_path = "";
    int wait_delay = 1; // Padrão: 1ms (máxima velocidade)

    // --- 1. Processamento de Argumentos ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-no_red") {
            draw_red_points = false;
        } else if (arg == "-no_show") {
            show = false;
        } else if (arg == "--input") {
            if (i + 1 < argc) {
                input_base_path = argv[++i];
            } else {
                std::cerr << "Erro: O argumento --input requer um caminho." << std::endl;
                return EXIT_FAILURE;
            }
        } else if (arg == "-v") {
            // Verifica se o usuário passou um valor para a velocidade
            if (i + 1 < argc) {
                try {
                    wait_delay = std::stoi(argv[++i]); // Converte string para int
                    if (wait_delay < 1) wait_delay = 1; // Garante mínimo de 1ms
                } catch (...) {
                    std::cerr << "Erro: Valor inválido para -v. Use um número inteiro (ex: -v 100)." << std::endl;
                    return EXIT_FAILURE;
                }
            } else {
                std::cerr << "Erro: O argumento -v requer um valor em milissegundos." << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    // --- 2. Validação do Caminho de Entrada ---
    if (input_base_path.empty()) {
        std::cerr << "Erro: Caminho de entrada nao especificado." << std::endl;
        std::cerr << "Uso: " << argv[0] << " --input <caminho> [-v <ms>] [-no_red] [-no_show]" << std::endl;
        return EXIT_FAILURE;
    }

    if (input_base_path.back() == '/') {
        input_base_path.pop_back();
    }

    std::string bin_root_str = input_base_path + "/bin_files";
    std::string pose_root_str = input_base_path + "/poses";
    std::string bbox_root_str = input_base_path + "/objs_bbox";

    const char *bin_root_dir = bin_root_str.c_str();
    const char *pose_root_dir = pose_root_str.c_str();
    const char *bbox_root_dir = bbox_root_str.c_str();

    std::chrono::duration<double> global_delta_time(0.0);

    // --- 3. Listar Cenas ---
    vector<string> scenes = list_subdirectories(bin_root_dir);
    if (scenes.empty()) {
        printf("Nenhuma cena encontrada em: %s\n", bin_root_dir);
        return EXIT_FAILURE;
    }

    std::vector<float> intensities(DIM1 * DIM2 * DIM3_INTENSITY, 0.0f);
    std::vector<float> all_points_colors(DIM1 * DIM2 * DIM3_COLOR, 0.0f);

    int index = 1;

    // --- 4. Loop das Cenas ---
    for (const auto& scene : scenes) {
        auto initial_time = std::chrono::high_resolution_clock::now();
        printf("\nProcessando cena: %s\n", scene.c_str());

        string bin_scene_dir = string(bin_root_dir) + "/" + scene;
        string pose_scene_dir = string(pose_root_dir) + "/" + scene;
        string bbox_scene_dir = string(bbox_root_dir) + "/" + scene;

        struct stat st;
        if (stat(pose_scene_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            printf("Diretório de poses não encontrado: %s\n", scene.c_str());
            continue;
        }
        if (stat(bbox_scene_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            printf("Diretório de bbox não encontrado: %s\n", scene.c_str());
            continue;
        }

        vector<string> bin_files = list_files_with_extension(bin_scene_dir.c_str(), ".bin");
        vector<string> pose_files = list_files_with_extension(pose_scene_dir.c_str(), ".txt");

        if (bin_files.empty() || pose_files.empty()) {
            printf("Arquivos faltando na cena: %s\n", scene.c_str());
            continue;
        }

        size_t num_files = min(bin_files.size(), pose_files.size());

        const char **bin_files_c = (const char **)malloc(bin_files.size() * sizeof(char *));
        for (size_t i = 0; i < bin_files.size(); i++) {
            string full_path = bin_scene_dir + "/" + bin_files[i];
            bin_files_c[i] = strdup(full_path.c_str());
        }

        const char **pose_files_c = (const char **)malloc(pose_files.size() * sizeof(char *));
        for (size_t i = 0; i < pose_files.size(); i++) {
            string full_path = pose_scene_dir + "/" + pose_files[i];
            pose_files_c[i] = strdup(full_path.c_str());
        }

        qsort((void*)bin_files_c, bin_files.size(), sizeof(char*), compare_files);
        qsort((void*)pose_files_c, pose_files.size(), sizeof(char*), compare_files);

        vector<lidar_point> lidar_points;

        // --- 5. Loop dos Arquivos ---
        for (size_t i = 0; i < num_files; i++) {
            auto local_initial_time = std::chrono::high_resolution_clock::now();

            string bin_file_path = string(bin_files_c[i]);
            string pose_file_path = string(pose_files_c[i]);

            cv::Mat birdview_image;
            read_bin_file(bin_file_path.c_str(), lidar_points, birdview_image);

            float pose[4][4];
            if (!read_pose_file(pose_file_path.c_str(), pose)) {
                printf("Falha ao ler pose: %s\n", pose_file_path.c_str());
                continue;
            }

            // Extração robusta do nome do arquivo
            size_t last_slash = pose_file_path.find_last_of('/');
            string filename_full = (last_slash == string::npos) ? pose_file_path : pose_file_path.substr(last_slash + 1);
            size_t last_dot = filename_full.find_last_of('.');
            string pose_file_name = (last_dot == string::npos) ? filename_full : filename_full.substr(0, last_dot);
            
            if (pose_file_name.empty()) continue;

            vector<vector<array<float, 3>>> all_bbox;
            vector<vector<float>> all_transformed_bbox_for_rangeview;
            read_bbox_file(bbox_root_dir, scene, pose_file_name, all_bbox, pose, birdview_image, all_transformed_bbox_for_rangeview);
           
            if (show) {
                if (!lidar_points.empty()) {
                    for(int k = 0; k < (int)lidar_points.size(); k++) {
                        intensities[k] = lidar_points[k].range;
                    }
                }

                if (draw_red_points)
                    color_points_within_bbox(lidar_points, all_transformed_bbox_for_rangeview);
                
                size_t max_points = std::min((size_t)(DIM1 * DIM2), lidar_points.size());
                for (size_t k = 0; k < max_points; ++k) {
                    all_points_colors[k * 3 + 0] = (float)lidar_points[k].r;
                    all_points_colors[k * 3 + 1] = (float)lidar_points[k].g;
                    all_points_colors[k * 3 + 2] = (float)lidar_points[k].b;
                }

                cv::Mat range_image_reshaped(DIM1, DIM2, CV_32F, intensities.data());
                cv::Mat all_points_colors_reshaped(DIM1, DIM2, CV_32FC3, all_points_colors.data());

                float threshold = 75.0f;
                cv::Mat normalized_image = normalizar(range_image_reshaped, all_points_colors_reshaped, threshold);

                cv::Mat resized_range;
                cv::resize(normalized_image, resized_range, cv::Size(1920, 128), 0, 0, cv::INTER_LINEAR);

                cv::Mat resized_birdview;
                cv::resize(birdview_image, resized_birdview, cv::Size(920, 920), 0, 0, cv::INTER_LINEAR);

                cv::Mat bg(1080, 1920, CV_8UC3, cv::Scalar(255, 255, 255));
                resized_range.copyTo(bg(cv::Rect(0, 0, resized_range.cols, resized_range.rows)));

                cv::Rect roi_birdview(480, 128, resized_birdview.cols, resized_birdview.rows);
                resized_birdview.copyTo(bg(roi_birdview));

                const std::string winName = "LiDAR Range and Bird's-Eye View";
                cv::namedWindow(winName, cv::WINDOW_NORMAL);
                cv::setWindowProperty(winName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
                cv::imshow(winName, bg);

                bool paused = false;
                while (true) {
                    // --- AQUI A MUDANÇA ---
                    // Usa wait_delay em vez de 1
                    int key = cv::waitKey(wait_delay);

                    if (key == 32) paused = !paused; // Espaço
                    else if (key == 27) return 0;    // ESC
                    
                    if (!paused) break;
                }
            }

            lidar_points.clear();
            all_bbox.clear();

            auto local_final_time = std::chrono::high_resolution_clock::now();
            global_delta_time += (local_final_time - local_initial_time);
            index++;
        }

        for (size_t i = 0; i < bin_files.size(); i++) free((void*)bin_files_c[i]);
        free(bin_files_c);
        for (size_t i = 0; i < pose_files.size(); i++) free((void*)pose_files_c[i]);
        free(pose_files_c);
    }
    
    // Evita divisão por zero se index for 1
    if (index > 1) index--; 
    std::cout << " Global Deltatime Average: " << global_delta_time.count()/index << " seconds" << std::endl;

    return EXIT_SUCCESS;
}