#include "voxel_representation.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

// Includes da PCL
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// Includes do OpenCV
#include <vector>
#include <string>
#include <iostream>
#include <algorithm> // para std::transform
#include <opencv2/core.hpp> // Certifique-se de ter o OpenCV incluído
#include <opencv2/opencv.hpp>
#include "opengl_viewer.hpp"

using namespace std;

std::string default_color_map = "jet"; 

std::vector<std::pair<float, cv::Vec3f>> get_color_map(std::string name) {
    // 1. Normaliza a entrada para minúsculas (para aceitar "Turbo", "TURBO", "turbo")
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    // 2. Retorna o padrão correspondente
    if (name == "viridis") {
        return {
            {0.0f, {68, 1, 84}}, {0.25f, {59, 82, 139}},
            {0.5f, {33, 145, 140}}, {0.75f, {94, 201, 98}},{1.0f, {253, 231, 37}}
        };
    }
    if (name == "plasma") {
        return {
            {0.0f, {13, 8, 135}}, {0.25f, {126, 3, 168}}, 
            {0.5f, {204, 71, 120}}, {0.75f, {248, 149, 64}}, {1.0f, {240, 249, 33}}
        };
    } 
    else if (name == "inferno") {
        return {
            {0.0f, {0, 0, 4}}, {0.25f, {87, 16, 110}}, 
            {0.5f, {188, 55, 84}}, {0.75f, {249, 142, 9}}, {1.0f, {252, 255, 164}}
        };
    }
    else if (name == "jet") {
        return {
            {0.0f, {0, 0, 128}}, {0.25f, {0, 255, 255}}, 
            {0.5f, {0, 255, 0}}, {0.75f, {255, 255, 0}}, {1.0f, {255, 0, 0}}
        };
    }
    else if (name == "hot") {
        return {
            {0.0f, {0, 0, 0}}, {0.33f, {255, 0, 0}}, 
            {0.66f, {255, 255, 0}}, {1.0f, {255, 255, 255}}
        };
    }
    // "Turbo" é geralmente um excelente padrão (default)
    else {
        if (name != "turbo") {
            std::cerr << "[Aviso] Mapa de cores '" << name << "' nao encontrado. Usando 'turbo'." << std::endl;
        }
        return {
            {0.0f, {48, 18, 59}}, {0.2f, {70, 134, 250}}, 
            {0.4f, {27, 217, 172}}, {0.6f, {213, 187, 33}}, 
            {0.8f, {253, 114, 52}}, {1.0f, {122, 4, 3}}
        };
    }
}

std::vector<std::pair<float, cv::Vec3f>> color_points;

// Função para aplicar um mapa de cores viridis a um valor normalizado (0-1)
cv::Vec3b colormap(float value) {
    // Formato: {posição, {R, G, B}}

    if (value <= 0.0) return cv::Vec3b(color_points.front().second[2], color_points.front().second[1], color_points.front().second[0]);
    if (value >= 1.0) return cv::Vec3b(color_points.back().second[2], color_points.back().second[1], color_points.back().second[0]);

    for (size_t i = 0; i < color_points.size() - 1; ++i) {
        if (value >= color_points[i].first && value <= color_points[i+1].first) {
            float t = (value - color_points[i].first) / (color_points[i+1].first - color_points[i].first);
            cv::Vec3f color = (1.0f - t) * color_points[i].second + t * color_points[i+1].second;
            return cv::Vec3b(color[2], color[1], color[0]); // OpenCV usa BGR
        }
    }
    return cv::Vec3b(color_points.back().second[2], color_points.back().second[1], color_points.back().second[0]);
}


GroundCorrectionResult find_and_correct_ground_plane(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float distance_threshold,
    int num_iterations) 
{
    std::cout << "\n--- Executando RANSAC para encontrar e corrigir o plano do solo ---" << std::endl;

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold);
    seg.setMaxIterations(num_iterations);
    
    Eigen::Vector3f axis = Eigen::Vector3f(0.0, 0.0, 1.0);
    seg.setAxis(axis);
    seg.setEpsAngle(0.2);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        std::cerr << "RANSAC não conseguiu encontrar um plano." << std::endl;
        return {cloud, 0.0, false};
    }

    Eigen::Vector3f normal_vector(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    if (normal_vector.z() < 0) {
        normal_vector = -normal_vector;
    }

    const Eigen::Vector3f z_axis_vector(0.0f, 0.0f, 1.0f);
    float dot_product = normal_vector.dot(z_axis_vector);
    double angle_rad = acos(std::max(-1.0f, std::min(1.0f, dot_product)));
    double angle_deg = angle_rad * 180.0 / M_PI;
    std::cout << "O plano do solo está inclinado em aproximadamente " << angle_deg << " graus." << std::endl;

    Eigen::Quaternionf q;
    q.setFromTwoVectors(normal_vector, z_axis_vector);
    Eigen::Matrix4f transform_rotation = Eigen::Matrix4f::Identity();
    transform_rotation.block<3,3>(0,0) = q.toRotationMatrix();
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *rotated_cloud, transform_rotation);

    pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud_corrected(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*rotated_cloud, *inliers, *inlier_cloud_corrected);

    float mean_z_inliers = 0.0f;
    for (const auto& point : inlier_cloud_corrected->points) {
        mean_z_inliers += point.z;
    }
    if (!inlier_cloud_corrected->points.empty()){
      mean_z_inliers /= inlier_cloud_corrected->points.size();
    }


    Eigen::Affine3f transform_translation = Eigen::Affine3f::Identity();
    transform_translation.translation() << 0.0, 0.0, -mean_z_inliers;

    pcl::PointCloud<pcl::PointXYZ>::Ptr corrected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*rotated_cloud, *corrected_cloud, transform_translation);
    
    std::cout << "Nuvem de pontos corrigida. O plano do solo agora está em Z=0." << std::endl;

    return {corrected_cloud, angle_deg, true};
}

void create_encoded_bev_from_height_map(
    const cv::Mat& height_map,
    const std::pair<float, float>& z_range,
    float height_step,
    const std::string& filename)
{
    int image_size = height_map.rows;
    cv::Mat encoded_image = cv::Mat::zeros(image_size, image_size, CV_8UC3);

    for (int r = 0; r < image_size; ++r) {
        for (int c = 0; c < image_size; ++c) {
            float height = height_map.at<float>(r, c);
            if (height > z_range.first) {
                int height_index = static_cast<int>(round((height - z_range.first) / height_step));
                uchar R = height_index % 256;
                uchar G = height_index / 256;
                encoded_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, G, R); // OpenCV is BGR
            }
        }
    }
    
    cv::rotate(encoded_image, encoded_image, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::imwrite(filename, encoded_image);
    std::cout << "Imagem codificada com altura salva em '" << filename << "'" << std::endl;
}


cv::Mat generate_bev_from_points(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcd,
    const std::pair<float, float>& x_range,
    const std::pair<float, float>& y_range,
    const std::pair<float, float>& z_range,
    int image_size)
{
    std::cout << "\n--- Gerando BEV a partir de Pontos ---" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(pcd);
    pass.setFilterFieldName("x"); pass.setFilterLimits(x_range.first, x_range.second); pass.filter(*cloud_filtered);
    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("y"); pass.setFilterLimits(y_range.first, y_range.second); pass.filter(*cloud_filtered);
    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("z"); pass.setFilterLimits(z_range.first, z_range.second); pass.filter(*cloud_filtered);

    if (cloud_filtered->empty()) {
        std::cerr << "Nenhum ponto encontrado no range especificado." << std::endl;
        return cv::Mat();
    }

    cv::Mat height_map(image_size, image_size, CV_32FC1, cv::Scalar(z_range.first - 1.0f));
    float x_res = (x_range.second - x_range.first) / image_size;
    float y_res = (y_range.second - y_range.first) / image_size;

    for(const auto& point : *cloud_filtered) {
        int px = static_cast<int>((point.x - x_range.first) / x_res);
        int py = static_cast<int>((point.y - y_range.first) / y_res);
        if (px >= 0 && px < image_size && py >= 0 && py < image_size) {
            if (point.z > height_map.at<float>(py, px)) {
                height_map.at<float>(py, px) = point.z;
            }
        }
    }

    height_map.setTo(z_range.first, height_map < z_range.first);
    
    float z_delta = z_range.second - z_range.first;
    if (z_delta == 0) z_delta = 1.0f;
    
    cv::Mat normalized_map, colored_image(image_size, image_size, CV_8UC3);
    cv::normalize(height_map, normalized_map, 1.0, 0.0, cv::NORM_MINMAX, CV_32F);

    for (int r=0; r < image_size; ++r) {
        for (int c=0; c < image_size; ++c) {
            colored_image.at<cv::Vec3b>(r,c) = colormap(normalized_map.at<float>(r,c));
        }
    }
    
    cv::rotate(colored_image, colored_image, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::imwrite("bird_eye_view_points.png", colored_image);
    std::cout << "Imagem 'bird_eye_view_points.png' salva com sucesso." << std::endl;
    
    return height_map;
}

cv::Mat load_and_decode_bev_image(
    const std::string& encoded_image_path,
    const std::pair<float, float>& z_range,
    float height_step)
{
    std::cout << "\n--- Decodificando imagem de " << encoded_image_path << " ---" << std::endl;
    cv::Mat encoded_image_bgr = cv::imread(encoded_image_path, cv::IMREAD_COLOR);
    if (encoded_image_bgr.empty()) {
        std::cerr << "Erro: Arquivo de imagem não encontrado em '" << encoded_image_path << "'" << std::endl;
        return cv::Mat();
    }

    cv::Mat encoded_image_float;
    encoded_image_bgr.convertTo(encoded_image_float, CV_32FC3, 1.0/255.0);

    float max_r = ((z_range.second - z_range.first) / height_step) / 256.0f;

    for (int r = 0; r < encoded_image_float.rows; ++r) {
        for (int c = 0; c < encoded_image_float.cols; ++c) {
            cv::Vec3f& pixel = encoded_image_float.at<cv::Vec3f>(r, c);
            if (pixel[2] > max_r) { // R channel is at index 2 in BGR
                pixel = {0,0,0};
            }
             if (pixel[0] == 1.0 && pixel[1] == 1.0 && pixel[2] == 1.0) { // white
                pixel = {0,0,0};
            }
            pixel[0] = 0; // Zero blue
            pixel[1] = 0; // Zero green
        }
    }

    cv::Mat encoded_image;
    encoded_image_float.convertTo(encoded_image, CV_8UC3, 255.0);

    cv::rotate(encoded_image, encoded_image, cv::ROTATE_90_CLOCKWISE);
    int image_size = encoded_image.rows;
    cv::Mat height_map = cv::Mat(image_size, image_size, CV_32FC1, cv::Scalar(z_range.first));

    for (int r = 0; r < image_size; ++r) {
        for (int c = 0; c < image_size; ++c) {
            cv::Vec3b pixel = encoded_image.at<cv::Vec3b>(r, c);
            if (pixel[1] > 0 || pixel[2] > 0) { // if G or R is not zero
                int height_indices = static_cast<int>(pixel[1]) * 256 + static_cast<int>(pixel[2]);
                float height = height_indices * height_step + z_range.first;
                height_map.at<float>(r, c) = height;
            }
        }
    }

    std::cout << "Mapa de altura reconstruído com sucesso a partir da imagem." << std::endl;
    return height_map;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr load_bin_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Erro ao abrir o arquivo: " << file_path << std::endl;
        return nullptr;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Erro ao ler o arquivo: " << file_path << std::endl;
        return nullptr;
    }
    file.close();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < buffer.size(); i += 4) {
        pcl::PointXYZ point;
        point.x = buffer[i];
        point.y = buffer[i + 1];
        point.z = buffer[i + 2];
        cloud->points.push_back(point);
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

// Função para parsear argumentos da linha de comando
std::map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> args;
    args["--mode"] = "pillars"; // default
    // args["--input"] = "/home/lume/astro/data/lidar_sweep_viewer/waymo_10/bin_files"; // default

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) {
             if (i + 1 < argc) {
                args[arg] = argv[++i];
            }
        }
    }
    return args;
}


void generate_bev_and_visualize_voxels_opengl(
    OpenGLViewer& viewer,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcd,
    const std::pair<float, float>& x_range,
    const std::pair<float, float>& y_range,
    const std::pair<float, float>& z_range,
    float voxel_size,
    int image_size)
    {
    std::cout << "\n--- Gerando BEV e visualização a partir de Voxels com OpenGL ---" << std::endl;
    std::cout << "Tamanho do Voxel: " << voxel_size << " metros" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(pcd);
    pass.setFilterFieldName("x"); pass.setFilterLimits(x_range.first, x_range.second); pass.filter(*cloud_filtered);
    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("y"); pass.setFilterLimits(y_range.first, y_range.second); pass.filter(*cloud_filtered);
    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("z"); pass.setFilterLimits(z_range.first, z_range.second); pass.filter(*cloud_filtered);

    if (cloud_filtered->empty()) {
        std::cerr << "Nenhum ponto encontrado no range especificado para voxelização." << std::endl;
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_centroids(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setLeafSize(voxel_size, voxel_size, voxel_size);
    sor.filter(*voxel_centroids);
    std::cout << "Nuvem de pontos convertida em " << voxel_centroids->size() << " voxels (centroides)." << std::endl;

    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> colors;
    float z_delta = z_range.second - z_range.first;
    

    cv::Mat height_map(image_size, image_size, CV_32FC1, cv::Scalar(z_range.first - 1));
    float x_res = (x_range.second - x_range.first) / image_size;
    float y_res = (y_range.second - y_range.first) / image_size;

    float half_voxel_px_f = (voxel_size / x_res) / 2.0f;
    float half_voxel_py_f = (voxel_size / y_res) / 2.0f;

    if (z_delta == 0) z_delta = 1.0f;

    for (const auto& point : *voxel_centroids) {
        positions.emplace_back(point.x, point.y, point.z);
        float normalized_z = (point.z - z_range.first) / z_delta;
        cv::Vec3b color_bgr = colormap(normalized_z);
        colors.emplace_back(color_bgr[2] / 255.0f, color_bgr[1] / 255.0f, color_bgr[0] / 255.0f);
        float center_px = (point.x - x_range.first) / x_res;
        float center_py = (point.y - y_range.first) / y_res;

        int px_start = std::max(0, static_cast<int>(center_px - half_voxel_px_f));
        int px_end = std::min(image_size, static_cast<int>(center_px + half_voxel_px_f));
        int py_start = std::max(0, static_cast<int>(center_py - half_voxel_py_f));
        int py_end = std::min(image_size, static_cast<int>(center_py + half_voxel_py_f));
        
        for (int r = py_start; r < py_end; ++r) {
            for (int c = px_start; c < px_end; ++c) {
                if (point.z > height_map.at<float>(r, c)) {
                    height_map.at<float>(r, c) = point.z;
                }
            }
        }
    }
    
    height_map.setTo(z_range.first, height_map < z_range.first);
    create_encoded_bev_from_height_map(height_map, z_range, voxel_size, "bird_eye_view_voxels.png");

    try {
        viewer.add_voxels(positions, colors);
        viewer.run();
    } catch (const std::exception& e) {
        std::cerr << "Erro ao inicializar o visualizador OpenGL: " << e.what() << std::endl;
    }
}


void visualize_pillars_from_map_opengl(
    OpenGLViewer& viewer,
    const cv::Mat& height_map,
    const std::pair<float, float>& x_range,
    const std::pair<float, float>& y_range,
    const std::pair<float, float>& z_range)
{
    if (height_map.empty()) return;
    std::cout << "\nMostrando visualização 3D de Pilares com OpenGL..." << std::endl;
    
    int image_size = height_map.rows;
    float x_res = (x_range.second - x_range.first) / image_size;
    float y_res = (y_range.second - y_range.first) / image_size;
    float z_delta = z_range.second - z_range.first;
    if (z_delta == 0) z_delta = 1.0f;

    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> colors;
    float voxel_size = 0.2f; // The same as in the other opengl function

    for (int r = 0; r < image_size; ++r) {
        for (int c = 0; c < image_size; ++c) {
            float height = height_map.at<float>(r, c);
            if (height > z_range.first + 1e-6) {
                float x = c * x_res + x_range.first;
                float y = r * y_res + y_range.first;
                
                for (float z = z_range.first; z < height; z += voxel_size) {
                    positions.emplace_back(x, -y, z);
                    float normalized_z = (z - z_range.first) / z_delta;
                    cv::Vec3b color_bgr = colormap(normalized_z);
                    colors.emplace_back(color_bgr[2] / 255.0f, color_bgr[1] / 255.0f, color_bgr[0] / 255.0f);
                }
            }
        }
    }
    try {
        viewer.add_voxels(positions, colors);
    } catch (const std::exception& e) {
        std::cerr << "Erro ao inicializar o visualizador OpenGL: " << e.what() << std::endl;
    }

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


// Função para extrair o número do nome do arquivo
int
extract_number_from_filename(const char *filename)
{
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
int
compare_files(const void *a, const void *b)
{
    const char *file_a = *(const char**)a;
    const char *file_b = *(const char**)b;

    int num_a = extract_number_from_filename(file_a);
    int num_b = extract_number_from_filename(file_b);

    return num_a - num_b;
}


// Função para listar subdiretórios (scenes)
vector<string>
list_subdirectories(const char *dir_path)
{
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


int
main(int argc, char** argv)
{
    int screen_width = 1920; 
    int screen_height = 1080;
    float height_step = 0.2f; 
    
    // --- 1. LÓGICA MANUAL PARA O PARÂMETRO -v ---
    int wait_delay = 1; // Padrão: 1ms (máxima velocidade)
    
    // Varredura manual para garantir que pegamos o -v independente do parse_args
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v") {
            if (i + 1 < argc) {
                try {
                    wait_delay = std::stoi(argv[i + 1]);
                    if (wait_delay < 1) wait_delay = 1;
                } catch (...) {
                    std::cerr << "Aviso: valor inválido para -v." << std::endl;
                }
            }
        }
    }
    std::cout << "--- Delay entre frames: " << wait_delay << "ms ---" << std::endl;

    // --- 2. CONFIGURAÇÃO DE INPUT ---
    // (Lógica para tentar pegar input via parse_args ou manual)
    std::string input_base = "";
    auto args = parse_args(argc, argv); // Supondo que você tem essa função
    
    if (args.find("--input") != args.end()) input_base = args["--input"];
    
    if (input_base.empty()) {
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--input" && i + 1 < argc) {
                input_base = argv[i+1];
                break;
            }
        }
    }

    if (input_base.empty()) {
        std::cerr << "Erro: --input obrigatorio." << std::endl;
        return EXIT_FAILURE;
    }

    if (input_base.back() == '/') input_base.pop_back();
        std::string input = input_base;
        std::string mode = args["--mode"]; // image, points, etc.
        std::string color_map = args["--color"];

    // Configuração de cores e faixas
    if (color_map != "") color_points = get_color_map(color_map);
    else color_points = get_color_map(default_color_map);

    std::pair<float, float> x_filter_range = {-50.0f, 50.0f};
    std::pair<float, float> y_filter_range = {-50.0f, 50.0f};
    std::pair<float, float> z_filter_range = {0.0f, 3.0f};
    
    OpenGLViewer viewer(screen_width, screen_height, "Visualizador LiDAR OpenGL");
    
    // Modo Imagem Estático
    if (mode == "image") 
    {
        cv::Mat height_map = load_and_decode_bev_image(input, z_filter_range, height_step);
        visualize_pillars_from_map_opengl(viewer, height_map, x_filter_range, y_filter_range, z_filter_range);
        // Loop simples para manter janela aberta
        while(!viewer.should_close() && !viewer.get_quitting()) {
            viewer.render_once();
            cv::waitKey(10);
        }
    } 
    else // Modo Sequencial (Points / Cloud)
    {
        input = input_base + "/bin_files";

        // --- 3. INICIALIZAÇÃO ---
        vector<string> scenes = list_subdirectories(input.c_str());
        if (scenes.empty()) {
            printf("Nenhuma cena encontrada em: %s\n", input.c_str());
            return EXIT_FAILURE;
        }
        int scene_index = 0;
        while (scene_index < scenes.size()) 
        {
            printf("\nProcessando cena: %s\n", scenes[scene_index].c_str());
            string bin_scene_dir = string(input) + "/" + scenes[scene_index];
            vector<string> bin_files = list_files_with_extension(bin_scene_dir.c_str(), ".bin");
        
            if (bin_files.empty()) {
                scene_index++; continue;
            }
            
            // Ordenação
            const char **bin_files_c = (const char **)malloc(bin_files.size() * sizeof(char *));
            for (size_t i = 0; i < bin_files.size(); i++) {
                string full_path = bin_scene_dir + "/" + bin_files[i];
                bin_files_c[i] = strdup(full_path.c_str());
            }
            qsort((void*)bin_files_c, bin_files.size(), sizeof(char*), compare_files);
            
            int file_index = 0;
            // --- LOOP DOS ARQUIVOS ---
            while (file_index < bin_files.size()) {
                string bin_file_path = string(bin_files_c[file_index]);
            
                // Carrega dados
                pcl::PointCloud<pcl::PointXYZ>::Ptr pcd = load_bin_file(bin_file_path);
                if (!pcd || pcd->empty()) {
                    std::cerr << "Frame vazio ou erro de leitura." << std::endl;
                    file_index++; continue;
                }

                GroundCorrectionResult correction_result = find_and_correct_ground_plane(pcd);
                
                // Atualiza Visualizações
                const std::string winName = "Bird's-Eye View";
                cv::Mat height_map = generate_bev_from_points(pcd, x_filter_range, y_filter_range, z_filter_range, screen_height);
                cv::imshow(winName, height_map);
                cv::moveWindow(winName, screen_width/2, 0);
                
                if (mode == "points")
                    generate_bev_and_visualize_voxels_opengl(viewer, pcd, x_filter_range, y_filter_range, z_filter_range, height_step, screen_height);
                else
                    visualize_pillars_from_map_opengl(viewer, height_map, x_filter_range, y_filter_range, z_filter_range);


                // =========================================================
                // === LOOP DE TIMING NÃO-BLOQUEANTE (CRUCIAL PARA OPENGL) ===
                // =========================================================
                
                // Marca a hora que mostramos este frame
                auto start_display_time = std::chrono::high_resolution_clock::now();

                while (true)
                {
                    // 1. Renderiza OpenGL (Isso permite girar a câmera enquanto espera!)
                    viewer.render_once();

                    // 2. Processa eventos do OpenCV (Isso mantêm a janela BEV responsiva)
                    // Usamos apenas 1ms aqui para não travar o loop
                    int key = cv::waitKey(1); 

                    // 3. Controle de Input
                    if (key == 32) viewer.set_paused(!viewer.get_paused()); // Espaço
                    if (viewer.get_quitting() || key == 27) { // ESC ou fechar janela
                        // Limpeza
                        for(size_t i=0; i<bin_files.size(); i++) free((void*)bin_files_c[i]);
                        free(bin_files_c);
                        return 0;
                    }

                    // 4. Lógica de Tempo
                    if (!viewer.get_paused()) {
                        auto current_time = std::chrono::high_resolution_clock::now();
                        // Calcula quanto tempo passou em milissegundos
                        double elapsed_ms = std::chrono::duration<double, std::milli>(current_time - start_display_time).count();

                        // Se já passou o tempo definido em -v, sai do loop e vai pro próximo arquivo
                        if (elapsed_ms >= wait_delay) {
                            break; 
                        }
                    } 
                    else {
                        // Se estiver pausado, dorme um pouco para não fritar a CPU (opcional)
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                }
                // =========================================================

                file_index++;
            }
            
            for(size_t i=0; i<bin_files.size(); i++) free((void*)bin_files_c[i]);
            free(bin_files_c);
            scene_index++;
        }
    }
    
    return 0;
}