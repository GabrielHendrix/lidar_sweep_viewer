#ifndef VOXEL_REPRESENTATION_HPP
#define VOXEL_REPRESENTATION_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


// Estrutura para manter a nuvem de pontos corrigida e o ângulo de inclinação
struct GroundCorrectionResult {
    pcl::PointCloud<pcl::PointXYZ>::Ptr corrected_cloud;
    double angle_deg;
    bool success;
};

/**
 * @brief Encontra o plano do solo usando RANSAC, corrige a inclinação e a altura da nuvem de pontos.
 */
GroundCorrectionResult find_and_correct_ground_plane(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float distance_threshold = 0.05,
    int num_iterations = 1000);

/**
 * @brief (MODO POINTS) Gera um mapa de altura (BEV) diretamente dos pontos da nuvem.
 * @return O mapa de altura como uma matriz do OpenCV.
 */
cv::Mat generate_bev_from_points(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcd,
    const std::pair<float, float>& x_range,
    const std::pair<float, float>& y_range,
    const std::pair<float, float>& z_range,
    int image_size = 1024);

/**
 * @brief (MODO IMAGE) Carrega uma imagem BEV codificada, decodifica para um mapa de altura.
 * @return O mapa de altura reconstruído.
 */
cv::Mat load_and_decode_bev_image(
    const std::string& encoded_image_path,
    const std::pair<float, float>& z_range,
    float height_step);

/**
 * @brief Codifica um mapa de altura float em uma imagem RGB e a salva.
 */
void create_encoded_bev_from_height_map(
    const cv::Mat& height_map,
    const std::pair<float, float>& z_range,
    float height_step,
    const std::string& filename);


/**
 * @brief Carrega uma nuvem de pontos de um arquivo binário (.bin).
 * @return Um ponteiro para a nuvem de pontos carregada.
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr load_bin_file(const std::string& file_path);

#endif // VOXEL_REPRESENTATION_HPP
