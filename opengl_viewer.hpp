#ifndef OPENGL_VIEWER_HPP
#define OPENGL_VIEWER_HPP

#include <vector>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Eigen/Dense>
#include <thread>
class OpenGLViewer {
public:
    OpenGLViewer(int &width, int &height, const std::string& title);
    ~OpenGLViewer();

    // --- NOVOS MÉTODOS ---
    void clear_voxels(); // Limpa a tela (logicamente)
    void update_voxels(const std::vector<Eigen::Vector3f>& positions, const std::vector<Eigen::Vector3f>& colors); // Atualiza dados sem recriar buffers
    void render_once(); // Renderiza um frame e retorna (não bloqueante)
    bool should_close(); // Helper para checar se o usuário fechou a janela
    // ---------------------

    void add_voxels(const std::vector<Eigen::Vector3f>& positions, const std::vector<Eigen::Vector3f>& colors);
    void run();
    // Verifica se está pausado
    bool get_paused() const { return is_paused; }
    bool get_quitting() const { return is_quitting; }
    
    // Força um estado (opcional)
    void set_paused(bool paused) { is_paused = paused; }

    // ... outros métodos (render_once, update_voxels) ...

private:
    // Variável de estado
    bool is_paused;
    bool is_quitting;

    GLFWwindow* window;
    GLuint shader_program, vao, vbo_vertices, vbo_positions, vbo_colors;
    size_t num_voxels;

    Eigen::Matrix4f projection;
    Eigen::Matrix4f view;

    // --- Variáveis de Controle da Câmera (Estilo Open3D) ---
    Eigen::Vector3f camera_target; // O ponto para onde a câmera olha
    float camera_distance;         // Raio (distância do alvo)
    float camera_azimuth;          // Ângulo horizontal (radianos)
    float camera_elevation;        // Ângulo vertical (radianos)
    
    // Estado do Mouse
    bool is_dragging_rotate;
    bool is_dragging_pan;
    double last_mouse_x, last_mouse_y;

    // Camera parameters
    float camera_radius;
    float camera_theta;
    float camera_phi;

    // Mouse state
    bool is_mouse_dragging;

    void init_glfw(int &width, int &height, const std::string& title);
    void init_glew();
    void init_shaders();
    void init_buffers(const std::vector<Eigen::Vector3f>& positions, const std::vector<Eigen::Vector3f>& colors);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void error_callback(int error, const char* description);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void window_size_callback(GLFWwindow* window, int width, int height);
    void update_view_matrix();
    void ensure_static_buffers();
};

#endif // OPENGL_VIEWER_HPP
