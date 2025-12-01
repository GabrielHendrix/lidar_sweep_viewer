#include "opengl_viewer.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Shaders permanecem iguais...
const char* vertex_shader_source = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aInstancePosition;
layout (location = 2) in vec3 aInstanceColor;

out vec3 fColor;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    gl_Position = projection * view * vec4(aPos + aInstancePosition, 1.0);
    fColor = aInstanceColor;
}
)";

const char* fragment_shader_source = R"(
#version 330 core
out vec4 FragColor;

in vec3 fColor;

void main()
{
    FragColor = vec4(fColor, 1.0f);
}
)";

OpenGLViewer::OpenGLViewer(int &width, int &height, const std::string& title)
    : window(nullptr), shader_program(0), vao(0), vbo_vertices(0), vbo_positions(0), vbo_colors(0), num_voxels(0), is_paused(false), is_quitting(false),
      // Inicializa parâmetros da câmera
      camera_target(0.0f, 0.0f, 0.0f),
      camera_distance(100.0f),
      camera_azimuth(0.0f),
      camera_elevation(M_PI / 4.0f), // 45 graus
      is_dragging_rotate(false),
      is_dragging_pan(false),
      last_mouse_x(0), last_mouse_y(0)
{
    init_glfw(width, height, title);
    init_glew();
    init_shaders();
    
    // Inicializa matrizes com identidade para evitar lixo de memória antes do loop
    projection = Eigen::Matrix4f::Identity();
    view = Eigen::Matrix4f::Identity();
}

OpenGLViewer::~OpenGLViewer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_vertices);
    glDeleteBuffers(1, &vbo_positions);
    glDeleteBuffers(1, &vbo_colors);
    glDeleteProgram(shader_program);
    glfwDestroyWindow(window);
    glfwTerminate();
}

void OpenGLViewer::error_callback(int error, const char* description) {
    std::cerr << "GLFW Error: " << description << std::endl;
}

void OpenGLViewer::init_glfw(int &width, int &height, const std::string& title) {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwSetErrorCallback(error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    // 1. Pega o monitor principal
    GLFWmonitor* primary_monitor = glfwGetPrimaryMonitor();
    
    // 2. Pega o "Video Mode" (contém resolução, refresh rate, etc)
    const GLFWvidmode* mode = glfwGetVideoMode(primary_monitor);

    // Agora você sabe o tamanho da tela do usuário:
    // mode->width  = Largura da tela (ex: 1920)
    // mode->height = Altura da tela (ex: 1080)

    std::cout << "Resolucao do Monitor: " << mode->width << "x" << mode->height << std::endl;

    // 3. Calcula a posição para CENTRALIZAR
    // A conta é: (LarguraTela - LarguraJanela) / 2
    int x_pos = (mode->width - width) / 2;
    int y_pos = (mode->height - height) / 2;
    width = mode->width;
    height = mode->height;
    
    window = glfwCreateWindow(width/2, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    // 4. Define a posição inicial
    glfwSetWindowPos(window, 0, 0);

    // -------------------------------------

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // --- Configuração dos Callbacks ---
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, window_size_callback);
}

// --- Implementação dos Callbacks de Input ---
void OpenGLViewer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Recupera a instância da classe
    OpenGLViewer* viewer = static_cast<OpenGLViewer*>(glfwGetWindowUserPointer(window));
    if (!viewer) return;

    // Se apertar ESPAÇO e for o evento de "Pressionar" (ignora se segurar a tecla)
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        viewer->is_paused = !viewer->is_paused; // Inverte o estado
        std::cout << (viewer->is_paused ? "PAUSADO" : "RODANDO") << std::endl;
    }
    // Se apertar ESC e for o evento de "Pressionar" (ignora se segurar a tecla)
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        viewer->is_quitting = !viewer->is_quitting; // Inverte o estado
        std::cout << (viewer->is_quitting ? "FECHANDO" : "RODANDO") << std::endl;
    }
}


void OpenGLViewer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    OpenGLViewer* viewer = static_cast<OpenGLViewer*>(glfwGetWindowUserPointer(window));
    if (!viewer) return;

    double x, y;
    glfwGetCursorPos(window, &x, &y);
    viewer->last_mouse_x = x;
    viewer->last_mouse_y = y;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        // Open3D: Shift + Left = Pan, Left = Rotate
        if (mods & GLFW_MOD_SHIFT) {
            viewer->is_dragging_pan = (action == GLFW_PRESS);
            viewer->is_dragging_rotate = false;
        } else {
            viewer->is_dragging_rotate = (action == GLFW_PRESS);
            viewer->is_dragging_pan = false;
        }
    } 
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        viewer->is_dragging_pan = (action == GLFW_PRESS);
        viewer->is_dragging_rotate = false;
    }
}

void OpenGLViewer::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    OpenGLViewer* viewer = static_cast<OpenGLViewer*>(glfwGetWindowUserPointer(window));
    if (!viewer) return;

    double dx = xpos - viewer->last_mouse_x;
    double dy = ypos - viewer->last_mouse_y;
    viewer->last_mouse_x = xpos;
    viewer->last_mouse_y = ypos;

    if (viewer->is_dragging_rotate) {
        // Sensibilidade da rotação
        float sensitivity = 0.005f;
        viewer->camera_azimuth -= dx * sensitivity;
        viewer->camera_elevation += dy * sensitivity;

        // Limita a elevação para não "capotar" a câmera (gimbal lock prevention)
        // Mantém entre -89 e 89 graus aproximadamente
        if (viewer->camera_elevation > 1.55f) viewer->camera_elevation = 1.55f;
        if (viewer->camera_elevation < -1.55f) viewer->camera_elevation = -1.55f;
    }
    
    if (viewer->is_dragging_pan) {
        // Pan move o 'target' baseado nos vetores Right e Up da câmera
        float sensitivity = viewer->camera_distance * 0.001f; // Pan deve escalar com a distância
        
        // Precisamos recalcular os vetores de base da câmera para fazer o Pan correto
        float r = viewer->camera_distance;
        float x_cam = r * sin(viewer->camera_elevation) * cos(viewer->camera_azimuth);
        float y_cam = r * sin(viewer->camera_elevation) * sin(viewer->camera_azimuth); // Dependendo do eixo UP
        // Vamos usar a convenção: Z é Up no mundo (comum em LiDAR), mas OpenGL Y é Up.
        // Vamos assumir Z Up do mundo para calcular a posição:
        
        // Convertendo esféricas para cartesianas (assumindo Z-up)
        Eigen::Vector3f cam_dir_inv;
        cam_dir_inv.x() = cos(viewer->camera_elevation) * cos(viewer->camera_azimuth);
        cam_dir_inv.y() = cos(viewer->camera_elevation) * sin(viewer->camera_azimuth);
        cam_dir_inv.z() = sin(viewer->camera_elevation);
        
        Eigen::Vector3f z_axis = -cam_dir_inv; // Câmera olha para -Z local
        Eigen::Vector3f world_up(0, 0, 1);
        Eigen::Vector3f x_axis = world_up.cross(z_axis).normalized(); // Right
        Eigen::Vector3f y_axis = z_axis.cross(x_axis).normalized();   // Up da camera

        // Move o target na direção oposta do movimento do mouse
        viewer->camera_target += x_axis * dx * sensitivity; // Movimento horizontal do mouse -> Eixo X da camera
        viewer->camera_target += y_axis * dy * sensitivity; // Movimento vertical do mouse -> Eixo Y da camera
    }
}

void OpenGLViewer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    OpenGLViewer* viewer = static_cast<OpenGLViewer*>(glfwGetWindowUserPointer(window));
    if (!viewer) return;

    // Zoom estilo Open3D: diminui ou aumenta o raio
    float zoom_sensitivity = 0.1f;
    viewer->camera_distance -= viewer->camera_distance * yoffset * zoom_sensitivity;
    
    if (viewer->camera_distance < 0.1f) viewer->camera_distance = 0.1f;
}

void OpenGLViewer::window_size_callback(GLFWwindow* window, int width, int height) {
     OpenGLViewer* viewer = static_cast<OpenGLViewer*>(glfwGetWindowUserPointer(window));
     if (!viewer) return;
     glViewport(0, 0, width, height);
}

// --- Lógica de Visualização ---

void OpenGLViewer::init_glew() {
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLEW");
    }
}

void OpenGLViewer::init_shaders() {
    // (Mesmo código anterior...)
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
    glCompileShader(vertex_shader);
    // ... verificação de erro ...
    int success;
    char info_log[512];
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertex_shader, 512, nullptr, info_log);
        throw std::runtime_error("Vertex shader error");
    }

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
    glCompileShader(fragment_shader);
    // ... verificação de erro ...
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragment_shader, 512, nullptr, info_log);
        throw std::runtime_error("Fragment shader error");
    }

    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);
    
    // ... verificação de erro ...
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if(!success) throw std::runtime_error("Link error");

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
}

void OpenGLViewer::init_buffers(const std::vector<Eigen::Vector3f>& positions, const std::vector<Eigen::Vector3f>& colors) {
    // (Mesmo código anterior...)
    float vertices[] = {
        -0.1f, -0.1f, -0.1f, 0.1f, -0.1f, -0.1f, 0.1f,  0.1f, -0.1f, -0.1f,  0.1f, -0.1f,
        -0.1f, -0.1f,  0.1f, 0.1f, -0.1f,  0.1f, 0.1f,  0.1f,  0.1f, -0.1f,  0.1f,  0.1f,
    };
    unsigned int indices[] = {
        0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 0, 4, 7, 7, 3, 0,
        1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3, 0, 1, 5, 5, 4, 0
    };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    glGenBuffers(1, &vbo_vertices);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glGenBuffers(1, &vbo_positions);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_positions);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(Eigen::Vector3f), positions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    glGenBuffers(1, &vbo_colors);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f), colors.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// void OpenGLViewer::add_voxels(const std::vector<Eigen::Vector3f>& positions, const std::vector<Eigen::Vector3f>& colors) {
//     num_voxels = positions.size();
//     init_buffers(positions, colors);
// }

// Cálculo da matriz View baseada no estado atual
void OpenGLViewer::update_view_matrix() {
    // Converte Coordenadas Esféricas para Cartesianas
    // Assumindo Z-up
    float x = camera_distance * cos(camera_elevation) * cos(camera_azimuth);
    float y = camera_distance * cos(camera_elevation) * sin(camera_azimuth);
    float z = camera_distance * sin(camera_elevation);

    Eigen::Vector3f camera_offset(x, y, z);
    Eigen::Vector3f camera_pos = camera_target + camera_offset;
    Eigen::Vector3f world_up(0.0f, 0.0f, 1.0f); // Z é para cima

    // Constrói a matriz LookAt manualmente (ou usar biblioteca se preferir)
    Eigen::Vector3f z_axis = (camera_pos - camera_target).normalized(); // Forward (aponta para trás da câmera)
    Eigen::Vector3f x_axis = world_up.cross(z_axis).normalized();       // Right
    Eigen::Vector3f y_axis = z_axis.cross(x_axis).normalized();         // Up local da câmera

    Eigen::Matrix4f look_at = Eigen::Matrix4f::Identity();
    
    // Rotação
    look_at(0,0) = x_axis.x(); look_at(0,1) = x_axis.y(); look_at(0,2) = x_axis.z();
    look_at(1,0) = y_axis.x(); look_at(1,1) = y_axis.y(); look_at(1,2) = y_axis.z();
    look_at(2,0) = z_axis.x(); look_at(2,1) = z_axis.y(); look_at(2,2) = z_axis.z();

    // Translação
    look_at(0,3) = -x_axis.dot(camera_pos);
    look_at(1,3) = -y_axis.dot(camera_pos);
    look_at(2,3) = -z_axis.dot(camera_pos);

    view = look_at;
}


// 1. Limpar Voxels
void OpenGLViewer::clear_voxels() {
    num_voxels = 0;
    // Não precisamos deletar a memória da GPU, apenas dizemos ao draw call para desenhar 0 instâncias
}

// 2. Atualizar Voxels (Otimizado)
void OpenGLViewer::update_voxels(const std::vector<Eigen::Vector3f>& positions, const std::vector<Eigen::Vector3f>& colors) {
    num_voxels = positions.size();
    if (num_voxels == 0) return;

    // Se os buffers ainda não existem (primeira chamada), cria tudo
    if (vao == 0) {
        init_buffers(positions, colors);
        return;
    }

    glBindVertexArray(vao);

    // Atualiza Posições (VBO Index 1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_positions);
    // GL_DYNAMIC_DRAW avisa a GPU que esses dados vão mudar muito
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(Eigen::Vector3f), positions.data(), GL_DYNAMIC_DRAW);
    
    // Atualiza Cores (VBO Index 2)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f), colors.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// 3. Renderizar um único frame (Não bloqueante)
void OpenGLViewer::render_once() {
    glEnable(GL_DEPTH_TEST);
    
    if (glfwWindowShouldClose(window)) return;

    // --- Lógica de Renderização (Extraída do run original) ---
    
    // 1. Setup do Viewport e Projeção
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    if (height == 0) height = 1;
    float aspect_ratio = (float)width / (float)height;
    glViewport(0, 0, width, height);

    float near_plane = 0.1f;
    float far_plane = 1000.0f;
    float fov = 45.0f;
    float f = 1.0f / tan(fov * M_PI / 360.0);
    
    projection.setZero();
    projection(0,0) = f / aspect_ratio;
    projection(1,1) = f;
    projection(2,2) = (far_plane + near_plane) / (near_plane - far_plane);
    projection(2,3) = (2.0f * far_plane * near_plane) / (near_plane - far_plane);
    projection(3,2) = -1.0f;

    // 2. Atualiza Câmera
    update_view_matrix();

    // 3. Limpa Tela
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 4. Desenha
    glUseProgram(shader_program);
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, projection.data());
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, view.data());

    if (num_voxels > 0) {
        glBindVertexArray(vao);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, num_voxels);
        glBindVertexArray(0);
    }

    // 5. Troca Buffers e Processa Eventos (Mouse/Teclado ainda funcionam!)
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool OpenGLViewer::should_close() {
    return glfwWindowShouldClose(window);
}

void OpenGLViewer::add_voxels(const std::vector<Eigen::Vector3f>& positions, const std::vector<Eigen::Vector3f>& colors) {
    update_voxels(positions, colors);
}

void OpenGLViewer::run() {
    glEnable(GL_DEPTH_TEST);
    while (!should_close() && is_paused && !is_quitting) {
        render_once();
    }
}