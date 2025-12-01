////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                //
//                         #####              #####       #####	  ######         #########   #################                    //
//                        #####              #####       #####	 ########      ##########   #################                     //
//                       #####              #####       #####	##########   ###########   #####                                  //
//                      #####              #####       #####   #####  ##########  #####   ###############                         //
//                     #####              #####       #####   #####    #######   #####   ###############                          //
//                    #####              #####	     #####   #####     ####     #####   #####                                     //
//                   #####              #####       #####   #####              #####   #####                                      //
//                  ################   #################   #####              #####   #################                           //
//                 ################   #################   #####              #####   #################                            //
//                                                                                                                                //
//                     ##########  ##########  ########    ##########   ###     ###  ##########  ##########                       //
//                    ###    ###  ###    ###  ###    ###  ###    ### #######   ###  ###    ###  ###                               //
//                   ##########  ###    ###  ##########  ###    ###   ###     ###  ###           #####                            //
//                  ###  ###    ###    ###  ###    ###  ###    ###   ###     ###  ###    ###         ###                          //
//                 ###    ###  ##########  #########   ##########   ######  ###  ##########  ##########                           //
//                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <astro/task_manager_interface.h>
#include <astro/velodyne_interface.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv/cv.hpp>
#include <fstream>
#include <tf.h>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
// #include <cstdlib>
// #include <numeric>
#include <cmath>
// #include <iostream>
// #include <fstream>

using namespace std;

#define RESOLUTION 27.34
#define MAX_NUMBER_OF_LIDARS_NEURAL 10   // 15 is the Maximum number of astro_lidar_variable_scan_message defined, so is the maximun number of lidars

bool lidars_inserted_in_global_points[MAX_NUMBER_OF_LIDARS_NEURAL] = { false };
astro_localize_ackerman_globalpos_message *globalpos_msg = NULL;
astro_lidar_config lidar_config[MAX_NUMBER_OF_LIDARS_NEURAL];
bool lidars_alive[MAX_NUMBER_OF_LIDARS_NEURAL] = { false };
astro_pose_3D_t lidar_pose;
// astro_pose_3D_t board_pose;
astro_pose_3D_t choosed_sensor_referenced[3] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
astro_pose_3D_t board_pose[3] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

int is_localized = 0;
int stretch_y = 0;
int save = 1;
vector<int> lidar_index = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int bin_index = 0;

tf::Transformer transformer(false);
astro_semi_trailers_config_t semi_trailer_config;

std::string output_dir = "/home/lume/Desktop/range_images";


astro_pose_3D_t
compute_new_rear_bullbar_from_beta(astro_pose_3D_t rear_bullbar_pose, double beta[MAX_NUM_TRAILERS], astro_semi_trailers_config_t semi_trailer_config)
{
	// Linha adicionada após a mudança de beta para trailer_theta. A variável beta recebida pela função é trailer_theta, a linha abaixo transforma em beta novamente para funcionar nas fórmulas
//	beta = convert_theta1_to_beta(rear_bullbar_pose.orientation.yaw, beta);
	//
	
	double current_beta = -beta[0];
	astro_pose_3D_t temp_rear_bullbar_pose;
	temp_rear_bullbar_pose.position.x 			= -semi_trailer_config.semi_trailers[0].M + rear_bullbar_pose.position.x * cos(current_beta) - rear_bullbar_pose.position.y * sin(current_beta);
	temp_rear_bullbar_pose.position.y 			= 						   rear_bullbar_pose.position.x * sin(current_beta) + rear_bullbar_pose.position.y * cos(current_beta);

//	temp_rear_bullbar_pose.position.x 			= rear_bullbar_pose.position.x * cos(beta) - rear_bullbar_pose.position.y * sin(beta);
//	temp_rear_bullbar_pose.position.y 			= rear_bullbar_pose.position.x * sin(beta) + rear_bullbar_pose.position.y * cos(beta);

	temp_rear_bullbar_pose.position.z 			= rear_bullbar_pose.position.z;
	temp_rear_bullbar_pose.orientation.pitch 	= rear_bullbar_pose.orientation.pitch;
	temp_rear_bullbar_pose.orientation.roll 	= rear_bullbar_pose.orientation.roll;
	temp_rear_bullbar_pose.orientation.yaw 		= astro_normalize_theta(rear_bullbar_pose.orientation.yaw + current_beta); // Verificar se soma o beta ou subtrai

	return (temp_rear_bullbar_pose);
}


bool
isValidPose(astro_pose_3D_t pose)
{
	return (!std::isnan(pose.position.x) && !std::isnan(pose.position.y) && !std::isnan(pose.position.z) &&
			!std::isnan(pose.orientation.yaw) && !std::isnan(pose.orientation.pitch) && !std::isnan(pose.orientation.roll));
}


tf::Point
move_to_board_reference(tf::Point p3d_lidar_reference, astro_pose_3D_t lidar_pose, astro_pose_3D_t car_pose, int lidar_sensor_ref)
{	
	if (!isValidPose(lidar_pose) || !isValidPose(choosed_sensor_referenced[lidar_sensor_ref]) || !isValidPose(car_pose))
	{
		return tf::Point(0, 0, 0); // Retornar um ponto padrão ou lidar com o erro de forma apropriada
	}

	// tf::Transform pose_car_in_world(
    //         tf::Quaternion(car_pose.orientation.yaw, car_pose.orientation.pitch, car_pose.orientation.roll),
    //         tf::Vector3(car_pose.position.x, car_pose.position.y, car_pose.position.z));
	tf::Transform pose_car_in_world(
		tf::Quaternion(car_pose.orientation.yaw, car_pose.orientation.pitch, car_pose.orientation.roll),
		tf::Vector3(car_pose.position.x, car_pose.position.y, car_pose.position.z));

	tf::StampedTransform car_to_board_transform(pose_car_in_world, tf::Time(0), "/world", "/car");
	transformer.setTransform(car_to_board_transform, "car_to_board_transform");
	
	tf::Transform pose_board_in_car(
		tf::Quaternion(choosed_sensor_referenced[lidar_sensor_ref].orientation.yaw, choosed_sensor_referenced[lidar_sensor_ref].orientation.pitch, choosed_sensor_referenced[lidar_sensor_ref].orientation.roll),
		tf::Vector3(choosed_sensor_referenced[lidar_sensor_ref].position.x, choosed_sensor_referenced[lidar_sensor_ref].position.y, choosed_sensor_referenced[lidar_sensor_ref].position.z));

	tf::StampedTransform board_to_car_transform(pose_board_in_car, tf::Time(0), "/car", "/board");
	transformer.setTransform(board_to_car_transform, "board_to_car_transform");
	
    tf::Transform pose_lidar_in_board(
            tf::Quaternion(lidar_pose.orientation.yaw, lidar_pose.orientation.pitch, lidar_pose.orientation.roll),
            tf::Vector3(lidar_pose.position.x, lidar_pose.position.y, lidar_pose.position.z));

	tf::StampedTransform lidar_to_board_transform(pose_lidar_in_board, tf::Time(0), "/board", "/lidar");
	transformer.setTransform(lidar_to_board_transform, "lidar_to_board_transform");

	tf::StampedTransform lidar_to_board_pose;
	
	transformer.lookupTransform("/car", "/lidar", tf::Time(0), lidar_to_board_pose);

	return lidar_to_board_pose * p3d_lidar_reference;
}


void
show_lidar_sweep(vector<vector<double>> distance, int &index, int color_mode = 0)
{
    int width = distance.size();
	printf("width: %d \n", width);
    if (width == 0) return;
    int height = distance[0].size();

    if (color_mode == 0) {
        // Modo original: escala de cinza
        cv::Mat img(height, width, CV_8UC1);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double distance_value = distance[x][y];
                if (distance_value == 0) {
                    img.at<uint8_t>(y, x) = 0;
                } else {
                    img.at<uint8_t>(y, x) = (uint8_t)(255 - ((distance_value * 100) / RESOLUTION));
                }
            }
        }
        if (stretch_y == 1) {
            cv::Mat stretched_img;
            int stretch_factor = 5;
            cv::resize(img, stretched_img, cv::Size(width, height * stretch_factor), 0, 0, cv::INTER_LINEAR);
            img = stretched_img;
        }
        cv::imshow("Lidar Sweep Visualization", img);
        cv::waitKey(50);
        if (save == 1) {
            char filename[100];
            sprintf(filename, "lidar_sweep_%04d.png", index);
            cv::imwrite(filename, img);
            index++;
        }
    } else {
        // Novo modo: RGB, cada distância recebe uma cor única
        cv::Mat img_rgb(height, width, CV_8UC3);
        double min_dist = 1e9, max_dist = -1e9;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double d = distance[x][y];
                if (d > 0) {
                    if (d < min_dist) min_dist = d;
                    if (d > max_dist) max_dist = d;
                }
            }
        }
        if (min_dist == max_dist) max_dist = min_dist + 1.0;
        // Salva min_dist e max_dist em arquivo de metadados para reconstrução fiel no Python
        char meta_filename[256];
        sprintf(meta_filename, "%s/lidar_sweep_rgb_%04d_meta.txt", output_dir.c_str(), index);
        std::ofstream meta_file(meta_filename);
        meta_file << min_dist << " " << max_dist << std::endl;
        meta_file.close();
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double d = distance[x][y];
                cv::Vec3b color(0,0,0);
                if (d == 0) {
                    color = cv::Vec3b(0,0,0);
                } else {
                    // Gradiente: azul (perto) -> verde -> vermelho (longe)
                    double norm = (d - min_dist) / (max_dist - min_dist);
                    int r = (int)(255 * norm);
                    int g = (int)(255 * (1.0 - norm));
                    int b = (int)(255 * (0.5 - fabs(norm-0.5)) * 2.0); // azul no meio
                    color = cv::Vec3b(b,g,r);
                }
                img_rgb.at<cv::Vec3b>(y, x) = color;
            }
        }
        if (stretch_y == 1) {
            cv::Mat stretched_img;
            int stretch_factor = 1;
            cv::resize(img_rgb, stretched_img, cv::Size(width, height * stretch_factor), 0, 0, cv::INTER_LINEAR);
            img_rgb = stretched_img;
        }
        cv::imshow("Lidar Sweep Visualization RGB", img_rgb);
        cv::waitKey(50);
        if (save == 1) {
            char img_filename[256];
            sprintf(img_filename, "%s/lidar_sweep_rgb_%04d.png", output_dir.c_str(), index);
            cv::imwrite(img_filename, img_rgb);
            index++;
        }
    }
}


tf::Point
spherical_to_cartesian(double hangle, double vangle, double range)
{
	double x, y, z;

	x = range * cos(vangle) * cos(hangle);
	y = range * cos(vangle) * sin(hangle);
	z = range * sin(vangle);

	return tf::Point(x, y, z);
}


vector<vector<double>>
take_lidar_sweep(astro_velodyne_variable_scan_message lidar_message, astro_lidar_config lidar_config)
{
	vector<vector<double>> distances;
	double range = 0.0;
	// vector<tf::Point> sweep_lidar;
	string bin_name = to_string(bin_index);
	if (bin_name.length() == 1)
		bin_name = "000" + bin_name;
	else if (bin_name.length() == 2)
		bin_name = "00" + bin_name;
	else if (bin_name.length() == 3)
		bin_name = "0" + bin_name;

	// Cria a nuvem de pontos do tipo XYZ + intensidade
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
	// std::ofstream fs("/home/lume/Desktop/velodyne_0/" + bin_name + ".bin", std::ios::out | std::ios::binary | std::ios::app);
	// for (int col = 0; col < 1024; col++)
	for (int col = 0; col < lidar_message.number_of_shots; col++)
	{
		vector<double> line_d;
		for (int line = 0; line < lidar_message.partial_scan[col].shot_size; line++)
		{
			double horizontal_angle = 0.0f, vertical_angle = 0.0f;

			if (strstr(lidar_config.model, "OT-128") != NULL)
				horizontal_angle = astro_normalize_theta(-astro_degrees_to_radians(lidar_message.partial_scan[col].angle + lidar_config.horizontal_angles_deltas[line]));
			else
				horizontal_angle = astro_normalize_theta(-astro_degrees_to_radians(lidar_message.partial_scan[col].angle));

			range = (((double) lidar_message.partial_scan[col].distance[line]) / lidar_config.range_division_factor);
			
            // if (range <= init_range_for_valid_point_cloud || range >= end_range_for_valid_point_cloud)
            //     continue;

            vertical_angle = astro_normalize_theta(astro_degrees_to_radians(lidar_config.vertical_angles[line]));

            tf::Point p3d_lidar_reference = spherical_to_cartesian(horizontal_angle, vertical_angle, range);
			// sweep_lidar.push_back(p3d_lidar_reference);

			// float x = (float) p3d_lidar_reference.x();
			// float y = (float) p3d_lidar_reference.y();
			// float z = (float) p3d_lidar_reference.z();
			// float w = (float) lidar_message.partial_scan[col].intensity[line];
			
			// // fs << x << " " << y << " " << z << " " << w << std::endl;
			// fs.write(reinterpret_cast<const char*>(&x), sizeof x);
			// fs.write(reinterpret_cast<const char*>(&y), sizeof y);
			// fs.write(reinterpret_cast<const char*>(&z), sizeof z);
			// fs.write(reinterpret_cast<const char*>(&w), sizeof w);


			// point.x = static_cast<float>(p3d_lidar_reference.z());   // Z → X
			// point.y = -static_cast<float>(p3d_lidar_reference.x());  // -X → Y
			// point.z = -static_cast<float>(p3d_lidar_reference.y());  // -Y → Z
			// tf::Transform pose_lidar_in_board(tf::Quaternion(lidar_config.pose.orientation.yaw, lidar_config.pose.orientation.pitch, lidar_config.pose.orientation.roll),
        	// 								  tf::Vector3(lidar_config.pose.position.x, lidar_config.pose.position.y, lidar_config.pose.position.z));
			
			// tf::StampedTransform lidar_to_board_transform(pose_lidar_in_board, tf::Time(0), "/board", "/lidar");
			// transformer.setTransform(lidar_to_board_transform, "lidar_to_board_transform");

			// tf::StampedTransform lidar_to_board_pose;

			// transformer.lookupTransform("/board", "/lidar", tf::Time(0), lidar_to_board_pose);

			// tf::Point p3d_board_reference = lidar_to_board_pose * p3d_lidar_reference;
			tf::Point p3d_board_reference = move_to_board_reference(p3d_lidar_reference, lidar_config.pose, globalpos_msg->pose, lidar_config.sensor_reference);


			pcl::PointXYZI point;
			point.x = static_cast<float>(p3d_board_reference.x());
			point.y = static_cast<float>(p3d_board_reference.y());
			point.z = static_cast<float>(p3d_board_reference.z());
			point.intensity = static_cast<float>(lidar_message.partial_scan[col].intensity[line]);

			cloud->points.push_back(point);
			// int size = abs(1024 / lidar_message.partial_scan[col].shot_size);
			// int size = abs(lidar_message.number_of_shots / lidar_message.partial_scan[col].shot_size);
			// for (int i = 0; i < 32; i++)
			line_d.push_back(range);
		}

		distances.push_back(line_d);
	}

	// fs.close();

	// Define as dimensões da nuvem
	cloud->width = cloud->points.size();
	cloud->height = 1;
	cloud->is_dense = false;


		// Caminho para salvar o arquivo
	std::string ply_path = "/home/lume/Desktop/velodyne_0/" + bin_name + ".ply";

	// Escreve no formato .ply (ASCII = false → human-readable; true = binário)
	pcl::PLYWriter writer;
	writer.write<pcl::PointXYZI>(ply_path, *cloud, false);  // false = ASCII, true = binário
	bin_index += 1;

    // Exporta parâmetros relevantes para reconstrução fiel da nuvem
    char params_filename[256];
    sprintf(params_filename, "%s/lidar_sweep_rgb_%s_params.txt", output_dir.c_str(), bin_name.c_str());
    std::ofstream params_file(params_filename);
    params_file << "model " << lidar_config.model << std::endl;
    params_file << "range_division_factor " << lidar_config.range_division_factor << std::endl;
    params_file << "number_of_shots " << lidar_message.number_of_shots << std::endl;
    params_file << "vertical_angles ";
    for (int i = 0; i < lidar_message.partial_scan[0].shot_size; i++)
        params_file << lidar_config.vertical_angles[i] << " ";
    params_file << std::endl;
    params_file << "horizontal_angles_deltas ";
    for (int i = 0; i < lidar_message.partial_scan[0].shot_size; i++)
        params_file << lidar_config.horizontal_angles_deltas[i] << " ";
    params_file << std::endl;
    params_file.close();

	return distances;
}


void
variable_scan_message_handler0(astro_velodyne_variable_scan_message *message)
{
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[0]);
	show_lidar_sweep(distances, lidar_index[0], 1);
}


void
variable_scan_message_handler1(astro_velodyne_variable_scan_message *message)
{
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[1]);
	show_lidar_sweep(distances, lidar_index[1]);
}


void
variable_scan_message_handler2(astro_velodyne_variable_scan_message *message)
{
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[2]);
	show_lidar_sweep(distances, lidar_index[2]);
}


void
variable_scan_message_handler3(astro_velodyne_variable_scan_message *message)
{	
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[3]);
	show_lidar_sweep(distances, lidar_index[3]);
}


void
variable_scan_message_handler4(astro_velodyne_variable_scan_message *message)
{	
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[4]);
	show_lidar_sweep(distances, lidar_index[4]);
}


void
variable_scan_message_handler5(astro_velodyne_variable_scan_message *message)
{
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[5]);
	show_lidar_sweep(distances, lidar_index[5], 1);
}


void
variable_scan_message_handler6(astro_velodyne_variable_scan_message *message)
{
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[6]);
	show_lidar_sweep(distances, lidar_index[6]);
}


void
variable_scan_message_handler7(astro_velodyne_variable_scan_message *message)
{
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[7]);
	show_lidar_sweep(distances, lidar_index[7]);
}


void
variable_scan_message_handler8(astro_velodyne_variable_scan_message *message)
{
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[8]);
	show_lidar_sweep(distances, lidar_index[8]);
}


void
variable_scan_message_handler9(astro_velodyne_variable_scan_message *message)
{
	astro_velodyne_variable_scan_message lidar_msg;
	
	lidar_msg.host = message->host;
	lidar_msg.number_of_shots = message->number_of_shots;
	lidar_msg.partial_scan = message->partial_scan;
	lidar_msg.timestamp = message->timestamp;

	vector<vector<double>> distances = take_lidar_sweep(lidar_msg, lidar_config[9]);
	show_lidar_sweep(distances, lidar_index[9]);
}


void
localize_ackerman_globalpos_message_handler(astro_localize_ackerman_globalpos_message *msg)
{
	globalpos_msg = msg;
	if(globalpos_msg != NULL)
	{
		is_localized = 1;
		for (int ref_index = 0; ref_index < 3; ref_index++)
		{
			if (semi_trailer_config.num_semi_trailers != 0 && ref_index == 2)
			{
				double beta[semi_trailer_config.num_semi_trailers] = {0.0};
				for (int i = 0; i < semi_trailer_config.num_semi_trailers; i++)
				{
					if (i == 0)
						beta[i] = convert_theta1_to_beta(globalpos_msg->pose.orientation.yaw, globalpos_msg->trailer_theta[i]);
					else
						beta[i] = convert_theta1_to_beta(globalpos_msg->trailer_theta[i - 1], globalpos_msg->trailer_theta[i]);
				}
				choosed_sensor_referenced[ref_index] = compute_new_rear_bullbar_from_beta(board_pose[ref_index], beta, semi_trailer_config);
			}
			else{
				choosed_sensor_referenced[ref_index] = board_pose[ref_index];
			}
		}
		
	}
}


void
shutdown_module(int signo)
{
    if (signo == SIGINT) {
        astro_ipc_disconnect();
        cvDestroyAllWindows();

        printf("Lidar Sweep Viewer: Disconnected.\n");
        exit(0);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////


void
read_parameters(int argc, char **argv)
{
	int parameter_number = 1;
	if ((argc < parameter_number))
		astro_die("%s: Wrong number of parameters. lidar_sweep_viewer requires at least %d parameters and received %d. \n Usage: %s -lidar <lidar_number> \n>", argv[0], parameter_number, argc - 1, argv[0]);

	

	astro_param_allow_unfound_variables(0);
	astro_param_t param_list[] =
	{
		{(char *) "sensor_board_1", (char*) "x",     		ASTRO_PARAM_DOUBLE, &board_pose[0].position.x, 0, NULL },
		{(char *) "sensor_board_1", (char*) "y",     		ASTRO_PARAM_DOUBLE, &board_pose[0].position.y, 0, NULL },
		{(char *) "sensor_board_1", (char*) "z",     		ASTRO_PARAM_DOUBLE, &board_pose[0].position.z, 0, NULL },
		{(char *) "sensor_board_1", (char*) "roll",  		ASTRO_PARAM_DOUBLE, &board_pose[0].orientation.roll, 0, NULL},
		{(char *) "sensor_board_1", (char*) "pitch", 		ASTRO_PARAM_DOUBLE, &board_pose[0].orientation.pitch, 0, NULL},
		{(char *) "sensor_board_1", (char*) "yaw",   		ASTRO_PARAM_DOUBLE, &board_pose[0].orientation.yaw, 0, NULL},
		{(char *) "semi_trailer", (char *) "initial_type", 	ASTRO_PARAM_INT, &(semi_trailer_config.num_semi_trailers), 0, NULL},
	
	};

	astro_param_install_params(argc, argv, param_list, sizeof(param_list) / sizeof(param_list[0]));
 	
	astro_param_allow_unfound_variables(1);
   	astro_param_t optional_commandline_param_list[] =
   	{
   		{(char *) "commandline", (char *) "stretch", 	ASTRO_PARAM_ONOFF, &stretch_y, 0, NULL},
		{(char *) "commandline", (char *) "save", 		ASTRO_PARAM_ONOFF, &save, 0, NULL},
		{(char *) "rear_bullbar", (char*) "x",     		ASTRO_PARAM_DOUBLE, &board_pose[2].position.x, 0, NULL },
		{(char *) "rear_bullbar", (char*) "y",     		ASTRO_PARAM_DOUBLE, &board_pose[2].position.y, 0, NULL },
		{(char *) "rear_bullbar", (char*) "z",     		ASTRO_PARAM_DOUBLE, &board_pose[2].position.z, 0, NULL },
		{(char *) "rear_bullbar", (char*) "roll",  		ASTRO_PARAM_DOUBLE, &board_pose[2].orientation.roll, 0, NULL},
		{(char *) "rear_bullbar", (char*) "pitch", 		ASTRO_PARAM_DOUBLE, &board_pose[2].orientation.pitch, 0, NULL},
		{(char *) "rear_bullbar", (char*) "yaw",   		ASTRO_PARAM_DOUBLE, &board_pose[2].orientation.yaw, 0, NULL},
		// {(char *) "semi_trailer1", (char*) "M",   		ASTRO_PARAM_DOUBLE, &semi_trailer1_M, 0, NULL},
   	};
	astro_param_install_params(argc, argv, optional_commandline_param_list, sizeof(optional_commandline_param_list) / sizeof(optional_commandline_param_list[0]));

   	for (int i = 0; i < argc; i++) // De acordo com o que pensei, i deveria iniciar igual a 3. Mas se o valor não for =0, o primeiro elemento do lidars_alive não ativa. Não entendi a razão.
	{
		if (strcmp(argv[i], "-lidar") == 0 && i < argc - 1 && argv[i + 1][0] != '-')
		{
			if (atoi(argv[i + 1]) < MAX_NUMBER_OF_LIDARS_NEURAL)
				lidars_alive[atoi(argv[i + 1])] = true;
		}

	   if (strcmp(argv[i], "-ouster64") == 0)
	   {
			lidars_alive[0] = true;
			lidars_alive[1] = true;
			lidars_alive[2] = true;
			lidars_alive[3] = true;
	   }
	}

   int at_least_one_lidar_alive = 0;
   astro_lidar_config *p;
   for(int i = 0; i < MAX_NUMBER_OF_LIDARS_NEURAL; i++)
   {
	   if (lidars_alive[i])  // Lidars start from 10 in the sensors_params vector
	   {
		   at_least_one_lidar_alive = 1;
		   p = &lidar_config[i];
		   load_lidar_config(argc, argv, i, &p);
			if (lidar_config[i].pose.orientation.yaw < 0.0)
			{
				lidar_config[i].pose.orientation.yaw += (2.0 * M_PI);
			}
	   }
   }

   if (at_least_one_lidar_alive == 0)
	   astro_die("Nenhum lidar classificado como 'alive', verifique se seus argumentos estão corretos!\nExemplos de argumentos:\n ./neural_object_detector_tracker 3 1 -lidar 1 -lidar 3 -lidar 16\n ./neural_object_detector_tracker intelbras1 1 -lidar 0 -lidar 1 -lidar 2 -lidar 3\n ./neural_object_detector_tracker intelbras1 1 -velodyne\n ./neural_object_detector_tracker intelbras1 1 -ouster64\n");
	if (semi_trailer_config.num_semi_trailers > 0)
		astro_task_manager_read_semi_trailer_parameters(&semi_trailer_config, argc, argv, semi_trailer_config.num_semi_trailers);
}


void
subscribe_messages()
{
	if (lidars_alive[0])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler0, ASTRO_SUBSCRIBE_LATEST, 0);

	if (lidars_alive[1])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler1, ASTRO_SUBSCRIBE_LATEST, 1);

	if (lidars_alive[2])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler2, ASTRO_SUBSCRIBE_LATEST, 2);

	if (lidars_alive[3])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler3, ASTRO_SUBSCRIBE_LATEST, 3);

	if (lidars_alive[4])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler4, ASTRO_SUBSCRIBE_LATEST, 4);

	if (lidars_alive[5])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler5, ASTRO_SUBSCRIBE_LATEST, 5);

	if (lidars_alive[6])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler6, ASTRO_SUBSCRIBE_LATEST, 6);

	if (lidars_alive[7])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler7, ASTRO_SUBSCRIBE_LATEST, 7);

	if (lidars_alive[8])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler8, ASTRO_SUBSCRIBE_LATEST, 8);

	if (lidars_alive[9])
		astro_velodyne_subscribe_variable_scan_message(NULL, (astro_handler_t) variable_scan_message_handler9, ASTRO_SUBSCRIBE_LATEST, 9);

    astro_localize_ackerman_subscribe_globalpos_message(NULL, (astro_handler_t) localize_ackerman_globalpos_message_handler, ASTRO_SUBSCRIBE_LATEST);
}


int
main(int argc, char **argv)
{
	setlocale(LC_ALL, "C");

	astro_ipc_initialize(argc, argv);

	read_parameters(argc, argv);

	signal(SIGINT, shutdown_module);

	subscribe_messages();

	astro_ipc_dispatch();

	return 0;
}
