include Makefile.conf

# MODULE_NAME ="$(shell cat logo)"
MODULE_NAME = Lidar Sweep Viewer
MODULE_COMMENT = .

LINK ?= g++
CXXFLAGS += -std=c++17

# IFLAGS += -I/usr/local/astro_boost/include
# LFLAGS += -L/usr/local/astro_boost/lib

CXXFLAGS += -Wno-parentheses -Wno-deprecated-copy -Wno-implicit-fallthrough


IFLAGS += -I$(ASTRO_HOME)/sharedlib/libtf/src  -I/usr/include/bullet/ -I/usr/local/include/bullet/
PCL_INC = $(wildcard /usr/local/include/pcl-1.8)
VTK_INC = /usr/include/vtk-6.3
IFLAGS += -I/usr/include/eigen3 -I $(PCL_INC) -I $(VTK_INC) -I/usr/include/GLFW

# LFLAGS += -lglobal -lipc -lvelodyne_interface \
# 		-lparam_interface -llocalize_ackerman_interface 
LFLAGS +=`pkg-config --libs opencv` -ltf -lpthread -lBulletCollision -lBulletDynamics \
		-lBulletSoftBody -lLinearMath -lboost_thread-mt -lrt -lboost_signals -lboost_system -lpcl_io -lpcl_common -lpcl_io_ply -lpcl_visualization -lpcl_filters -lpcl_segmentation -ltask_manager_interface -lastro_util -lGL -lGLU -lglfw -lGLEW

SOURCES = voxel_representation.cpp show_point_cloud.cpp opengl_viewer.cpp #lidar_sweep_viewer_main.cpp

PUBLIC_BINARIES = voxel_representation show_point_cloud #lidar_sweep_viewer

TARGETS = voxel_representation show_point_cloud #lidar_sweep_viewer

VOXEL_REP_OBJS = voxel_representation.o opengl_viewer.o

voxel_representation: $(VOXEL_REP_OBJS)
	$(ECHO) "    ---- Linking $^ to $@ ("$(LINK)")"
	$(SILENT) $(LINK) $(CFLAGS) $(IFLAGS) $^ -o $@ $(LFLAGS)

# lidar_sweep_viewer: lidar_sweep_viewer_main.o 
show_point_cloud: show_point_cloud.o

include Makefile.rules
