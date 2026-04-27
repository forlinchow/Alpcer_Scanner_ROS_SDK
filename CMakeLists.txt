cmake_minimum_required(VERSION 3.16)
project(livox_color)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-lstdc++fs)

# 必须先找到 ament_cmake
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_library(LIVOX_SDK2_LIB livox_lidar_sdk_static REQUIRED)
find_library(MPP_LIB rockchip_mpp REQUIRED)
find_library(RGA_LIB rga REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(livox_color_node src/main.cpp)

# 添加依赖链接
ament_target_dependencies(livox_color_node rclcpp sensor_msgs Threads OpenCV OpenMP)

# 设置安装路径
install(TARGETS livox_color_node
        DESTINATION lib/${PROJECT_NAME})

target_link_libraries(livox_color_node
        ${LIVOX_SDK2_LIB} ${MPP_LIB} ${RGA_LIB} stdc++fs)

ament_package()
#colcon build --packages-select livox_color