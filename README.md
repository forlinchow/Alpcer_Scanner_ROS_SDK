# Alpcer_Scanner_ROS_SDK

Alpcer_Scanner_ROS_SDK is the package used to connect LiDAR products produced by Livox, applicable for ROS2 (foxy recommended).

  **Note :**

  Alpcer_Scanner_ROS_SDK is not recommended for mass production but limited to test scenarios. You should optimize the code based on the original source to meet your various needs.

## 1. Preparation

### 1.1 OS requirements

  * Ubuntu 20.04 for ROS2 Foxy;

  **Tips:**

  Colcon is a build tool used in ROS2.

  How to install colcon: [Colcon installation instructions](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html)

### 1.2 Install ROS2

For ROS2 Foxy installation, please refer to:
[ROS Foxy installation instructions](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

Desktop-Full installation is recommend.

## 2. Build & Run Alpcer_Scanner_ROS_SDK

### 2.1 Clone Alpcer_Scanner_ROS_SDK source code:

```shell
git clone https://github.com/forlinchow/Alpcer_Scanner_ROS_SDK.git ws_livox/src/livox_color
```

  **Note :**

  Be sure to clone the source code in a '[work_space]/src/' folder (as shown above), otherwise compilation errors will occur due to the compilation tool restriction.

### 2.2 Build & install the Livox-SDK2

  **Note :**

  Please follow the guidance of installation in the [Livox-SDK2/README.md](https://github.com/Livox-SDK/Livox-SDK2/blob/master/README.md)

### 2.3 Build the Alpcer_Scanner_ROS_SDK:

#### For ROS2 Foxy:
```shell
source /opt/ros/foxy/setup.sh
cd ws_livox
colcon build --packages-select livox_color
```

### 2.4 Run Alpcer_Scanner_ROS_SDK:

#### For ROS2:
```shell
cd ws_livox
source install/setup.sh
ros2 run livox_color livox_color_node
```