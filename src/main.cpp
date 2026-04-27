#include <iostream>
#include "livox_lidar_def.h"
#include "livox_lidar_api.h"
#include <eigen3/Eigen/Dense>
#include <unistd.h>
#include <experimental/filesystem>
#include <stdio.h>
#include <ctime>
#include <cstring>
#include "common_utils.h"
#include <iomanip>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>

// 引入 ROS2 及 PointCloud2 相关头文件
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include "camera.h"

#include "ply_fast_writer.h"

namespace fs = std::experimental::filesystem::v1;
using namespace cv;
using namespace std;

// 自定义带有 RGB 信息的点结构体，用于在线程间传递
struct ColoredPoint {
    float x, y, z;
    float reflectivity;
    uint8_t r, g, b;
};

volatile uint32_t mLidarHandle = std::numeric_limits<uint32_t>::max();
std::vector<ColoredPoint> frame_points;
volatile int framesScanned = 0;
const int FRAMES_SCAN_COUNT = 200;

// --- 全局队列与线程同步变量 ---
std::mutex g_cloud_mutex;
std::condition_variable g_cloud_cv;
std::queue<std::vector<ColoredPoint>> g_cloud_queue;
// 防爆核心：最多只缓存 2 帧点云。来不及处理就丢弃旧的，保证 RViz 永远看最新帧。
const size_t MAX_QUEUE_SIZE = 1;

CAMERA::CameraSingle* pCameraSingle = nullptr;
Eigen::Matrix4d transform_to_camera_matrix;

template<typename ... Args>
static std::string str_format(const std::string &format, Args ... args)
{
    auto size_buf = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    std::unique_ptr<char[]> buf(new(std::nothrow) char[size_buf]);

    if (!buf)
        return std::string("");

    std::snprintf(buf.get(), size_buf, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size_buf - 1);
}

std::string time_format(uint64_t ns) {
    auto t = (time_t)(ns/1000000000);
    std::tm *tm = std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// --- 回调函数：Livox 数据入口 (10Hz) ---
void PointCloudCallback(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket *data, void *client_data) {
    if (data == nullptr) return;
//    printf("point cloud handle: %u, data_num: %d, data_type: %d, length: %d, frame_counter: %d, time_type: %d\n",
//           handle, data->dot_num, data->data_type, data->length, data->frame_cnt, data->time_type);

    // 假设 Mid-360 使用的是 kExtendCartesian 模式
    if (data->data_type == kLivoxLidarCartesianCoordinateHighData) {
        LivoxLidarCartesianHighRawPoint *p_point_data = (LivoxLidarCartesianHighRawPoint *)data->data;
        uint32_t points_num = data->length / sizeof(LivoxLidarCartesianHighRawPoint);

        for (uint32_t i = 0; i < points_num; i++) {
            if (p_point_data[i].x == 0 && p_point_data[i].y == 0 && p_point_data[i].z == 0) {
                continue;
            }
            ColoredPoint cp;
            cp.x = p_point_data[i].x / 1000.0f; // 毫米转米
            cp.y = p_point_data[i].y / 1000.0f;
            cp.z = p_point_data[i].z / 1000.0f;
            cp.reflectivity = p_point_data[i].reflectivity;
            cp.r = 255; cp.g = 255; cp.b = 255; // 默认白色
            frame_points.push_back(cp);
        }
    }

    framesScanned++;

    // 将解析好的点云深拷贝放入队列
    if (framesScanned >= FRAMES_SCAN_COUNT) {
        {
            std::lock_guard<std::mutex> lock(g_cloud_mutex);

            if (g_cloud_queue.size() >= MAX_QUEUE_SIZE) {
                g_cloud_queue.pop();
            }

            g_cloud_queue.push(frame_points);
            g_cloud_cv.notify_one();
        }

        frame_points.clear();
        framesScanned = 0;
    }

}

void LivoxLidarPushMsgCallback(const uint32_t handle, const uint8_t dev_type, const char* info, void* client_data) {
    struct in_addr tmp_addr;
    tmp_addr.s_addr = handle;
    std::cout << "handle: " << handle << ", ip: " << inet_ntoa(tmp_addr) << ", push msg info: " << std::endl;
    std::cout << info << std::endl;
    return;
}

void WorkModeCallback(livox_status status, uint32_t handle,LivoxLidarAsyncControlResponse *response, void *client_data) {
    if (response == nullptr) {
        return;
    }
    printf("WorkModeCallack, status:%u, handle:%u, ret_code:%u, error_key:%u",
           status, handle, response->ret_code, response->error_key);

}

void QueryInternalInfoCallback(livox_status status, uint32_t handle,
                               LivoxLidarDiagInternalInfoResponse* response, void* client_data) {
    if (status != kLivoxLidarStatusSuccess) {
        printf("Query lidar internal info failed:%d.\n", status);
        QueryLivoxLidarInternalInfo(handle, QueryInternalInfoCallback, nullptr);
        return;
    }
    if (response == nullptr) {
        return;
    }
    uint8_t host_point_ipaddr[4] {0};
    uint16_t host_point_port = 0;
    uint16_t lidar_point_port = 0;

    uint8_t host_imu_ipaddr[4] {0};
    uint16_t host_imu_data_port = 0;
    uint16_t lidar_imu_data_port = 0;

    uint16_t off = 0;

    std::string versionApp;
    std::string versionLoader;
    std::string mac;
    std::string versionHardware;
    std::string curWorkState;
    std::string coreTemp;
    std::string powerUpCnt;
    std::string localTimeNow;
    std::string lastSyncTime;
    long timeOffset;
    long timeSyncType;
    std::string errorCode;
    long fwType;
    uint8_t detectMode;
    for (uint8_t i = 0; i < response->param_num; ++i) {
        LivoxLidarKeyValueParam* kv = (LivoxLidarKeyValueParam*)&response->data[off];
        if (kv->key == kKeyLidarPointDataHostIpCfg) {
            memcpy(host_point_ipaddr, &(kv->value[0]), sizeof(uint8_t) * 4);
            memcpy(&(host_point_port), &(kv->value[4]), sizeof(uint16_t));
            memcpy(&(lidar_point_port), &(kv->value[6]), sizeof(uint16_t));
        } else if (kv->key == kKeyLidarImuHostIpCfg) {
            memcpy(host_imu_ipaddr, &(kv->value[0]), sizeof(uint8_t) * 4);
            memcpy(&(host_imu_data_port), &(kv->value[4]), sizeof(uint16_t));
            memcpy(&(lidar_imu_data_port), &(kv->value[6]), sizeof(uint16_t));
        } else if (kv->key == kKeyVersionApp) {
            versionApp = str_format("%d.%d.%d.%d", kv->value[0], kv->value[1], kv->value[2], kv->value[3]);
        } else if (kv->key == kKeyVersionLoader) {
            versionLoader = str_format("%d.%d.%d.%d", kv->value[0], kv->value[1], kv->value[2], kv->value[3]);
        } else if (kv->key == kKeyMac) {
            mac = str_format("%02x:%02x:%02x:%02x:%02x:%02x", kv->value[0], kv->value[1], kv->value[2], kv->value[3], kv->value[4], kv->value[5]);
        } else if (kv->key == kKeyVersionHardware) {
            versionHardware = str_format("%d.%d.%d.%d", kv->value[0], kv->value[1], kv->value[2], kv->value[3]);
        } else if (kv->key == kKeyCurWorkState) {
            curWorkState = std::to_string(*((uint8_t *)kv->value));
        } else if (kv->key == kKeyCoreTemp) {
            coreTemp = std::to_string(*((int32_t *)kv->value));
        } else if (kv->key == kKeyPowerUpCnt) {
            powerUpCnt = std::to_string(*((uint32_t *)kv->value));
        } else if (kv->key == kKeyLocalTimeNow) {
            localTimeNow = time_format(*((uint64_t *)kv->value));
        } else if (kv->key == kKeyLastSyncTime) {
            lastSyncTime = time_format(*((uint64_t *)kv->value));
        } else if (kv->key == kKeyTimeOffset) {
            timeOffset = *((int64_t *)kv->value);
        } else if (kv->key == kKeyTimeSyncType) {
            timeSyncType = *((uint8_t *)kv->value);
        } else if (kv->key == kKeyLidarDiagStatus) {
            errorCode = std::to_string(*((uint16_t*)kv->value));
        } else if (kv->key == kKeyFwType) {
            fwType = *((uint8_t *)kv->value);
        } else if (kv->key == kKeyDetectMode) {
            detectMode = *((uint8_t *)kv->value);
        }
        off += sizeof(uint16_t) * 2;
        off += kv->length;
    }

    printf("Host point cloud ip addr:%u.%u.%u.%u, host point cloud port:%u, lidar point cloud port:%u.\n",
           host_point_ipaddr[0], host_point_ipaddr[1], host_point_ipaddr[2], host_point_ipaddr[3], host_point_port, lidar_point_port);

    printf("Host imu ip addr:%u.%u.%u.%u, host imu port:%u, lidar imu port:%u.\n",
           host_imu_ipaddr[0], host_imu_ipaddr[1], host_imu_ipaddr[2], host_imu_ipaddr[3], host_imu_data_port, lidar_imu_data_port);

}

void LidarInfoChangeCallback(const uint32_t handle, const LivoxLidarInfo* info, void* client_data) {
    if (info == nullptr) {
        printf("lidar info change callback failed, the info is nullptr.\n");
        return;
    }
    printf("LidarInfoChangeCallback Lidar handle: %u SN: %s\n", handle, info->sn);
    // set the work mode to kLivoxLidarNormal, namely start the lidar
    SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, WorkModeCallback, nullptr);

    QueryLivoxLidarInternalInfo(handle, QueryInternalInfoCallback, nullptr);
    mLidarHandle = handle;
}

// --- 线程 1：相机持续采集 (15Hz) ---
void CameraCaptureThread() {
    while (rclcpp::ok()) {
        // 调用你补充好的相机抓拍缓存函数
        long t_s = common_utils::currentTimeMilliseconds();
        pCameraSingle->shootAutoToCacheRealtime();
        printf("shoot, cost: %f s\n", (common_utils::currentTimeMilliseconds()-t_s)/1000.0f);
    }
}

// --- 线程 2：点云处理与赋色 (速率取决于此线程性能) ---
void ColorizationWorkerThread(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher) {
    cv::Mat bgr_image;
    long timestamp;
//    pfw::PlyFastWriter writer;
    size_t count = 0;

    while (rclcpp::ok()) {
        long t_s = common_utils::currentTimeMilliseconds();
        std::vector<ColoredPoint> points_to_process;

        // 1. 阻塞等待，直到拿到最新的点云帧
        {
            std::unique_lock<std::mutex> lock(g_cloud_mutex);
            g_cloud_cv.wait(lock, [] { return !g_cloud_queue.empty() || !rclcpp::ok(); });
            if (!rclcpp::ok()) break;

            points_to_process = std::move(g_cloud_queue.front());
            g_cloud_queue.pop();
        }

        // 2. 从相机缓存抓取最新图片并解码
        if (!pCameraSingle->decodeRealtimeImage(bgr_image, timestamp)) {
            continue;
        }

        // 3. 开始赋色
        std::vector<cv::Point3f> object_points;
        std::vector<Point2f> image_points;
        object_points.reserve(points_to_process.size());
        for (auto& pt : points_to_process) {
            Eigen::Vector4d p_cam = transform_to_camera_matrix * Eigen::Vector4d(-pt.z, pt.y, pt.x, 1);
            object_points.emplace_back(cv::Point3f(p_cam.x(), p_cam.y(), p_cam.z()));
        }
        if (pCameraSingle->projectPoints(object_points, image_points)) {
            for (size_t i=0; i<points_to_process.size(); i++) {
                auto& pi = image_points[i];
                if (object_points[i].z > 0 && pi.y >= 0 && pi.y < bgr_image.rows && pi.x >=0 && pi.x < bgr_image.cols) {
                    auto pix = bgr_image.at<Vec3b>(pi.y, pi.x).val;
                    auto& pt = points_to_process[i];
                    pt.b = pix[0];
                    pt.g = pix[1];
                    pt.r = pix[2];
                }
            }
        } else {
            printf("projectPoints fail\n");
        }

        // 4. 将处理后的点云转换为 ROS 2 消息
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp = rclcpp::Clock().now();
        msg.header.frame_id = "livox_frame";
        msg.height = 1;
        msg.width = points_to_process.size();

        // 配置 RViz 支持的字段：xyz, intensity, rgb
        sensor_msgs::PointCloud2Modifier modifier(msg);
        modifier.setPointCloud2Fields(5,
                                      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                                      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                      "intensity", 1, sensor_msgs::msg::PointField::FLOAT32,
                                      "rgb", 1, sensor_msgs::msg::PointField::FLOAT32 // rviz通常使用FLOAT32装载RGB
        );
        modifier.resize(points_to_process.size());

        sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_intensity(msg, "intensity");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_rgb(msg, "rgb");

        count += points_to_process.size();
//        writer.reserve(count);
        for (const auto& pt : points_to_process) {
//            writer.addPoint({(float)pt.x, (float)pt.y, (float)pt.z, (uint8_t)pt.reflectivity, pt.r, pt.g, pt.b});

            *iter_x = pt.x;
            *iter_y = pt.y;
            *iter_z = pt.z;
            *iter_intensity = pt.reflectivity;

            // ROS 中 rgb 打包为 uint32
            uint32_t rgb = (static_cast<uint32_t>(pt.r) << 16 |
                            static_cast<uint32_t>(pt.g) << 8 |
                            static_cast<uint32_t>(pt.b));
            memcpy(&iter_rgb[0], &rgb, sizeof(uint32_t));

            ++iter_x; ++iter_y; ++iter_z; ++iter_intensity; ++iter_rgb;
        }

        // 5. 发布给 RViz
        publisher->publish(msg);
        printf("published, cost: %f s\n", (common_utils::currentTimeMilliseconds()-t_s)/1000.0f);
//        if (count > 100000) {
//            cv::imwrite("/factory_tools/tmp/test.jpg", bgr_image);
//            writer.write("/factory_tools/tmp/test.ply");
//            printf("write end.");
//            // 优雅退出
//            rclcpp::shutdown();
//        }
    }
}

int main(int argc, const char *argv[]) {

    cv::Mat scannerMat, cam_left;
    FileStorage fs("/camera/config.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Open config file failed!");
    }
    fs["scannerMat"] >> scannerMat;
    fs["cam_left"] >> cam_left;
    fs.release();

    Eigen::Matrix4d scannerMatrix, cam_left_matrix;
    cv::cv2eigen(scannerMat, scannerMatrix);
    cv::cv2eigen(cam_left, cam_left_matrix);
    transform_to_camera_matrix = cam_left_matrix*scannerMatrix;

    std::cout << "transform_to_camera_matrix:" << std::endl << transform_to_camera_matrix << std::endl;
    frame_points.reserve(FRAMES_SCAN_COUNT * 96);

    // 初始化 ROS 2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("livox_color_node");

    // 创建发布者
    auto publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/colored_point_cloud", 10);

    std::string config_filepath = "/usr/local/mid360_config.json";

    pCameraSingle = new CAMERA::CameraSingle("/dev/video0", "/camera/camera_params_left.xml", 4000, 3000);

    // --- 启动双线程 ---
    // 线程1：相机以 ~15Hz 疯狂向缓存写入 JPEG
    std::thread t_camera(CameraCaptureThread);

    // 线程2：点云处理线程消费队列，并发布 ROS2 消息
    std::thread t_worker(ColorizationWorkerThread, publisher);

    // REQUIRED, to init Livox SDK2
    if (!LivoxLidarSdkInit(config_filepath.c_str())) {
        printf("Livox Init Failed\n");
        rclcpp::shutdown();
        if (t_camera.joinable()) t_camera.join();
        if (t_worker.joinable()) t_worker.join();
        delete pCameraSingle;
        LivoxLidarSdkUninit();
        return -1;
    }

    // REQUIRED, to get point cloud data via 'PointCloudCallback'
    SetLivoxLidarPointCloudCallBack(PointCloudCallback, nullptr);

    // OPTIONAL, to get imu data via 'ImuDataCallback'
    // some lidar types DO NOT contain an imu component
    //SetLivoxLidarImuDataCallback(ImuDataCallback, nullptr);

    SetLivoxLidarInfoCallback(LivoxLidarPushMsgCallback, nullptr);

    // REQUIRED, to get a handle to targeted lidar and set its work mode to NORMAL
    SetLivoxLidarInfoChangeCallback(LidarInfoChangeCallback, nullptr);

    // ROS 2 异步 Spin
    rclcpp::spin(node);

    // 优雅退出
    rclcpp::shutdown();
    if (t_camera.joinable()) t_camera.join();
    if (t_worker.joinable()) t_worker.join();

    delete pCameraSingle;

    {
        livox_status status = SetLivoxLidarWorkMode(mLidarHandle, kLivoxLidarWakeUp, WorkModeCallback, nullptr);
        if (status != kLivoxLidarStatusSuccess) {
            throw std::runtime_error("SetLivoxLidarWorkMode failed! Status = " + std::to_string(status));
        }
        status = SetLivoxLidarWorkMode(mLidarHandle, kLivoxLidarSleep, WorkModeCallback, nullptr);
        if (status != kLivoxLidarStatusSuccess) {
            throw std::runtime_error("SetLivoxLidarWorkMode failed! Status = " + std::to_string(status));
        }
    }

    LivoxLidarSdkUninit();

    printf("livox_color end!\n");
    return 0;
}
