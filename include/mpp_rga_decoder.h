//
// Created by Zachary on 2026/4/22.
//

#ifndef LIVOX_COLOR_MPP_RGA_DECODER_H
#define LIVOX_COLOR_MPP_RGA_DECODER_H

#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_buffer.h>
#include <rga/RgaApi.h>
#include <rga/im2d.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class MppRgaDecoder {
public:
    /**
     * @param target_width 最终输出的图像宽度 (例如 4000)
     * @param target_height 最终输出的图像高度 (例如 3000)
     */
    MppRgaDecoder(int target_width, int target_height)
            : width_(target_width), height_(target_height) {

        // 1. 初始化 MPP
        mpp_create(&mpp_ctx_, &mpp_api_);

        // --- 关键配置 A: 开启分包模式 (Split Mode) ---
        // 告诉 MPP 每一个 Packet 就是一个完整的图像，不要尝试去拼包
        RK_U32 need_split = 1;
        mpp_api_->control(mpp_ctx_, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);

        // --- 关键配置 B: 设置立即输出 ---
        // 禁用延迟参考帧逻辑，这对单帧 JPEG 非常重要
        RK_U32 immediate_out = 1;
        mpp_api_->control(mpp_ctx_, MPP_DEC_SET_IMMEDIATE_OUT, &immediate_out);

        // 此时再初始化 MJPEG
        mpp_init(mpp_ctx_, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG);

        // 关键配置 C: 增加 Parser 内部缓存限制 (防止因大图报-1012)
        RK_U32 max_size = 10 * 1024 * 1024; // 10MB
        mpp_api_->control(mpp_ctx_, MPP_DEC_SET_PARSER_SPLIT_MODE, &max_size);

        // 设置超时为阻塞模式 (50ms 超时，防止彻底卡死)
        MppPollType timeout = MPP_POLL_BLOCK;
        mpp_api_->control(mpp_ctx_, MPP_SET_OUTPUT_TIMEOUT, &timeout);

        std::cout << "MPP Decoder initialized for Thread-Safe mode." << std::endl;
    }

    ~MppRgaDecoder() {
        if (mpp_ctx_) mpp_destroy(mpp_ctx_);
    }

    /**
     * 在赋色线程调用此函数
     * @param jpeg_data 已经拷贝出来的 JPEG 字节流
     * @param size 字节流大小
     * @param out_mat 预分配好的 BGR Mat
     */
    bool decode(const uint8_t* jpeg_data, size_t size, cv::Mat& out_mat) {
        if (!jpeg_data || size == 0) return false;

        MppPacket packet = nullptr;
        MppFrame frame = nullptr;

        // 1. 初始化 Packet (内部会自动处理内存分配和拷贝)
        mpp_packet_init(&packet, const_cast<uint8_t*>(jpeg_data), size);

        // 设置 EOS 标志，告诉 MPP 这一包就是一个完整的帧
        mpp_packet_set_eos(packet);

        // 2. 发送给硬解单元
        MPP_RET ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);
        printf("after decode_put_packet\n");
        if (ret != MPP_OK) {
            // 如果这里依然报 -1012，说明 MPP 内部 Parser 认为数据非法或太大
            // 尝试重置解码器
            mpp_api_->control(mpp_ctx_, MPP_DEC_SET_IMMEDIATE_OUT, nullptr);
            mpp_packet_deinit(&packet);
            printf("decode fail 1, %d\n", ret);
            printf("JPEG Header: %02x %02x, Size: %zu\n", jpeg_data[0], jpeg_data[1], size);
            return false;
        }

        // 3. 获取解码后的帧 (NV12)
        ret = mpp_api_->decode_get_frame(mpp_ctx_, &frame);
        printf("after decode_get_frame\n");

        if (ret == MPP_OK && frame) {
            // 4. 使用 RGA 硬件进行格式转换与缩放
            process_with_rga(frame, out_mat);
            printf("after process_with_rga\n");
            mpp_frame_deinit(&frame);
        } else {
            if (packet) mpp_packet_deinit(&packet);
            printf("decode fail 2, %d\n", ret);
            return false;
        }

        if (packet) mpp_packet_deinit(&packet);
        printf("decode success\n");
        return true;
    }

private:
    MppCtx mpp_ctx_ = nullptr;
    MppApi *mpp_api_ = nullptr;
    int width_, height_;

    void process_with_rga(MppFrame frame, cv::Mat& out_mat) {
        if (out_mat.empty()) {
            out_mat.create(height_, width_, CV_8UC3);
        }
        uint32_t fw = mpp_frame_get_width(frame);
        uint32_t fh = mpp_frame_get_height(frame);
        uint32_t fs_h = mpp_frame_get_hor_stride(frame);
        uint32_t fs_v = mpp_frame_get_ver_stride(frame);
        MppBuffer mpp_buf = mpp_frame_get_buffer(frame);
        int mpp_fd = mpp_buffer_get_fd(mpp_buf);

        // 输入：MPP硬解出的 NV12 (通过 fd 引用，零拷贝)
        rga_buffer_t src_img = wrapbuffer_fd(mpp_fd, (int)fw, (int)fh,
                                             RK_FORMAT_YCbCr_420_SP, (int)fs_h, (int)fs_v);

        // 输出：OpenCV 的 BGR 内存
        rga_buffer_t dst_img = wrapbuffer_virtualaddr(out_mat.data, out_mat.cols, out_mat.rows,
                                                      RK_FORMAT_BGR_888);

        // RGA 硬件执行转换 (NV12 -> BGR)
        // 如果输入是 8K，out_mat 是 4K，imcvtcolor 会自动处理缩放
        imcvtcolor(src_img, dst_img, src_img.format, dst_img.format);
    }
};

#endif //LIVOX_COLOR_MPP_RGA_DECODER_H
