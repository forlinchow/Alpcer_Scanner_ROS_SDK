//
// Created by Zachary on 2026/4/22.
// 完全对齐 rkmppdec.c MJPEG 解码流程，修复引用计数错误
//

#ifndef LIVOX_COLOR_MPP_RGA_DECODER_H
#define LIVOX_COLOR_MPP_RGA_DECODER_H

#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_meta.h>
#include <rga/im2d.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

class MppRgaDecoder {
public:
    MppRgaDecoder(int target_width, int target_height)
            : width_(target_width), height_(target_height) {

        MPP_RET ret = mpp_create(&mpp_ctx_, &mpp_api_);
        if (ret != MPP_OK) {
            std::cerr << "Failed to create MPP context: " << ret << std::endl;
            return;
        }

        MppDecCfg cfg = nullptr;
        mpp_dec_cfg_init(&cfg);
        mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1);
        mpp_api_->control(mpp_ctx_, MPP_DEC_SET_CFG, cfg);
        mpp_dec_cfg_deinit(cfg);

        ret = mpp_init(mpp_ctx_, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG);
        if (ret != MPP_OK) {
            std::cerr << "Failed to init MPP for MJPEG: " << ret << std::endl;
            return;
        }

        mpp_buffer_group_get_internal(&mem_group_,
                                      MPP_BUFFER_TYPE_DRM |
                                      MPP_BUFFER_FLAGS_DMA32 |
                                      MPP_BUFFER_FLAGS_CACHABLE);

        allocate_buffers();

        initialized_ = true;
        std::cout << "MPP MJPEG Decoder initialized. Zero-Copy Ready." << std::endl;
    }

    ~MppRgaDecoder() {
        if (yuv_buf_) {
            mpp_buffer_put(yuv_buf_);
            yuv_buf_ = nullptr;
        }
        if (bgr_buf_) {
            mpp_buffer_put(bgr_buf_);
            bgr_buf_ = nullptr;
        }
        if (mem_group_) {
            mpp_buffer_group_put(mem_group_);
            mem_group_ = nullptr;
        }
        if (mpp_ctx_) {
            mpp_destroy(mpp_ctx_);
            mpp_ctx_ = nullptr;
        }
    }

    bool decode(const uint8_t* jpeg_data, size_t size, cv::Mat& out_mat) {
        if (!initialized_ || !jpeg_data || size == 0) return false;

        MppPacket packet = nullptr;
        MppBuffer pkt_buf = nullptr;
        MppFrame output_frame = nullptr;
        MPP_RET ret;

        // --- 1. 拷贝 JPEG 数据到 DMA 缓冲区 ---
        ret = mpp_buffer_get(mem_group_, &pkt_buf, size);
        if (ret != MPP_OK || !pkt_buf) {
            std::cerr << "Failed to get buffer for packet: " << ret << std::endl;
            return false;
        }

        void* buf_ptr = mpp_buffer_get_ptr(pkt_buf);
        if (!buf_ptr) {
            mpp_buffer_put(pkt_buf);
            return false;
        }
        memcpy(buf_ptr, jpeg_data, size);

        // --- 2. 用 DMA 缓冲区初始化 Packet ---
        ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
        mpp_buffer_put(pkt_buf);  // 释放本地引用，packet 已持有
        if (ret != MPP_OK) {
            std::cerr << "Failed to init packet with buffer: " << ret << std::endl;
            return false;
        }

        // --- 3. 创建输出 Frame，绑定 yuv_buf_，注入到 packet meta ---
        ret = mpp_frame_init(&output_frame);
        if (ret != MPP_OK) {
            std::cerr << "Failed to init output frame: " << ret << std::endl;
            mpp_packet_deinit(&packet);
            return false;
        }
        mpp_frame_set_buffer(output_frame, yuv_buf_);

        MppMeta meta = mpp_packet_get_meta(packet);
        if (!meta) {
            mpp_frame_deinit(&output_frame);
            mpp_packet_deinit(&packet);
            return false;
        }
        ret = mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, output_frame);
        if (ret != MPP_OK) {
            std::cerr << "Failed to set output frame in meta: " << ret << std::endl;
            mpp_frame_deinit(&output_frame);
            mpp_packet_deinit(&packet);
            return false;
        }

        // --- 4. 发送到解码器（成功则 output_frame 所有权转移）---
        ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);
        mpp_packet_deinit(&packet);

        if (ret != MPP_OK) {
            std::cerr << "decode_put_packet failed: " << ret << std::endl;
            mpp_frame_deinit(&output_frame);  // 失败才需要释放
            return false;
        }
        // 从这里开始，output_frame 不能再被用户代码释放，后续由 MPP 管理

        // --- 5. 获取解码结果 ---
        MppFrame result_frame = nullptr;
        ret = mpp_api_->decode_get_frame(mpp_ctx_, &result_frame);
        if (ret != MPP_OK || !result_frame) {
            std::cerr << "decode_get_frame failed: " << ret << std::endl;
            return false;
        }

        // --- 6. 检查错误帧 ---
        if (mpp_frame_get_errinfo(result_frame) != 0) {
            std::cerr << "MPP returned an error frame!" << std::endl;
            mpp_frame_deinit(&result_frame);
            return false;
        }

        uint32_t actual_width  = mpp_frame_get_width(result_frame);
        uint32_t actual_height = mpp_frame_get_height(result_frame);
        MppFrameFormat fmt     = mpp_frame_get_fmt(result_frame);

        // --- 7. RGA 转换 ---
        bool success = process_with_rga(result_frame, actual_width, actual_height, fmt);

        // --- 8. 只释放 result_frame ---
        mpp_frame_deinit(&result_frame);

        if (success) {
            out_mat = cv::Mat(height_, width_, CV_8UC3, bgr_ptr_);
        }
        return success;
    }

private:
    MppCtx mpp_ctx_ = nullptr;
    MppApi *mpp_api_ = nullptr;
    int width_, height_;
    bool initialized_ = false;

    MppBufferGroup mem_group_ = nullptr;
    MppBuffer yuv_buf_ = nullptr;
    MppBuffer bgr_buf_ = nullptr;
    int bgr_fd_ = -1;
    void* bgr_ptr_ = nullptr;

    void allocate_buffers() {
        uint32_t hor_stride = MPP_ALIGN(width_, 16);
        uint32_t ver_stride = MPP_ALIGN(height_, 16);

        size_t yuv_size = hor_stride * ver_stride * 4;
        mpp_buffer_get(mem_group_, &yuv_buf_, yuv_size);

        size_t bgr_size = width_ * height_ * 3;
        mpp_buffer_get(mem_group_, &bgr_buf_, bgr_size);
        bgr_fd_  = mpp_buffer_get_fd(bgr_buf_);
        bgr_ptr_ = mpp_buffer_get_ptr(bgr_buf_);
    }

    bool process_with_rga(MppFrame decoded_frame,
                          uint32_t fw, uint32_t fh,
                          MppFrameFormat fmt) {
        int src_fd        = mpp_buffer_get_fd(yuv_buf_);
        uint32_t hor_stride = mpp_frame_get_hor_stride(decoded_frame);
        uint32_t ver_stride = mpp_frame_get_ver_stride(decoded_frame);

        int rga_src_fmt;
        switch (fmt & MPP_FRAME_FMT_MASK) {
            case MPP_FMT_YUV420SP: rga_src_fmt = RK_FORMAT_YCbCr_420_SP; break;
            case MPP_FMT_YUV422SP: rga_src_fmt = RK_FORMAT_YCbCr_422_SP; break;
            case MPP_FMT_YUV420P:  rga_src_fmt = RK_FORMAT_YCbCr_420_P;  break;
            case MPP_FMT_YUV422P:  rga_src_fmt = RK_FORMAT_YCbCr_422_P;  break;
            default:
                std::cerr << "Unsupported MPP format: " << (fmt & MPP_FRAME_FMT_MASK) << std::endl;
                return false;
        }

        rga_buffer_t src_img = wrapbuffer_fd(src_fd, (int)fw, (int)fh,
                                             rga_src_fmt,
                                             (int)hor_stride, (int)ver_stride);
        rga_buffer_t dst_img = wrapbuffer_fd(bgr_fd_, width_, height_, RK_FORMAT_BGR_888);
        IM_STATUS rga_ret = imcvtcolor(src_img, dst_img, src_img.format, dst_img.format);
        if (rga_ret != IM_STATUS_SUCCESS) {
            std::cerr << "RGA conversion failed: " << rga_ret << std::endl;
            return false;
        }
        return true;
    }
};

#endif //LIVOX_COLOR_MPP_RGA_DECODER_H