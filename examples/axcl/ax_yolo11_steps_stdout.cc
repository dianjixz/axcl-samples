/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 *
 * Copyright (c) 2024, AXERA Semiconductor Co., Ltd. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */
/*
 * Note: For the YOLO11 series exported by the ultralytics project.
 * Author: QQC
 */
/*
Usage example:
ffmpeg -i input.mp4 -f rawvideo -s 640x640 -pix_fmt rgb24 - | ./ax_yolo11_steps_stdout -m yolo11.axmodel -s 640x640 -t RGB | ffmpeg -f rawvideo -pix_fmt rgb24 -s 640x640 -i - -c:v libx264 -f rtsp rtsp://yourserver/app/stream
ffmpeg -i /dev/video0 -r 15 -f rawvideo -s 640x640 -pix_fmt rgb24 - 2>/dev/null | ./dist/ax_yolo11_steps_stdout -m ~/dist/static/yolo11x.axmodel -s 640x640 | ffplay -f rawvideo -pixel_format rgb24 -video_size 640x640 - 2>/dev/null

Parameters:
-m: Path to the YOLO11 model file (.axmodel)
-s: Input image dimensions in format WIDTHxHEIGHT (e.g., 1920x1080)
-t: Input image type: RGB, I420, NV12 (default: RGB)
The program reads raw video frames from stdin, performs YOLO object detection,
draws bounding boxes on the frames, and outputs the processed frames to stdout.
*/


#include <cstdio>
#include <cstring>

#ifndef LOG_OUT_PROT
#define LOG_OUT_PROT stderr
#endif


#include <numeric>
#include <unordered_map>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "base/common.hpp"
#include "base/detection.hpp"
#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"
#include <axcl.h>
#include "ax_model_runner/ax_model_runner_axcl.hpp"


const int DEFAULT_IMG_H = 640;
const int DEFAULT_IMG_W = 640;
std::array<int, 2> input_size;

const char *CLASS_NAMES[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

int NUM_CLASS = 80;

const int DEFAULT_LOOP_COUNT = 1;
const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD  = 0.45f;

enum InputType {
    RGB_TYPE = 0,
    I420_TYPE,
    NV12_TYPE,
};

static const std::unordered_map<std::string, InputType> input_type_map = {
    {"RGB",  RGB_TYPE},
    {"I420", I420_TYPE},
    {"NV12", NV12_TYPE}
};

InputType INPUT_TYPE = RGB_TYPE;

// Helper function to convert BGR to NV12 manually
void bgr_to_nv12(const cv::Mat& bgr, std::vector<uint8_t>& nv12_data, int width, int height) {
    cv::Mat yuv_i420;
    cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);
    
    nv12_data.resize(width * height * 3 / 2);
    
    // Copy Y plane
    memcpy(nv12_data.data(), yuv_i420.data, width * height);
    
    // Convert UV from I420 to NV12 format
    uint8_t* u_plane = yuv_i420.data + width * height;
    uint8_t* v_plane = yuv_i420.data + width * height + width * height / 4;
    uint8_t* uv_plane = nv12_data.data() + width * height;
    
    for (int i = 0; i < width * height / 4; i++) {
        uv_plane[2 * i] = u_plane[i];     // U
        uv_plane[2 * i + 1] = v_plane[i]; // V
    }
}

// Helper function to convert NV12 to BGR manually
void nv12_to_bgr(const uint8_t* nv12_data, cv::Mat& bgr, int width, int height) {
    // Convert NV12 to I420 first
    std::vector<uint8_t> i420_data(width * height * 3 / 2);
    
    // Copy Y plane
    memcpy(i420_data.data(), nv12_data, width * height);
    
    // Convert UV from NV12 to I420 format
    const uint8_t* uv_plane = nv12_data + width * height;
    uint8_t* u_plane = i420_data.data() + width * height;
    uint8_t* v_plane = i420_data.data() + width * height + width * height / 4;
    
    for (int i = 0; i < width * height / 4; i++) {
        u_plane[i] = uv_plane[2 * i];     // U
        v_plane[i] = uv_plane[2 * i + 1]; // V
    }
    
    // Convert I420 to BGR
    cv::Mat yuv_i420(height * 3 / 2, width, CV_8UC1, i420_data.data());
    cv::cvtColor(yuv_i420, bgr, cv::COLOR_YUV2BGR_I420);
}

namespace ax {

void post_process(const ax_runner_tensor_t *output, const int nOutputSize, cv::Mat &mat, int input_w, int input_h,
                  const std::vector<float> &time_costs, int orig_w, int orig_h)
{
    std::vector<detection::Object> proposals;
    std::vector<detection::Object> objects;
    timer timer_postprocess;
    
    for (int i = 0; i < 3; ++i) {
        auto feat_ptr  = (float *)output[i].pVirAddr;
        int32_t stride = (1 << i) * 8;
        detection::generate_proposals_yolov8_native(stride, feat_ptr, PROB_THRESHOLD, proposals, input_w, input_h,
                                                    NUM_CLASS);
    }
    
    // 这里检测框坐标会自动映射到 mat.rows, mat.cols 尺寸
    detection::get_out_bbox(proposals, objects, NMS_THRESHOLD, input_h, input_w, orig_h, orig_w);
    
    fprintf(LOG_OUT_PROT, "post process cost time:%.2f ms \n", timer_postprocess.cost());
    fprintf(LOG_OUT_PROT, "--------------------------------------\n");
    
    auto total_time   = std::accumulate(time_costs.begin(), time_costs.end(), 0.f);
    auto min_max_time = std::minmax_element(time_costs.begin(), time_costs.end());
    
    fprintf(LOG_OUT_PROT, "Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
            (int)time_costs.size(), total_time / (float)time_costs.size(), *min_max_time.second, *min_max_time.first);
    fprintf(LOG_OUT_PROT, "--------------------------------------\n");
    fprintf(LOG_OUT_PROT, "detection num: %zu\n", objects.size());
    
    detection::draw_objects(mat, objects, CLASS_NAMES, NULL);
}

bool run_model(const std::string &model, const int &repeat, int input_h, int input_w)
{
    // 1. init engine
    ax_runner_axcl runner;
    int ret = runner.init(model.c_str());
    if (ret != 0) {
        fprintf(stderr, "init ax model runner failed.\n");
        return false;
    }
    std::vector<uint8_t> indata(input_h * input_w * 3, 0);
    
    // Get original image dimensions from command line arguments
    int orig_h = input_size[0];  
    int orig_w = input_size[1];  
    
    // Calculate expected input size for different formats
    size_t expected_size = 0;
    switch (INPUT_TYPE) {
        case RGB_TYPE:
            expected_size = orig_h * orig_w * 3;
            break;
        case I420_TYPE:
            expected_size = orig_h * orig_w * 3 / 2;
            break;
        case NV12_TYPE:
            expected_size = orig_h * orig_w * 3 / 2;
            break;
        default:
            expected_size = orig_h * orig_w * 3;
            break;
    }
    
    std::vector<uint8_t> input_buffer(expected_size);
    
    // Warm up the model once at startup
    cv::Mat dummy_mat = cv::Mat::zeros(orig_h, orig_w, CV_8UC3);
    common::get_input_data_letterbox(dummy_mat, indata, input_h, input_w);
    memcpy(runner.get_input(0).pVirAddr, indata.data(), indata.size());
    
    for (int i = 0; i < 5; ++i) {
        runner.inference();
    }
    
    fprintf(stderr, "Model warmed up.\n");
    
    while (std::cin.read(reinterpret_cast<char *>(input_buffer.data()), expected_size)) {
        cv::Mat inmat;
        
        // Handle different input types - 解析输入数据到原始尺寸
        switch (INPUT_TYPE) {
            case RGB_TYPE: {
                // Create OpenCV Mat from RGB24 data
                inmat = cv::Mat(orig_h, orig_w, CV_8UC3, input_buffer.data()).clone();
                // Convert RGB to BGR for OpenCV processing
                cv::cvtColor(inmat, inmat, cv::COLOR_RGB2BGR);
                break;
            }
            case I420_TYPE: {
                // Convert I420 to BGR
                cv::Mat yuv_mat(orig_h * 3 / 2, orig_w, CV_8UC1, input_buffer.data());
                cv::cvtColor(yuv_mat, inmat, cv::COLOR_YUV2BGR_I420);
                break;
            }
            case NV12_TYPE: {
                // Convert NV12 to BGR using custom function
                nv12_to_bgr(input_buffer.data(), inmat, orig_w, orig_h);
                break;
            }
            default: {
                // Default to RGB
                inmat = cv::Mat(orig_h, orig_w, CV_8UC3, input_buffer.data()).clone();
                cv::cvtColor(inmat, inmat, cv::COLOR_RGB2BGR);
                break;
            }
        }
        
        // 创建用于推理的 640x640 图像
        cv::Mat inference_mat = inmat.clone();
        
        // Prepare input data with letterbox for model input size (640x640)
        common::get_input_data_letterbox(inference_mat, indata, input_h, input_w);
        
        // 2. insert input
        memcpy(runner.get_input(0).pVirAddr, indata.data(), indata.size());
        
        fprintf(LOG_OUT_PROT, "Engine push input is done. \n");
        fprintf(LOG_OUT_PROT, "--------------------------------------\n");
        
        // 3. run model
        std::vector<float> time_costs(repeat, 0);
        for (int i = 0; i < repeat; ++i) {
            ret = runner.inference();
            if (ret != 0) {
                fprintf(LOG_OUT_PROT, "inference failed, ret = %d\n", ret);
                break;
            }
            time_costs[i] = runner.get_inference_time();
        }
        
        // 4. 在原始尺寸图像上进行后处理和绘制
        // 注意：这里检测框会自动缩放到原始图像尺寸
        post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(), inmat, input_w, input_h, time_costs, orig_w, orig_h);
        
        // 5. 输出处理后的原始尺寸图像
        switch (INPUT_TYPE) {
            case RGB_TYPE: {
                cv::Mat output_mat;
                cv::cvtColor(inmat, output_mat, cv::COLOR_BGR2RGB);
                std::cout.write(reinterpret_cast<char *>(output_mat.data), output_mat.total() * output_mat.elemSize());
                break;
            }
            case I420_TYPE: {
                cv::Mat yuv_output;
                cv::cvtColor(inmat, yuv_output, cv::COLOR_BGR2YUV_I420);
                std::cout.write(reinterpret_cast<char *>(yuv_output.data), yuv_output.total());
                break;
            }
            case NV12_TYPE: {
                std::vector<uint8_t> nv12_output;
                bgr_to_nv12(inmat, nv12_output, orig_w, orig_h);
                std::cout.write(reinterpret_cast<char *>(nv12_output.data()), nv12_output.size());
                break;
            }
            default: {
                cv::Mat output_mat;
                cv::cvtColor(inmat, output_mat, cv::COLOR_BGR2RGB);
                std::cout.write(reinterpret_cast<char *>(output_mat.data), output_mat.total() * output_mat.elemSize());
                break;
            }
        }
        
        std::cout.flush();
        fprintf(LOG_OUT_PROT, "--------------------------------------\n");
    }
    
    runner.release();
    return true;
}

}  // namespace ax

int main(int argc, char *argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("size", 's', "input image size in format WIDTHxHEIGHT", false,
                         std::to_string(DEFAULT_IMG_W) + "x" + std::to_string(DEFAULT_IMG_H));
    cmd.add<std::string>("type", 't', "input image type: RGB, I420, NV12", false, "RGB");
    
    cmd.parse_check(argc, argv);
    
    // 0. get app args
    auto model_file = cmd.get<std::string>("model");
    auto input_type_string = cmd.get<std::string>("type");
    
    auto it = input_type_map.find(input_type_string);
    if (it != input_type_map.end()) {
        INPUT_TYPE = it->second;
    } else {
        fprintf(stderr, "Unsupported input type: %s, using RGB as default\n", input_type_string.c_str());
        INPUT_TYPE = RGB_TYPE;
    }
    
    auto model_file_flag = utilities::file_exist(model_file);
    if (!model_file_flag) {
        fprintf(stderr, "Input model file (%s) does not exist, please check it.\n", model_file.c_str());
        return -1;
    }
    
    auto input_size_string = cmd.get<std::string>("size");
    
    // Parse size in format WIDTHxHEIGHT
    size_t x_pos = input_size_string.find('x');
    if (x_pos == std::string::npos) {
        fprintf(stderr, "Invalid size format. Expected WIDTHxHEIGHT (e.g., 1920x1080)\n");
        return -1;
    }
    
    try {
        int width  = std::stoi(input_size_string.substr(0, x_pos));
        int height = std::stoi(input_size_string.substr(x_pos + 1));
        
        if (width <= 0 || height <= 0) {
            fprintf(stderr, "Invalid dimensions: width=%d, height=%d\n", width, height);
            return -1;
        }
        
        input_size = {height, width};  // Note: OpenCV uses height, width order
    } catch (const std::exception &e) {
        fprintf(stderr, "Failed to parse size '%s': %s\n", input_size_string.c_str(), e.what());
        return -1;
    }
    
    // 1. print args
    fprintf(LOG_OUT_PROT, "--------------------------------------\n");
    fprintf(LOG_OUT_PROT, "model file : %s\n", model_file.c_str());
    fprintf(LOG_OUT_PROT, "img_h, img_w : %d %d\n", input_size[0], input_size[1]);
    fprintf(LOG_OUT_PROT, "input type : %s\n", input_type_string.c_str());
    fprintf(LOG_OUT_PROT, "--------------------------------------\n");
    
    // 2. init axcl
    {
        auto ret = axclInit(0);
        if (0 != ret) {
            fprintf(stderr, "Init AXCL failed{0x%8x}.\n", ret);
            return -1;
        }
        
        axclrtDeviceList lst;
        ret = axclrtGetDeviceList(&lst);
        if (0 != ret || 0 == lst.num) {
            fprintf(stderr, "Get AXCL device failed{0x%8x}, find total %d device.\n", ret, lst.num);
            axclFinalize();
            return -1;
        }
        
        ret = axclrtSetDevice(lst.devices[0]);
        if (0 != ret) {
            fprintf(stderr, "Set AXCL device failed{0x%8x}.\n", ret);
            axclFinalize();
            return -1;
        }
        
        ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
        if (0 != ret) {
            fprintf(stderr, "axclrtEngineInit %d\n", ret);
            axclFinalize();
            return ret;
        }
    }
    
    // 3. run engine model
    {
        if (!ax::run_model(model_file, DEFAULT_LOOP_COUNT, DEFAULT_IMG_H, DEFAULT_IMG_W)) {
            fprintf(stderr, "Failed to run model\n");
            axclFinalize();
            return -1;
        }
    }
    
    // 4. finalize
    axclFinalize();
    return 0;
}