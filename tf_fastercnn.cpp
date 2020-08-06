/* Copyright 2019-2025 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <numeric>
#include <iostream>
#include <algorithm>
#include "tf_fastercnn.hpp"
#include "utils.hpp"
using namespace std;

TF_Fasterrcnn::TF_Fasterrcnn(const std::string bmodel, const int device_id) {
   /* create device handler */
  bm_dev_request(&bm_handle_, device_id);

  /* create inference runtime handler */
  p_bmrt_ = bmrt_create(bm_handle_);

  /* load bmodel by file */
  bool flag = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!flag) {
    std::cout << "ERROR: failed to load bmodel[" << bmodel << "] " << std::endl;
    exit(-1);
  }

  bmrt_get_network_names(p_bmrt_, &net_names_);
  std::cout << "> Load model " << net_names_[0] << " successfully" << std::endl;

  /* more info pelase refer to bm_net_info_t in bmdef.h */
  auto net_info = bmrt_get_network_info(p_bmrt_, net_names_[0]);
  std::cout << "input scale:" << net_info->input_scales[0] << std::endl;
  std::cout << "output scale:" << net_info->output_scales[0] << std::endl;
  std::cout << "input number:" << net_info->input_num << std::endl;
  std::cout << "output number:" << net_info->output_num << std::endl;

  /* TODO: get class number from net_info */

  /* get fp32/int8 type, the thresholds may be different */
  if (BM_INT8 == net_info->input_dtypes[0]) {
    std::cout <<  "only support fp32 model" << std::endl;
    exit(1);
  } 
  bmrt_print_network_info(net_info);

  /*
   * only one input shape supported in the pre-built model
   * you can get stage_num from net_info
   */
  auto &input_shape = net_info->stages[0].input_shapes[0];
  /* malloc input and output system memory for preprocess data */
  int count = bmrt_shape_count(&input_shape);
  std::cout << "input count:" << count << std::endl;

  batch_size_ = 1;
  net_h_ = input_shape.dims[0];
  net_w_ = input_shape.dims[1];
  num_channels_ = input_shape.dims[2];

  float input_scale = 1;
  if (int8_flag_) {
    input_scale *= net_info->input_scales[0];
  }

  input_num_ = net_info->input_num;
  input_tensors_ = new bm_tensor_t[input_num_];
  for (int idx = 0; idx < input_num_; idx++) {
    auto &input_tensor = input_tensors_[idx];
    bmrt_tensor(&input_tensor, p_bmrt_, net_info->input_dtypes[idx],
                      net_info->stages[0].input_shapes[idx]);
  }

  output_num_ = net_info->output_num;
  output_tensors_ = new bm_tensor_t[output_num_];
  for (int i = 0; i < output_num_; i++) {
    auto &output_shape = net_info->stages[0].output_shapes[i];
    count = bmrt_shape_count(&output_shape);
    float* out = new float[count];
    outputs_.push_back(out);
    auto &output_tensor = output_tensors_[i];
    bmrt_tensor(&output_tensor, p_bmrt_, net_info->output_dtypes[i],
                      net_info->stages[0].output_shapes[i]);
  }
  ts_ = nullptr;
}

TF_Fasterrcnn::TF_Fasterrcnn() {
  for (int i = 0; i < input_num_; ++i) {
    bm_free_device(bm_handle_, input_tensors_[i].device_mem);
  }
  if (input_tensors_) {
    delete []input_tensors_;
  }
  for (int i = 0; i < output_num_; ++i) {
    bm_free_device(bm_handle_, output_tensors_[i].device_mem);
  }
  if (output_tensors_) {
    delete []output_tensors_;
  }
  free(net_names_);
  bmrt_destroy(p_bmrt_);
}

void TF_Fasterrcnn::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

void TF_Fasterrcnn::preForward(std::vector<cv::Mat>& images) {
  vector<bm_image> processed_imgs;
  images_.clear();
  assert(images.size() == 1);
  for (size_t i = 0; i < images.size(); i++) {
    cv::Mat resize_img;
    cv::resize(images[i], resize_img, cv::Size(net_w_, net_h_));
    cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
    bm_memcpy_s2d(bm_handle_, input_tensors_[0].device_mem,
            reinterpret_cast<void *>(resize_img.ptr<uchar>()));
    images_.push_back(images[i]);
  }
}

void TF_Fasterrcnn::forward() {
  bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_names_[0], input_tensors_,
                  input_num_, output_tensors_, output_num_, true, false);
  if (ret == true) {
    bm_thread_sync(bm_handle_);
  }
  if (!ret) {
    std::cout << "ERROR : inference failed!!"<< std::endl;
    exit(1);
  }

  for (int i = 0; i < output_num_; i++) {
    bm_memcpy_d2s(bm_handle_, outputs_[i], output_tensors_[i].device_mem);
  }
}

std::vector<std::vector<st_DetectionResult> > TF_Fasterrcnn::postForward() {
  std::vector<std::vector<st_DetectionResult> > det_results;
  for (int i = 0; i < batch_size_; i++) {
    float* blobs = reinterpret_cast<float*>(outputs_[0]);
    std::vector<st_DetectionResult> det_result;
    auto &output_tensor = output_tensors_[i];
    size_t count = bmrt_tensor_bytesize(&output_tensor) / 4;
    for (size_t j = 0; j < count; j += 6) {
      st_DetectionResult res;
      if (blobs[j + 1] > score_threshold_) {
        res.class_id = blobs[j];
        res.score = blobs[j + 1];
        res.x1 = blobs[j + 2] * images_[i].cols / net_w_;
        res.y1 = blobs[j + 3] * images_[i].rows / net_h_;
        res.x2 = blobs[j + 4] * images_[i].cols / net_w_;
        res.y2 = blobs[j + 5] * images_[i].rows / net_h_;
        det_result.push_back(res);
      }
    }
    det_results.push_back(det_result);
  }
  return det_results;
}

int TF_Fasterrcnn::getBatchSize() {
  return batch_size_;
}
