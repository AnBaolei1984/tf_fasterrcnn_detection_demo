/* Copyright 2019-2024 by Bitmain Technologies Inc. All rights reserved.

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
#ifndef _TF_FASTERRCNN_HPP__
#define _TF_FASTERRCNN_HPP__

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "bmruntime_interface.h"
#include "utils.hpp"

#define USE_OPENCV

typedef struct __tag_st_DetectionResult{
  unsigned int class_id;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
}st_DetectionResult;

class TF_Fasterrcnn {
public:
  TF_Fasterrcnn(const std::string bmodel, const int device_id);
  TF_Fasterrcnn();
  void preForward(std::vector<cv::Mat>& images);
  void forward();
  std::vector<std::vector<st_DetectionResult> > postForward();
  void enableProfile(TimeStamp *ts);
  int getBatchSize();

private:
  /* handle of low level device */
  bm_handle_t bm_handle_;

  /* runtime helper */
  const char **net_names_;
  void *p_bmrt_;

  /* network input shape */
  int batch_size_;
  int num_channels_;

  int net_h_;
  int net_w_;

  float input_scale;
  float output_scale;
  std::vector<void*> outputs_;
  bool int8_flag_;
  int output_num_;
  int input_num_;
  /* for profiling */
  TimeStamp *ts_;

  bm_shape_t input_shape_;
  std::vector<cv::Mat> images_;
  const float score_threshold_ = 0.3f;
  bm_tensor_t* input_tensors_;
  bm_tensor_t* output_tensors_;
};

#endif /* YOLOV3_HPP */
