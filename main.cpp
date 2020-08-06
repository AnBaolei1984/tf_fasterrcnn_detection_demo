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
#include <boost/filesystem.hpp>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include "tf_fastercnn.hpp"
#include "utils.hpp"

namespace fs = boost::filesystem;
using namespace std;
using time_stamp_t = time_point<steady_clock, microseconds>;

static void detect(TF_Fasterrcnn &net, vector<cv::Mat>& images,
                                      vector<string> names, TimeStamp *ts) {
  ts->save("detection overall");
  ts->save("stage 1: pre-process");
  net.preForward(images);
  ts->save("stage 1: pre-process");
  ts->save("stage 2: detection  ");
  net.forward();
  ts->save("stage 2: detection  ");
  ts->save("stage 3:post-process");
  vector<vector<st_DetectionResult>> dets = net.postForward();
  ts->save("stage 3:post-process");
  ts->save("detection overall");

  string save_folder = "result_imgs";
  if (!fs::exists(save_folder)) {
    fs::create_directory(save_folder);
  }

  for (size_t i = 0; i < images.size(); i++) {
    for (size_t j = 0; j < dets[i].size(); j++) {
      int x_min = static_cast<int>(dets[i][j].x1);
      int x_max = static_cast<int>(dets[i][j].x2);
      int y_min = static_cast<int>(dets[i][j].y1);
      int y_max = static_cast<int>(dets[i][j].y2);

      std::cout << "class_id: " << dets[i][j].class_id
        << " Score: " << dets[i][j].score << " : " << x_min <<
        "," << y_min << "," << x_max << "," << y_max << std::endl;

      cv::Rect rc;
      rc.x = x_min;
      rc.y = y_min;;
      rc.width = x_max - x_min;
      rc.height = y_max - y_min;
      cv::rectangle(images[i], rc, cv::Scalar(255, 0, 0), 2, 1, 0);
    }
    cv::imwrite(save_folder + "/" + names[i], images[i]);
  }
}


int main(int argc, char **argv) {
  cout.setf(ios::fixed);

  if (argc < 4) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <image list> <bmodel file> <device_id> " << endl;
    exit(1);
  }

  string image_list = argv[1];
  if (!fs::exists(image_list)) {
    cout << "Cannot find input image file." << endl;
    exit(1);
  }

  string bmodel_file = argv[2];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  int device_id = atoi(argv[3]);

  TF_Fasterrcnn net(bmodel_file, device_id);
  int batch_size = net.getBatchSize();
  TimeStamp ts;
  net.enableProfile(&ts);

  char image_path[1024] = {0};
  vector<cv::Mat> batch_imgs;
  vector<string> batch_names;
  ifstream fp_img_list(image_list);
  while(fp_img_list.getline(image_path, 1024)) {
    cout << "process " << image_path << endl;
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR, device_id);
    if (img.empty()) {
      cout << "read image error!" << endl;
      exit(1);
    }
    fs::path fs_path(image_path);
    string img_name = fs_path.filename().string();
    batch_imgs.push_back(img);
    batch_names.push_back(img_name);
    if (static_cast<int>(batch_imgs.size()) == batch_size) {
      detect(net, batch_imgs, batch_names, &ts);
      batch_imgs.clear();
      batch_names.clear();
    }
  } 
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("tf detect");
  ts.show_summary("detect ");
  ts.clear();

  std::cout << std::endl;

  return 0;
}
