/*************************************************************************
    > File Name: common.h
    > Author: Guo Hengkai
    > Description: common include files, namspace, variable
    > Created Time: Fri 15 May 2015 03:46:01 PM CST
 ************************************************************************/
#ifndef FINAL_COMMON_H_
#define FINAL_COMMON_H_

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace ghk
{
using cv::Mat;
using cv::Rect;
using cv::imread;

using std::size_t;
using std::map;
using std::string;
using std::vector;
}  // namespace ghk
#endif  // FINAL_COMMON_H_
