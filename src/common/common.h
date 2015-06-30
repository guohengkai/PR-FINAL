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
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <vector>

namespace ghk
{
using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Size;

using std::size_t;
using std::map;
using std::max;
using std::min;
using std::set;
using std::sort;
using std::string;
using std::stringstream;
using std::vector;
using std::cout;
using std::endl;
using std::flush;
}  // namespace ghk
#endif  // FINAL_COMMON_H_
