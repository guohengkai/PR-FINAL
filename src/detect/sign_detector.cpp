/*************************************************************************
    > File Name: src/detect/sign_detector.cpp
    > Author: Guo Hengkai
    > Description: Base sign detector class implementation
    > Created Time: Wed 17 Jun 2015 10:17:12 AM CST
 ************************************************************************/
#include "sign_detector.h"

namespace ghk
{
int SignDetector::Detect(const Mat &image, vector<Rect> *rects)
{
    vector<Mat> image_vec(1, image);
    vector<int> labels;
    vector<vector<Rect>> rects_vec;
    if (!Detect(image_vec, &rects_vec, &labels))
    {
        return -1;
    }
    *rects = rects_vec[0];
    return labels[0];
}
}  // namespace ghk
