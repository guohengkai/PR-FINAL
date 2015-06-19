/*************************************************************************
    > File Name: src/detect/sign_detector.cpp
    > Author: Guo Hengkai
    > Description: Base sign detector class implementation
    > Created Time: Wed 17 Jun 2015 10:17:12 AM CST
 ************************************************************************/
#include "sign_detector.h"

namespace ghk
{
bool SignDetector::DetectSingle(const Mat &image,
        vector<Rect> *rects, vector<int> *labels)
{
    vector<Mat> image_vec(1, image);
    vector<vector<int>> labels_vec;
    vector<vector<Rect>> rects_vec;
    if (!Detect(image_vec, &rects_vec, &labels_vec))
    {
        return false;
    }
    *rects = rects_vec[0];
    *labels = labels[0];
    return true;
}
}  // namespace ghk
