/*************************************************************************
    > File Name: sign_classifier.cpp
    > Author: Guo Hengkai
    > Description: Base sign classifier class implementation
    > Created Time: Sat 16 May 2015 04:20:03 PM CST
 ************************************************************************/
#include "sign_classifier.h"

namespace ghk
{
int SignClassifier::Predict(const Mat &image)
{
    vector<Mat> image_vec(1, image);
    vector<int> labels;
    if (!Predict(image_vec, &labels))
    {
        return -1;
    }
    return labels[0];
}
}  // namespace ghk
