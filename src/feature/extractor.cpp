/*************************************************************************
    > File Name: extractor.cpp
    > Author: Guo Hengkai
    > Description: Base feature extractor class implementation
    > Created Time: Wed 20 May 2015 08:34:30 AM CST
 ************************************************************************/
#include "extractor.h"

namespace ghk
{
bool Extractor::ExtractFeat(const Mat &image, Mat *feat)
{
    vector<Mat> image_vec(1, image);
    return Extract(image_vec, feat);
}
}  // namespace ghk
