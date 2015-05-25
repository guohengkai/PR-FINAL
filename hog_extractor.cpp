/*************************************************************************
    > File Name: hog_extractor.cpp
    > Author: Guo Hengkai
    > Description: HOG feature extractor calss implementation
    > Created Time: Sun 24 May 2015 10:54:01 PM CST
 ************************************************************************/
#include "hog_extractor.h"
#include "mat_util.h"

namespace ghk
{
HogExtractor::HogExtractor()
{
    set_feat_dim(hog_.getDescriptorSize());
}

bool HogExtractor::Extract(const vector<Mat> &images, Mat *feats)
{
    if (feats == nullptr)
    {
        return false;
    }

    *feats = Mat(0, feat_dim(), CV_32F);
    for (auto image: images)
    {
        vector<float> feat_vec;
        Mat image_char;
        image.convertTo(image_char, CV_8U);
        hog_.compute(image, feat_vec, image.size(), Size(0, 0), vector<Point>(1, Point(0, 0)));
        Mat feat_row;
        Vec2Mat(feat_vec, &feat_row);
        feat_row = feat_row.t();
        feats->push_back(feat_row);
    }
    return true;
}
}  // namespace ghk
