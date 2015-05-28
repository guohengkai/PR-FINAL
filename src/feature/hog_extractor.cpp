/*************************************************************************
    > File Name: hog_extractor.cpp
    > Author: Guo Hengkai
    > Description: HOG feature extractor calss implementation
    > Created Time: Sun 24 May 2015 10:54:01 PM CST
 ************************************************************************/
#include "hog_extractor.h"
extern "C"
{
#include "hog.h"
}
#include "mat_util.h"

namespace ghk
{
HogExtractor::HogExtractor()
{
    hog_ = vl_hog_new(VlHogVariantDalalTriggs, 8, VL_FALSE);
}

HogExtractor::~HogExtractor()
{
    vl_hog_delete(hog_);
}

bool HogExtractor::Extract(const vector<Mat> &images, Mat *feats)
{
    if (feats == nullptr)
    {
        return false;
    }

    *feats = Mat();
    for (auto image: images)
    {
        // Convert Mat to float array
        float image_data[image.total()];
        int k = 0;
        for (int i = 0; i < image.rows; ++i)
            for (int j = 0; j < image.cols; ++j, ++k)
            {
                image_data[k] = image.at<int>(i, j);
            }

        // Extract HOG features
        vl_hog_put_image(hog_, image_data, image.rows, image.cols,
                image.channels(), 8);
        set_feat_dim(vl_hog_get_width(hog_) * vl_hog_get_height(hog_)
                * vl_hog_get_dimension(hog_));
        float *hog_arr = (float*)vl_malloc(feat_dim() * sizeof(float));
        vl_hog_extract(hog_, hog_arr);

        // Convert float array to Mat
        Mat feat_row(1, feat_dim(), CV_32F);
        for (int i = 0; i < feat_dim(); ++i)
        {
            feat_row.at<float>(0, i) = hog_arr[i];
        }
        vl_free(hog_arr);

        feats->push_back(feat_row);
    }
    return true;
}
}  // namespace ghk
