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
#include "file_util.h"

namespace ghk
{
HogExtractor::HogExtractor(int num_orient, int cell_size):
    num_orient_(num_orient), cell_size_(cell_size)
{
    hog_ = vl_hog_new(VlHogVariantDalalTriggs, num_orient_, VL_FALSE);
}

HogExtractor::~HogExtractor()
{
    if (hog_ != nullptr)
    {
        vl_hog_delete(hog_);
    }
}

bool HogExtractor::Save(const string &model_name) const
{
    vector<float> param;
    param.push_back(num_orient_);
    param.push_back(cell_size_);

    if (!SaveMat(model_name + "_para", Mat(), param))
    {
        printf("Fail to save the parameter.\n");
        return false;
    }
    return true;
}

bool HogExtractor::Load(const string &model_name)
{
    vector<float> param;
    Mat tmp;
    if (!LoadMat(model_name + "_para", &tmp, &param))
    {
        printf("Fail to load the parameter.\n");
        return false;
    }
    num_orient_ = param[0];
    cell_size_ = param[1];

    if (hog_ != nullptr)
    {
        vl_hog_delete(hog_);
    }
    hog_ = vl_hog_new(VlHogVariantDalalTriggs, num_orient_, VL_FALSE);
    return true;
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
                image.channels(), cell_size_);
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
