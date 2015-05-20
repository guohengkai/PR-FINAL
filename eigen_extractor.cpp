/*************************************************************************
    > File Name: eigen_extractor.cpp
    > Author: Guo Hengkai
    > Description: Feature extractor class implementation using PCA projection
    > Created Time: Wed 20 May 2015 09:04:27 AM CST
 ************************************************************************/
#include "eigen_extractor.h"
#include "file_util.h"
#include "mat_util.h"

namespace ghk
{
EigenExtractor::EigenExtractor(int feat_dim)
{
    set_feat_dim(feat_dim);
}

bool EigenExtractor::Save(const string &model_name) const
{
    if (!SaveMat(model_name + "_vec.txt", pca_.eigenvectors))
    {
        return false;
    }
    if (!SaveMat(model_name + "_val.txt", pca_.eigenvalues))
    {
        return false;
    }
    if (!SaveMat(model_name + "_mean.txt", pca_.mean))
    {
        return false;
    }
    return true;
}

bool EigenExtractor::Load(const string &model_name)
{
    if (!LoadMat(model_name + "_vec.txt", &pca_.eigenvectors))
    {
        return false;
    }
    if (!LoadMat(model_name + "_val.txt", &pca_.eigenvalues))
    {
        return false;
    }
    if (!LoadMat(model_name + "_mean.txt", &pca_.mean))
    {
        return false;
    }
    return true;
}

bool EigenExtractor::Train(const vector<Mat> &images,
                const vector<int> &labels)
{
    Mat image_vecs;
    if (!Image2Vec(images, &image_vecs))
    {
        printf("Error image set for PCA.\n");
        return false;
    }

    if (feat_dim() == 0)
    {
        pca_(image_vecs, Mat(), CV_PCA_DATA_AS_ROW);
        set_feat_dim(image_vecs.cols);
    }
    else
    {
        pca_(image_vecs, Mat(), CV_PCA_DATA_AS_ROW, feat_dim());
    }
    return true;
}

bool EigenExtractor::Extract(const vector<Mat> &images, Mat *feats) const
{
    if (feats == nullptr)
    {
        return false;
    }

    Mat image_vecs;
    if (!Image2Vec(images, &image_vecs))
    {
        printf("Error image set.\n");
        return false;
    }

    pca_.project(image_vecs, *feats);
    return true;
}
}  // namespace ghk
