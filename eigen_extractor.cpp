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
    vector<float> param(1, feat_dim());
    if (!SaveMat(model_name + "_vec.txt", eigen_vector_, param))
    {
        return false;
    }
    if (!SaveMat(model_name + "_mean.txt", mean_))
    {
        return false;
    }
    return true;
}

bool EigenExtractor::Load(const string &model_name)
{
    vector<float> param;
    if (!LoadMat(model_name + "_vec.txt", &eigen_vector_, &param))
    {
        return false;
    }
    set_feat_dim(param[0]);
    if (!LoadMat(model_name + "_mean.txt", &mean_))
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

    // PCA project
    cv::PCA pca(image_vecs, Mat(), CV_PCA_DATA_AS_ROW);

    // Save the full project matrix
    mean_ = pca.mean.reshape(1, 1);
    eigen_vector_ = pca.eigenvectors.t();
    if (feat_dim() == 0)
    {
        set_feat_dim(image_vecs.cols);
    }
    return true;
}

bool EigenExtractor::Extract(const vector<Mat> &images, Mat *feats)
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

    if (feat_dim() == 0 || feat_dim() > eigen_vector_.cols)
    {
        set_feat_dim(eigen_vector_.cols);
    }
    *feats = cv::subspaceProject(eigen_vector_.colRange(0, feat_dim()),
            mean_, image_vecs);
    return true;
}
}  // namespace ghk
