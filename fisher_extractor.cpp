/*************************************************************************
    > File Name: fisher_extractor.cpp
    > Author: Guo Hengkai
    > Description: Feature extractor class implementation using Fisher projection
    > Created Time: Wed 20 May 2015 10:10:45 AM CST
 ************************************************************************/
#include "fisher_extractor.h"
#include "file_util.h"
#include "mat_util.h"

namespace ghk
{
FisherExtractor::FisherExtractor()
{
}

bool FisherExtractor::Save(const string &model_name) const
{
    if (!SaveMat(model_name + "_vec.txt", eigen_vector_))
    {
        return false;
    }
    if (!SaveMat(model_name + "_mean.txt", mean_))
    {
        return false;
    }
    return true;
}

bool FisherExtractor::Load(const string &model_name)
{
    if (!LoadMat(model_name + "_vec.txt", &eigen_vector_))
    {
        return false;
    }
    if (!LoadMat(model_name + "_mean.txt", &mean_))
    {
        return false;
    }
    return true;
}

bool FisherExtractor::Train(const vector<Mat> &images,
        const vector<int> &labels)
{
    Mat image_vecs;
    if (!Image2Vec(images, &image_vecs))
    {
        printf("Error image set for Fisher.\n");
        return false;
    }

    int num_c = GetUniqueClassNum(labels);
    set_feat_dim(num_c - 1);

    // PCA project
    int n = image_vecs.rows;
    cv::PCA pca(image_vecs, Mat(), CV_PCA_DATA_AS_ROW, n - num_c);

    // Fisher project
    Mat label_mat;
    Vec2Mat(labels, &label_mat);
    cv::LDA lda(pca.project(image_vecs), label_mat, feat_dim());

    // Save the project matrix
    mean_ = pca.mean.reshape(1, 1);
    cv::gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0,
            eigen_vector_, cv::GEMM_1_T);
    return true;
}

bool FisherExtractor::Extract(const vector<Mat> &images, Mat *feats)
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
    
    *feats = cv::subspaceProject(eigen_vector_, mean_, image_vecs);
    return true;
}
}  // namespace ghk
