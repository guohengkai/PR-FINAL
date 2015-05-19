/*************************************************************************
    > File Name: knn_classifier.cpp
    > Author: Guo Hengkai
    > Description: KNN classifier class implementation
    > Created Time: Tue 19 May 2015 03:03:32 PM CST
 ************************************************************************/
#include "knn_classifier.h"
#include "mat_util.h"
#include "file_util.h"

namespace ghk
{
bool KnnClassifier::Save(const string &model_name) const
{
    if (!SaveMat(model_name + "_normA", normA_))
    {
        return false;
    }
    if (!SaveMat(model_name + "_normB", normB_))
    {
        return false;
    }
    if (!SaveMat(model_name, model_, vector<float>(1, near_num_)))
    {
        return false;
    }
    return true;
}

bool KnnClassifier::Load(const string &model_name)
{
    if (!LoadMat(model_name + "_normA", &normA_))
    {
        return false;
    }
    if (!LoadMat(model_name + "_normB", &normB_))
    {
        return false;
    }
    vector<float> near_num;
    if (!LoadMat(model_name, &model_, &near_num))
    {
        return false;
    }
    near_num_ = static_cast<int>(near_num[0]);

    int n = model_.cols;
    knn_.train(model_.colRange(0, n - 1), model_.col(n - 1),
            Mat(), false, near_num_, false);
    return true;
}

bool KnnClassifier::Train(const Mat &feats, const vector<int> &labels)
{
    if (feats.rows != static_cast<int>(labels.size()))
    {
        printf("KNN: dimension mismatch in training.\n");
        return false;
    }

    TrainNormalize(feats, &normA_, &normB_);
    Mat feats_norm;
    Normalize(feats, &feats_norm);

    Mat labels_mat;
    Vec2Mat(labels, &labels_mat);

    knn_.train(feats_norm, labels_mat, Mat(), false, near_num_, false);

    int m = feats.rows;
    int n = feats.cols + 1;
    model_ = Mat(m, n, CV_32F);
    feats_norm.copyTo(model_.colRange(0, n - 1));
    labels_mat.copyTo(model_.col(n - 1));
    return true;
}

bool KnnClassifier::Predict(const Mat &feats, vector<int> *labels) const
{
    if (labels == nullptr)
    {
        return false;
    }

    Mat feats_norm;
    Normalize(feats, &feats_norm);

    labels->clear();
    for (int i = 0; i < feats.rows; ++i)
    {
        int result = static_cast<int>(knn_.find_nearest(
                    feats_norm.row(i), near_num_));
        labels->push_back(result);
    }
    return true;
}

void KnnClassifier::Normalize(const Mat &feats, Mat *feats_norm) const
{
    ghk::Normalize(normA_, normB_, feats, feats_norm);
}
}  // namespace ghk