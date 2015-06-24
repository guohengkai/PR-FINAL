/*************************************************************************
    > File Name: src/classify/forest_classifier.cpp
    > Author: Guo Hengkai
    > Description: Random forest class implementation using CvRTrees
    > Created Time: Wed 24 Jun 2015 06:43:40 PM CST
 ************************************************************************/
#include "forest_classifier.h"
#include "mat_util.h"

namespace ghk
{
bool ForestClassifier::Save(const string &model_name) const
{
    forest_.save((model_name + ".yml").c_str());
    return true;
}

bool ForestClassifier::Load(const string &model_name)
{
    forest_.load((model_name + ".yml").c_str());
    return true;
}

bool ForestClassifier::Train(const Mat &feats, const vector<int> &labels)
{
    Mat var_type = Mat::ones(feats.cols + 1, 1, CV_8U) * CV_VAR_NUMERICAL;
    var_type.at<uchar>(feats.cols, 0) = CV_VAR_CATEGORICAL;

    Mat label_mat;
    Vec2Mat(labels, &label_mat);

    forest_.train(feats, CV_ROW_SAMPLE, label_mat,
            Mat(), Mat(), var_type, Mat(), param_);
    return true;
}

bool ForestClassifier::Predict(const Mat &feats, vector<int> *labels) const
{
    if (labels == nullptr)
    {
        return false;
    }

    labels->clear();
    for (int i = 0; i < feats.rows; ++i)
    {
        int label = forest_.predict(feats.row(i));
        labels->push_back(label);
    }
    return true;
}

bool ForestClassifier::Predict(const Mat &feats, vector<int> *labels,
        vector<float> *probs) const
{
    return false;
}
}  // namespace ghk
