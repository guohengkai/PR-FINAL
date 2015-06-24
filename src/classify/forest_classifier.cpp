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

    max_class_ = 0;
    for (auto label: labels)
    {
        max_class_ = max(max_class_, label);
    }
    Mat label_mat;
    Vec2Mat(labels, &label_mat);

    forest_.train(feats, CV_ROW_SAMPLE, label_mat,
            Mat(), Mat(), var_type, Mat(), param_);
    return true;
}

bool ForestClassifier::Predict(const Mat &feats, vector<int> *labels) const
{
    vector<float> p;
    return Predict(feats, labels, &p);
    /*
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
    return true;*/
}

bool ForestClassifier::Predict(const Mat &feats, vector<int> *labels,
        vector<float> *probs) const
{
    if (labels == nullptr || probs == nullptr)
    {
        return false;
    }

    labels->clear();
    probs->clear();
    int n = forest_.get_tree_count();
    for (int i = 0; i < feats.rows; ++i)
    {
        Mat count = Mat::zeros(1, max_class_ + 1, CV_32F);
        for (int j = 0; j < n; ++j)
        {
            auto tree = forest_.get_tree(j);
            int res = tree->predict(feats.row(i))->value;
            ++count.at<float>(0, res);
        }
        count /= n;
        
        double prob = 0;
        int idx[2];
        cv::minMaxIdx(count, nullptr, &prob, nullptr, idx);
        labels->push_back(idx[1]);
        probs->push_back(prob);
    }
    return true;
}
}  // namespace ghk
