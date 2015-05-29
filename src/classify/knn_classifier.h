/*************************************************************************
    > File Name: knn_classifier.h
    > Author: Guo Hengkai
    > Description: KNN classifier class definition
    > Created Time: Tue 19 May 2015 02:53:46 PM CST
 ************************************************************************/
#ifndef FINAL_KNN_CLASSIFIER_H_
#define FINAL_KNN_CLASSIFIER_H_

#include "classifier.h"
#include "common.h"

namespace ghk
{
class KnnClassifier: public Classifier
{
public:
    explicit KnnClassifier(int near_num): near_num_(near_num) {}

    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);

    virtual bool Train(const Mat &feats, const vector<int> &labels);
    bool Train(const Mat &feats, const vector<int> &labels, bool is_reset);
    virtual bool Predict(const Mat &feats, vector<int> *labels) const;
    bool Predict(const Mat &feats, vector<int> *labels,
            vector<float> *distances) const;
private:
    CvKNearest knn_;
    int near_num_;
    Mat model_;  // Points for training

    Mat normA_;
    Mat normB_;

    void Normalize(const Mat &feats, Mat *feats_norm) const;
};
}  // namespace ghk

#endif  // FINAL_KNN_CLASSIFIER_H_
