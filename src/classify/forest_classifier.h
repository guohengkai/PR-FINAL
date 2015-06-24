/*************************************************************************
    > File Name: src/classify/forest_classifier.h
    > Author: Guo Hengkai
    > Description: Random forest class defintion using CvRTrees
    > Created Time: Wed 24 Jun 2015 05:58:34 PM CST
 ************************************************************************/
#ifndef FINAL_FOREST_CLASSIFIER_H_
#define FINAL_FOREST_CLASSIFIER_H_

#include "common.h"
#include "classifier.h"

namespace ghk
{
class ForestClassifier: public Classifier
{
public:
    ForestClassifier(int max_depth, int min_sample_count)
    {
        param_.max_depth = max_depth;
        param_.min_sample_count = min_sample_count;
    }
    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);

    virtual bool Train(const Mat &feats, const vector<int> &labels);
    virtual bool Predict(const Mat &feats, vector<int> *labels) const;
    virtual bool Predict(const Mat &feats, vector<int> *labels,
            vector<float> *probs) const;
private:
    CvRTParams param_;
    CvRTrees forest_;
    int max_class_;
};
}  // namespace ghk

#endif  // FINAL_FOREST_CLASSIFIER_H_

