/*************************************************************************
    > File Name: classifier.h
    > Author: Guo Hengkai
    > Description: Base classifier class definition
    > Created Time: Tue 19 May 2015 12:57:51 PM CST
 ************************************************************************/
#ifndef FINAL_CLASSIFIER_H_
#define FINAL_CLASSIFIER_H_

#include "common.h"

namespace ghk
{
class Classifier
{
public:
    virtual bool Save(const string &model_name) const { return false; }
    virtual bool Load(const string &model_name) { return false; }

    virtual bool Train(const Mat &feats, const vector<int> &labels) = 0;
    virtual bool Predict(const Mat &feats, vector<int> *labels) const = 0;
    int Predict(const Mat &feat) const;
};
}  // namespace ghk

#endif  // FINAL_CLASSIFIER_H_
