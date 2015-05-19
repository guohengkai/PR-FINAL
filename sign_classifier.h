/*************************************************************************
    > File Name: sign_classifier.h
    > Author: Guo Hengkai
    > Description: Base sign classifier class definition
    > Created Time: Sat 16 May 2015 04:08:35 PM CST
 ************************************************************************/
#ifndef FINAL_SIGN_CLASSIFIER_H_
#define FINAL_SIGN_CLASSIFIER_H_

#include "common.h"
#include "dataset.h"

namespace ghk
{
class SignClassifier
{
public:
    virtual bool Save(const string &model_name) const { return false; }
    virtual bool Load(const string &model_name) { return false; }

    virtual bool Train(const Dataset &dataset) = 0;
    virtual bool Test(const Dataset &dataset) const { return false; }
    virtual int Predict(const Mat &image) const = 0;
    bool Predict(const vector<Mat> &images, vector<int> *labels) const;
};
}  // namespace ghk

#endif  // FINAL_SIGN_CLASSIFIER_H_
