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
    virtual bool Test(const Dataset &dataset) { return false; }
    virtual bool Predict(const vector<Mat> &images,
            vector<int> *labels) = 0;
    virtual bool FullTest(const Dataset &dataset,
            const string &dir) { return false; }
    int PredictSingle(const Mat &image);
};
}  // namespace ghk

#endif  // FINAL_SIGN_CLASSIFIER_H_
