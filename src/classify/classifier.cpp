/*************************************************************************
    > File Name: classifier.cpp
    > Author: Guo Hengkai
    > Description: Base classifier class implementation
    > Created Time: Tue 19 May 2015 01:11:48 PM CST
 ************************************************************************/
#include "classifier.h"

namespace ghk
{
int Classifier::PredictSample(const Mat &feat) const
{
    vector<int> label;
    if (!Predict(feat, &label))
    {
        return -1;
    }
    return label[0];
}
}  // namespace ghk
