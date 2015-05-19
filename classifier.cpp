/*************************************************************************
    > File Name: classifier.cpp
    > Author: Guo Hengkai
    > Description: Base classifier class implementation
    > Created Time: Tue 19 May 2015 01:11:48 PM CST
 ************************************************************************/
#include "classifier.h"

namespace ghk
{
bool Classifier::Predict(const Mat &feats, vector<int> *labels) const
{
    if (labels == nullptr)
    {
        return false;
    }

    labels->clear();
    for (int i = 0; i < feats.rows; ++i)
    {
        int res = Predict(feats.row(i));
        if (res < 0)
        {
            labels->clear();
            return false;
        }
        labels->push_back(res);
    }
    return true;
}
}  // namespace ghk
