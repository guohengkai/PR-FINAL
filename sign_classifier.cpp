/*************************************************************************
    > File Name: sign_classifier.cpp
    > Author: Guo Hengkai
    > Description: Base sign classifier class implementation
    > Created Time: Sat 16 May 2015 04:20:03 PM CST
 ************************************************************************/
#include "sign_classifier.h"

namespace ghk
{
bool SignClassifier::Predict(const vector<Mat> &images, vector<int> *labels) const
{
    if (labels == nullptr)
    {
        return false;
    }

    labels->clear();
    for (size_t i = 0; i < images.size(); ++i)
    {
        int res = Predict(images[i]);
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
