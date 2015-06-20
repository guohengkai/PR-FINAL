/*************************************************************************
    > File Name: src/detect/sign_detector.cpp
    > Author: Guo Hengkai
    > Description: Base sign detector class implementation
    > Created Time: Wed 17 Jun 2015 10:17:12 AM CST
 ************************************************************************/
#include "sign_detector.h"

namespace ghk
{
bool SignDetector::DetectSingle(const Mat &image,
        vector<Rect> *rects, vector<int> *labels)
{
    vector<Mat> image_vec(1, image);
    vector<vector<int>> labels_vec;
    vector<vector<Rect>> rects_vec;
    if (!Detect(image_vec, &rects_vec, &labels_vec))
    {
        return false;
    }
    *rects = rects_vec[0];
    *labels = labels[0];
    return true;
}

struct DetectedRect
{
    float prob;
    size_t idx;
};

bool DetectedRectComp(const DetectedRect &lhs, const DetectedRect &rhs)
{
    return lhs.prob > rhs.prob;
}

void MergeRects(const vector<Rect> &rects, const vector<int> &labels,
        const vector<float> &probs, vector<size_t> *idx)
{
    if (idx == nullptr)
    {
        return;
    }

    vector<DetectedRect> results;
    for (size_t i = 0; i < rects.size(); ++i)
    {
        results.push_back(DetectedRect{probs[i], i});
    }
    sort(results.begin(), results.end(), DetectedRectComp);

    idx->clear();
    for (auto result: results)
    {
        bool flag = true;
        size_t j = result.idx;
        for (auto i: *idx)
        {
            if (labels[i] == labels[j] && static_cast<float>(
                        (rects[i] & rects[j]).area())
                     / ((rects[i] | rects[j]).area()) >= 0.5)
            {
                flag = false;
                break;
            }
        }

        if (flag)
        {
            idx->push_back(j);
        }
    }
}
}  // namespace ghk
