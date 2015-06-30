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

int FindFather(vector<int> &father, int x)
{
    if (father[x] != x)
    {
        father[x] = FindFather(father, father[x]);
    }
    return father[x];
}

void MergeRects(vector<Rect> &rects, vector<int> &labels,
        vector<float> &probs, float hit_rate)
{
    // Create Union-Find Set for boxes
    vector<int> father;
    int n = static_cast<int>(rects.size());
    for (int i = 0; i < n; ++i)
    {
        father.push_back(i);
    }
    for (int i = 0; i < n - 1; ++i)
        for (int j = i + 1; j < n; ++j)
        {
            if (FindFather(father, i) != FindFather(father, j))
            {
                if (labels[i] == labels[j] &&
                        static_cast<float>((rects[i] & rects[j]).area())
                         / ((rects[i] | rects[j]).area()) >= hit_rate)
                {
                    father[FindFather(father, i)] = FindFather(father, j);
                }
            }
        }

    // Merge boxes in the same set
    vector<Rect> rrects;
    vector<int> rlabels;
    vector<float> rprobs;
    for (int i = 0; i < n; ++i)
    {
        if (father[i] == i)
        {
            Rect sum(0, 0, 0, 0);
            float score = -1000000.0f;
            int k = 0;
            int label = labels[i];
            for (int j = 0; j < n; ++j)
            {
                if (FindFather(father, j) == i)
                {
                    sum.x += rects[j].x;
                    sum.y += rects[j].y;
                    sum.width += rects[j].width;
                    sum.height += rects[j].height;
                    if (probs[j] > score)
                    {
                        score = probs[j];
                    }
                    ++k;
                }
            }

            if (k <= 1)
            {
                continue;
            }

            float ik = 1.0f / k;
            sum.x = static_cast<int>(sum.x * ik + 0.5);
            sum.y = static_cast<int>(sum.y * ik + 0.5);
            sum.width = static_cast<int>(sum.width * ik + 0.5);
            sum.height = static_cast<int>(sum.height * ik + 0.5);
            rrects.push_back(sum);
            rprobs.push_back(score);
            rlabels.push_back(label);
        }
    }

    // Deal with conflict boxes using non-maximum suppression
    rects.clear();
    labels.clear();
    probs.clear();
    n = static_cast<int>(rrects.size());
    for (int i = 0; i < n; ++i)
    {
        bool flag = true;
        for (int j = 0; j < n; ++j)
        {
            if (j != i && rlabels[j] == rlabels[i] && rprobs[j] > rprobs[i])
            {
                if ((rrects[i] & rrects[j]).area() * 2
                        > min(rrects[i].area(), rrects[j].area()))
                {
                    flag = false;
                    break;
                }
            }
        }
        if (flag)
        {
            rects.push_back(rrects[i]);
            labels.push_back(rlabels[i]);
            probs.push_back(rprobs[i]);
        }
    }
}
}  // namespace ghk
