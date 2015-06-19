/*************************************************************************
    > File Name: src/detect/sign_detector.h
    > Author: Guo Hengkai
    > Description: Base sign detector class definition
    > Created Time: Wed 17 Jun 2015 10:07:52 AM CST
 ************************************************************************/
#ifndef FINAL_SIGN_DETECTOR_H_
#define FINAL_SIGN_DETECTOR_H_

#include "common.h"
#include "dataset.h"

namespace ghk
{
class SignDetector
{
public:
    virtual bool Save(const string &model_name) const { return false; }
    virtual bool Load(const string &model_name) { return false; }
    
    virtual bool Train(const Dataset &dataset) = 0;
    virtual bool Test(const Dataset &dataset) { return false; }
    virtual bool Detect(const vector<Mat> &images,
            vector<vector<Rect>> *rects, vector<int> *labels) = 0;
    virtual bool FullTest(const Dataset &dataset,
            const string &dir) { return false; }
    int Detect(const Mat &image, vector<Rect> *rects);
};
}  // namespace ghk

#endif  // FINAL_SIGN_DETECTOR_H_

