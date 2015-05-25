/*************************************************************************
    > File Name: hog_extractor.h
    > Author: Guo Hengkai
    > Description: HOG feature extractor calss definition
    > Created Time: Sun 24 May 2015 10:39:26 PM CST
 ************************************************************************/
#ifndef FINAL_HOG_EXTRACTOR_H_
#define FINAL_HOG_EXTRACTOR_H_

#include "common.h"
#include "extractor.h"
extern "C"
{
#include "hog.h"
}

namespace ghk
{
class HogExtractor: public Extractor
{
public:
    HogExtractor();
    ~HogExtractor();

    virtual bool Train(const vector<Mat> &images,
                const vector<int> &labels) { return false; }
    virtual bool Extract(const vector<Mat> &images, Mat *feats);

private:
    VlHog *hog_;
};
}  // namespace ghk

#endif  // FINAL_HOG_EXTRACTOR_H_

