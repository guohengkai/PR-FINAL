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
    HogExtractor(int num_orient = 8, int cell_size = 8);
    ~HogExtractor();

    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);
    virtual bool Train(const vector<Mat> &images,
                const vector<int> &labels) { return false; }
    virtual bool Extract(const vector<Mat> &images, Mat *feats);

    inline void set_num_orient(int num_orient) { num_orient_ = num_orient; }
    inline void set_cell_size(int cell_size) { cell_size_ = cell_size; }

private:
    VlHog *hog_;
    int num_orient_;
    int cell_size_;
};
}  // namespace ghk

#endif  // FINAL_HOG_EXTRACTOR_H_

