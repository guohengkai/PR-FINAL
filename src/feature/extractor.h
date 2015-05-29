/*************************************************************************
    > File Name: extractor.h
    > Author: Guo Hengkai
    > Description: Base feature extractor class definition
    > Created Time: Wed 20 May 2015 08:30:04 AM CST
 ************************************************************************/
#ifndef FINAL_EXTRACTOR_H_
#define FINAL_EXTRACTOR_H_

#include "common.h"

namespace ghk
{
class Extractor
{
public:
    virtual bool Save(const string &model_name) const { return false; }
    virtual bool Load(const string &model_name) { return false; }

    virtual bool Train(const vector<Mat> &images,
                const vector<int> &labels) { return false; }
    virtual bool Extract(const vector<Mat> &images, Mat *feats) = 0;
    bool ExtractFeat(const Mat &image, Mat *feat);

    inline int feat_dim() const { return feat_dim_; }
    inline void set_feat_dim(int feat_dim) { feat_dim_ = feat_dim; }

private:
    int feat_dim_;
};
}  // namespace ghk

#endif  // FINAL_EXTRACTOR_H_

