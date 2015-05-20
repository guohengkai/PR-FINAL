/*************************************************************************
    > File Name: eigen_extractor.h
    > Author: Guo Hengkai
    > Description: Feature extractor class definition using PCA projection
    > Created Time: Wed 20 May 2015 08:52:16 AM CST
 ************************************************************************/
#ifndef FINAL_EIGEN_EXTRACTOR_H_
#define FINAL_EIGEN_EXTRACTOR_H_

#include "common.h"
#include "extractor.h"

namespace ghk
{
class EigenExtractor: public Extractor
{
public:
    explicit EigenExtractor(int feat_dim);

    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);

    virtual bool Train(const vector<Mat> &images,
                const vector<int> &labels);
    virtual bool Extract(const vector<Mat> &images, Mat *feats) const;

private:
    cv::PCA pca_;
};
}  // namespace ghk

#endif  // FINAL_EIGEN_EXTRACTOR_H_

