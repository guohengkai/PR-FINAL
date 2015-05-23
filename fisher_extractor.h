/*************************************************************************
    > File Name: fisher_extractor.h
    > Author: Guo Hengkai
    > Description: Feature extractor class definition using Fisher projection
    > Created Time: Wed 20 May 2015 09:44:11 AM CST
 ************************************************************************/
#ifndef FINAL_FISHER_EXTRACTOR_H_
#define FINAL_FISHER_EXTRACTOR_H_

#include "common.h"
#include "extractor.h"

namespace ghk
{
class FisherExtractor: public Extractor
{
public:
    FisherExtractor();

    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);

    virtual bool Train(const vector<Mat> &images,
                const vector<int> &labels);
    virtual bool Extract(const vector<Mat> &images, Mat *feats);

private:
    Mat eigen_vector_;
    Mat mean_;
};
}  // namespace ghk

#endif  // FINAL_FISHER_EXTRACTOR_H_

