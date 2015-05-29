/*************************************************************************
    > File Name: hog_sign_classifier.h
    > Author: Guo Hengkai
    > Description: HOG sign classifier class definition using SVM
    > Created Time: Mon 25 May 2015 03:24:27 PM CST
 ************************************************************************/
#ifndef FINAL_HOG_SIGN_CLASSIFIER_H_
#define FINAL_HOG_SIGN_CLASSIFIER_H_

#include "common.h"
#include "dataset.h"
#include "hog_extractor.h"
#include "svm_classifier.h"
#include "sign_classifier.h"

namespace ghk
{
class HogSignClassifier: public SignClassifier
{
public:
    HogSignClassifier(float c = 125, int img_size = 100):
        svm_classifier_(c), img_size_(img_size) {}

    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);

    virtual bool Train(const Dataset &dataset);
    virtual bool Test(const Dataset &dataset);
    virtual bool Predict(const vector<Mat> &images,
            vector<int> *labels);

private:
    HogExtractor hog_extractor_;
    SvmClassifier svm_classifier_;
    int img_size_;

    bool MiningHardSample(const Dataset &dataset,
            size_t neg_num, Size image_size, Mat *neg_feats);
};
}  // namespace ghk

#endif  // FINAL_HOG_SIGN_CLASSIFIER_H_

