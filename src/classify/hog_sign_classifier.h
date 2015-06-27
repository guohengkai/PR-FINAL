/*************************************************************************
    > File Name: hog_sign_classifier.h
    > Author: Guo Hengkai
    > Description: HOG sign classifier class definition
    > Created Time: Mon 25 May 2015 03:24:27 PM CST
 ************************************************************************/
#ifndef FINAL_HOG_SIGN_CLASSIFIER_H_
#define FINAL_HOG_SIGN_CLASSIFIER_H_

#include "common.h"
#include "dataset.h"
#include "classifier.h"
#include "hog_extractor.h"
#include "forest_classifier.h"
#include "svm_classifier.h"
#include "sign_classifier.h"

namespace ghk
{
class HogSignClassifier: public SignClassifier
{
public:
    HogSignClassifier(int num_orient = 8, int cell_size = 8,
            float c = 125, int img_size = 100, bool use_svm = true):
        hog_extractor_(num_orient, cell_size),
        svm_classifier_(c), forest_classifier_(13, 10, 200),
        img_size_(img_size)
    {
        use_svm_ = !use_svm;  // Force to update the pointer
        set_use_svm(use_svm);
    }

    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);

    virtual bool Train(const Dataset &dataset);
    bool Train(const Dataset &dataset, vector<Mat> &images,
            vector<int> &labels);
    virtual bool Test(const Dataset &dataset);
    virtual bool FullTest(const Dataset &dataset,
            const string &dir);
    virtual bool Predict(const vector<Mat> &images,
            vector<int> *labels);
    bool Predict(const vector<Mat> &images,
            vector<int> *labels, vector<float> *probs);

    inline void set_use_svm(bool use_svm)
    {
        if (use_svm != use_svm_)
        {
            use_svm_ = use_svm;
            if (use_svm_)
            {
                classifier_ = &svm_classifier_;
            }
            else
            {
                classifier_ = &forest_classifier_;
            }
        }
    }

private:
    HogExtractor hog_extractor_;
    SvmClassifier svm_classifier_;
    ForestClassifier forest_classifier_;
    Classifier *classifier_;
    bool use_svm_;

    int img_size_;

    bool MiningHardSample(const Dataset &dataset,
            size_t neg_num, Size image_size, Mat *neg_feats);
};
}  // namespace ghk

#endif  // FINAL_HOG_SIGN_CLASSIFIER_H_

