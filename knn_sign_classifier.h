/*************************************************************************
    > File Name: knn_sign_classifier.h
    > Author: Guo Hengkai
    > Description: KNN sign classifier class definition using eigenface or fisherface features
    > Created Time: Sat 23 May 2015 10:11:43 PM CST
 ************************************************************************/
#ifndef FINAL_KNN_SIGN_CLASSIFIER_H_
#define FINAL_KNN_SIGN_CLASSIFIER_H_

#include "common.h"
#include "dataset.h"
#include "extractor.h"
#include "eigen_extractor.h"
#include "fisher_extractor.h"
#include "knn_classifier.h"
#include "sign_classifier.h"

namespace ghk
{
class KnnSignClassifier: public SignClassifier
{
public:
    KnnSignClassifier(bool use_fisher, int near_num,
                                    int eigen_feat_num = 0);

    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);

    virtual bool Train(const Dataset &dataset);
    virtual bool Test(const Dataset &dataset);
    virtual bool Predict(const vector<Mat> &images,
            vector<int> *labels);

private:
    bool use_fisher_;
    Extractor *extractor_;
    EigenExtractor eigen_extractor_;
    FisherExtractor fisher_extractor_;
    KnnClassifier knn_classifier_;
};
}  // namespace ghk

#endif  // FINAL_KNN_SIGN_CLASSIFIER_H_

