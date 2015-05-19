/*************************************************************************
    > File Name: svm_classifier.h
    > Author: Guo Hengkai
    > Description: SVM classifier class definition for libsvm
    > Created Time: Tue 19 May 2015 01:47:59 PM CST
 ************************************************************************/
#ifndef FINAL_SVM_CLASSIFIER_H_
#define FINAL_SVM_CLASSIFIER_H_

#include "classifier.h"
#include "common.h"
#include "svm.h"

namespace ghk
{
class SvmClassifier: public Classifier
{
public:
    SvmClassifier(): svm_model_(NULL) {}
    ~SvmClassifier();

    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);

    virtual bool Train(const Mat &feats, const vector<int> &labels);
    virtual bool Predict(const Mat &feats, vector<int> *labels) const;

private:
    svm_model *svm_model_;

    // Normalization parameters: X' = (X - A) / B
    Mat normA_;
    Mat normB_;
    
    void Normalize(const Mat &feats, Mat *feats_norm) const;
    void PrepareParameter(int feat_dim, svm_parameter *param) const;
    void PrepareProblem(const Mat &feats, const vector<int> &labels,
            svm_problem *problem, svm_node *x_space) const;
};
}  // namespace ghk

#endif  // FINAL_SVM_CLASSIFIER_H_
