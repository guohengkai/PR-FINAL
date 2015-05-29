/*************************************************************************
    > File Name: svm_classifier.cpp
    > Author: Guo Hengkai
    > Description: SVM classifier class implementation for libsvm
    > Created Time: Tue 19 May 2015 01:58:23 PM CST
 ************************************************************************/
#include "svm_classifier.h"
#include "mat_util.h"
#include "file_util.h"

namespace ghk
{
SvmClassifier::~SvmClassifier()
{
    if (svm_model_ != NULL)
    {
        svm_free_and_destroy_model(&svm_model_);
    }
}

bool SvmClassifier::Save(const string &model_name) const
{
    // Save the normalization parameters and penalty coefficient
    vector<float> c(1, c_);
    if (!SaveMat(model_name + "_normA", normA_, c))
    {
        return false;
    }
    if (!SaveMat(model_name + "_normB", normB_))
    {
        return false;
    }

    // Save the SVM model
    if (svm_model_ == NULL)
    {
        return false;
    }
    if (svm_save_model((model_name + FILE_EXT).c_str(),
            svm_model_) != 0)
    {
        return false;
    }

    return true;
}

bool SvmClassifier::Load(const string &model_name)
{
    // Load the normalization parameters and penalty coefficient
    vector<float> c;
    if (!LoadMat(model_name + "_normA", &normA_, &c))
    {
        return false;
    }
    if (!LoadMat(model_name + "_normB", &normB_))
    {
        return false;
    }
    c_ = c[0];

    // Load the SVM model
    if (svm_model_ != NULL)
    {
        svm_free_and_destroy_model(&svm_model_);
    }
    svm_model_ = svm_load_model((model_name + FILE_EXT).c_str());

    return true;
}

void PrintNull(const char *s) {}

bool SvmClassifier::Train(const Mat &feats, const vector<int> &labels)
{
    if (svm_model_ != NULL)
    {
        svm_free_and_destroy_model(&svm_model_);
    }

    // Calculate the normalization parameters
    TrainNormalize(feats, &normA_, &normB_);

    // Normalize the features
    Mat feats_norm;
    Normalize(feats, &feats_norm);

    // Prepare the input for SVM
    svm_parameter param;
    svm_problem problem;
    svm_node* x_space = NULL;

    PrepareParameter(feats_norm.cols, &param);
    PrepareProblem(feats_norm, labels, &problem, x_space);

    // Train the SVM model
    svm_set_print_string_function(&PrintNull);  // Close the training output
    svm_model_ = svm_train(&problem, &param);

    // Release the parameters for training
    svm_destroy_param(&param);
    return true;

}

bool SvmClassifier::Predict(const Mat &feats, vector<int> *labels) const
{
    if (labels == nullptr)
    {
        return false;
    }

    int m = feats.cols;
    int n = feats.rows;
    labels->clear();

    // Normalize the features
    Mat feats_norm;
    Normalize(feats, &feats_norm);

    // Predict using SVM
    svm_node *x = static_cast<svm_node*>(malloc((m + 1) * sizeof(svm_node)));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            x[j].index = j + 1;
            x[j].value = feats_norm.at<float>(i, j);
        }
        x[m].index = -1;

        labels->push_back(svm_predict(svm_model_, x));
    }
    free(x);

    return true;
}

void SvmClassifier::Normalize(const Mat &feats, Mat *feats_norm) const
{
    ghk::Normalize(normA_, normB_, feats, feats_norm);
}

void SvmClassifier::PrepareParameter(int feat_dim, svm_parameter *param) const
{
    // default values
    param->svm_type = C_SVC;
    param->kernel_type = RBF;
    param->degree = 3;
    param->gamma = 1.0 / feat_dim;
    param->coef0 = 0;
    param->nu = 0.5;
    param->cache_size = 100;
    param->C = c_;
    param->eps = 1e-3;
    param->p = 0.1;
    param->shrinking = 1;
    param->probability = 0;
    param->nr_weight = 0;
    param->weight_label = NULL;
    param->weight = NULL;
}

void SvmClassifier::PrepareProblem(const Mat &feats, const vector<int> &labels,
        svm_problem *problem, svm_node *x_space) const
{
    int m = feats.cols;
    int n = feats.rows;
    problem->l = n;
    problem->y = static_cast<double*>(malloc(n * sizeof(double)));
    problem->x = static_cast<svm_node**>(malloc(n * sizeof(svm_node*)));
    x_space = static_cast<svm_node*>(malloc(n * (m + 1) * sizeof(svm_node)));

    for (int i = 0, k = 0; i < problem->l; ++i) {
        problem->y[i] = labels[i];
        problem->x[i] = &x_space[k];

        for (int32_t j = 0; j < feats.cols; ++j, ++k) {
            x_space[k].index = j + 1;
            x_space[k].value = feats.at<float>(i, j);
        }

        x_space[k++].index = -1;
    }

}
}  // namespace ghk
