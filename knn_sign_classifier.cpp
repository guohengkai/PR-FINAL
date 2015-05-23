/*************************************************************************
    > File Name: knn_sign_classifier.cpp
    > Author: Guo Hengkai
    > Description: KNN sign classifier class implemenation using eigenface or fisherface features
    > Created Time: Sat 23 May 2015 10:51:47 PM CST
 ************************************************************************/
#include "knn_sign_classifier.h"
#include "dataset.h"
#include "extractor.h"
#include "eigen_extractor.h"
#include "fisher_extractor.h"
#include "knn_classifier.h"
#include "file_util.h"

namespace ghk
{
KnnSignClassifier::KnnSignClassifier(bool use_fisher,
            int near_num, int eigen_feat_num): use_fisher_(use_fisher),
            eigen_extractor_(eigen_feat_num), knn_classifier_(near_num)
{
   if (use_fisher_) 
   {
       extractor_ = &fisher_extractor_;
   }
   else
   {
       extractor_ = &eigen_extractor_;
   }
}

bool KnnSignClassifier::Save(const string &model_name) const
{
    vector<float> param;
    if (use_fisher_)
    {
        param.push_back(1);
    }
    else
    {
        param.push_back(-1);
    }
    if (!SaveMat(model_name + "_para", Mat(), param))
    {
        printf("Fail to save the parameter.\n");
        return false;
    }

    if (!extractor_->Save(model_name + "_ext"))
    {
        printf("Fail to save the extractor.\n");
        return false;
    }
    if (!knn_classifier_.Save(model_name + "_c"))
    {
        printf("Fail to save the classifier.\n");
        return false;
    }
    return true;
}

bool KnnSignClassifier::Load(const string &model_name)
{
    vector<float> param;
    Mat tmp;
    if (!LoadMat(model_name + "_para", &tmp, &param))
    {
        printf("Fail to load the parameter.\n");
        return false;
    }
    if (param[0] > 0)
    {
        use_fisher_ = true;
        extractor_ = &fisher_extractor_;
    }
    else
    {
        use_fisher_ = false;
        extractor_ = &eigen_extractor_;
    }

    if (!extractor_->Load(model_name + "_ext"))
    {
        printf("Fail to load the extractor.\n");
        return false;
    }
    if (!knn_classifier_.Load(model_name + "_c"))
    {
        printf("Fail to load the classifier.\n");
        return false;
    }
    return true;
}

bool KnnSignClassifier::Train(const Dataset &dataset)
{
    // Get training data
    vector<Mat> images;
    vector<int> labels;
    size_t n = dataset.GetClassifyNum(true);
    for (size_t i = 0; i < n; ++i)
    {
        Mat image;
        dataset.GetClassifyImage(true, i, &image);
        images.push_back(image);
        labels.push_back(dataset.GetClassifyLabel(true, i));
    }

    // Feature extraction
    Mat feats;
    extractor_->Extract(images, &feats);

    // Train the KNN classifier
    knn_classifier_.Train(feats, labels);

    return true;
}

bool KnnSignClassifier::Test(const Dataset &dataset)
{
    return false;
}

bool KnnSignClassifier::Predict(const vector<Mat> &images,
        vector<int> *labels)
{
    if (labels == nullptr)
    {
        return false;
    }

    // Feature extraction
    Mat feats;
    extractor_->Extract(images, &feats);

    // Prediction
    knn_classifier_.Predict(feats, labels);

    return true;
}
}  // namespace ghk
