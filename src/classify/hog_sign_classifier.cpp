/*************************************************************************
    > File Name: hog_sign_classifier.cpp
    > Author: Guo Hengkai
    > Description: HOG sign classifier class implementation using SVM
    > Created Time: Mon 25 May 2015 03:37:00 PM CST
 ************************************************************************/
#include "hog_sign_classifier.h"
#include "dataset.h"
#include "hog_extractor.h"
#include "svm_classifier.h"
#include "file_util.h"
#include "test_util.h"

namespace ghk
{
bool HogSignClassifier::Save(const string &model_name) const
{
    vector<float> param(1, img_size_);
    if (!SaveMat(model_name + "_para", Mat(), param))
    {
        printf("Fail to save the parameter.\n");
        return false;
    }
    if (!svm_classifier_.Save(model_name + "_svm"))
    {
        printf("Fail to save SVM.\n");
        return false;
    }
    return true;
}

bool HogSignClassifier::Load(const string &model_name)
{
    vector<float> param;
    Mat tmp;
    if (!LoadMat(model_name + "_para", &tmp, &param))
    {
        printf("Fail to load the parameter.\n");
        return false;
    }
    if (!svm_classifier_.Load(model_name + "_svm"))
    {
        printf("Fail to load SVM.\n");
        return false;
    }
    return true;
}

bool HogSignClassifier::Train(const Dataset &dataset)
{
    // Get training data
    vector<Mat> images;
    vector<int> labels;
    size_t n = dataset.GetClassifyNum(true);
    Size img_size(img_size_, img_size_);
    printf("Preparing training data...\n");
    for (size_t i = 0; i < n; ++i)
    {
        Mat image;
        dataset.GetClassifyImage(true, i, &image, img_size);
        cv::cvtColor(image, image, CV_BGR2GRAY);
        images.push_back(image);
        labels.push_back(dataset.GetClassifyLabel(true, i));
    }

    // Feature extraction
    Mat feats;
    printf("Extracting features...\n");
    hog_extractor_.Extract(images, &feats);

    // Train the SVM classifier
    printf("Training SVM classifier...\n");
    svm_classifier_.Train(feats, labels);
    printf("Training done! Now testing...\n");

    // Test on training
    vector<int> predict_labels;
    svm_classifier_.Predict(feats, &predict_labels);
    float rate, fp;
    EvaluateClassify(labels, predict_labels, CLASS_NUM, false, &rate, &fp);
    printf("Test on training rate: %0.2f%%\n", rate * 100);

    return true;
}

bool HogSignClassifier::Test(const Dataset &dataset)
{
    // Get test data
    vector<Mat> images;
    vector<int> labels;
    size_t n = dataset.GetClassifyNum(false);
    Size img_size(img_size_, img_size_);
    printf("Preparing testing data...\n");
    for (size_t i = 0; i < n; ++i)
    {
        Mat image;
        dataset.GetClassifyImage(false, i, &image, img_size);
        cv::cvtColor(image, image, CV_BGR2GRAY);
        images.push_back(image);
        labels.push_back(dataset.GetClassifyLabel(false, i));
    }

    // Test on test dataset
    vector<int> predict_labels;
    Predict(images, &predict_labels);
    float rate, fp;
    EvaluateClassify(labels, predict_labels, CLASS_NUM, false, &rate, &fp);
    printf("Test rate: %0.2f%%\n", rate * 100);

    return true;
}

bool HogSignClassifier::Predict(const vector<Mat> &images,
        vector<int> *labels)
{
    if (labels == nullptr)
    {
        return false;
    }

    // Feature extraction
    Mat feats;
    printf("Extracting features...\n");
    hog_extractor_.Extract(images, &feats);

    // Prediction
    printf("Predicting with SVM...\n");
    svm_classifier_.Predict(feats, labels);
    printf("Prediction done!\n");

    return true;
}
}  // namespace ghk
