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
#include "test_util.h"

namespace ghk
{
KnnSignClassifier::KnnSignClassifier(bool use_fisher,
            int near_num, int eigen_feat_num, int img_size):
            use_fisher_(use_fisher), eigen_extractor_(eigen_feat_num),
            knn_classifier_(near_num), img_size_(img_size), threshold_(FLT_MAX)
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
    param.push_back(img_size_);
    param.push_back(threshold_);
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
    img_size_ = param[1];
    threshold_ = param[2];

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

    // Train the extractor
    printf("Training extractor...\n");
    extractor_->Train(images, labels);

    // Feature extraction
    Mat feats;
    printf("Extracting features...\n");
    extractor_->Extract(images, &feats);

    // Train the KNN classifier
    printf("Training KNN classifier...\n");
    knn_classifier_.Train(feats, labels);

    // Train the threshold
    printf("Training the threshold...\n");
    if (!TrainThreshold(dataset))
    {
        printf("Fail to train the threshold.\n");
        return false;
    }
    printf("Training done! Now testing...\n");

    // Test on training
    vector<int> predict_labels;
    knn_classifier_.Predict(feats, &predict_labels);
    float rate, fp;
    EvaluateClassify(labels, predict_labels, CLASS_NUM, false, &rate, &fp);
    printf("Test on training rate: %0.2f%%\n", rate * 100);

    return true;
}

bool KnnSignClassifier::Test(const Dataset &dataset)
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
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true, &rate, &fp);
    printf("Test rate: %0.2f%%\n", rate * 100);

    return true;
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
    printf("Extracting features...\n");
    extractor_->Extract(images, &feats);

    // Prediction
    vector<float> distance;
    printf("Predicting with KNN...\n");
    knn_classifier_.Predict(feats, labels, &distance);
    for (size_t i = 0; i < distance.size(); ++i)
    {
        if (distance[i] > threshold_)
        {
            (*labels)[i] = 0;
        }
    }
    printf("Prediction done!\n");

    return true;
}

bool KnnSignClassifier::TrainThreshold(const Dataset &dataset)
{
    srand(time(NULL));
    const size_t neg_num = 2000;
    vector<Mat> neg_images;
    Size img_size(img_size_, img_size_);
    printf("Randomly getting negative samples...");
    if (!dataset.GetRandomNegImage(neg_num, img_size, &neg_images))
    {
        printf("Fail to get negative samples.\n");
        return false;
    }

    Mat feats;
    printf("Extracting features...\n");
    extractor_->Extract(neg_images, &feats);

    vector<float> distance;
    vector<int> labels;
    printf("Getting distance with KNN...\n");
    knn_classifier_.Predict(feats, &labels, &distance);

    float mean = 0, deviation = 0;
    for (auto dis: distance)
    {
        mean += dis;
        deviation += dis * dis;
    }
    mean /= distance.size();
    deviation = sqrt(deviation / distance.size() - mean * mean);
    printf("Mean: %0.4f, Std: %0.4f\n", mean, deviation);
    threshold_ = mean + 2 * deviation;
    
    return true;
}
}  // namespace ghk
