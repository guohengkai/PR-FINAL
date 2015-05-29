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
#include "timer.h"

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

    // Find random negative sample
    srand(time(NULL));
    vector<Mat> neg_images;
    auto neg_num = images.size() / (CLASS_NUM - 1) * 2;
    printf("Randomly getting negative samples...\n");
    if (!dataset.GetRandomNegImage(neg_num, img_size, &neg_images))
    {
        printf("Fail to get negative samples.\n");
        return false;
    }
    images.insert(images.end(), neg_images.begin(), neg_images.end());
    vector<int> neg_labels(neg_images.size(), 0);
    labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());

    Timer timer;
    // Feature extraction
    Mat feats;
    printf("Extracting features...\n");
    timer.Start();
    hog_extractor_.Extract(images, &feats);
    float t1 = timer.Snapshot();
    printf("Time for extraction: %0.3fs\n", t1);

    // Train the SVM classifier
    printf("Training SVM classifier...\n");
    svm_classifier_.Train(feats, labels);
    float t2 = timer.Snapshot();
    printf("Time for training SVM: %0.3fs\n", t2 - t1);
    labels.erase(labels.begin() + labels.size() - neg_images.size(), labels.end());
    feats.resize(labels.size());

    // Test on training before mining
    vector<int> predict_labels;
    svm_classifier_.Predict(feats, &predict_labels);
    float rate, fp;
    EvaluateClassify(labels, predict_labels, CLASS_NUM, false, &rate, &fp);
    printf("Test on training rate before mining: %0.2f%%\n", rate * 100);

    // Test on testing dataset
    Test(dataset);

    // Mining hard negative sample
    timer.Start();
    printf("Mining hard negative samples...\n");
    Mat neg_feats;
    if (!MiningHardSample(dataset, neg_num, img_size, &neg_feats))
    {
        printf("Fail to retrain SVM.\n");
        return true;  // Because the original model can be used
    }
    float t3 = timer.Snapshot();
    printf("Time for mining: %0.3fs\n", t3);
    neg_labels.resize(neg_feats.rows, 0);
    labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());
    feats.push_back(neg_feats);

    // Retrain SVM classifier
    printf("Retraining SVM classifier...\n");
    svm_classifier_.Train(feats, labels);
    float t4 = timer.Snapshot();
    printf("Time for retrain SVM: %0.3fs\n", t4 - t3);
    labels.erase(labels.begin() + labels.size() - neg_feats.rows, labels.end());
    feats.resize(labels.size());
    printf("Total time: %0.3fs\n", t4 + t2);

    // Test on training again
    svm_classifier_.Predict(feats, &predict_labels);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, false, &rate, &fp);
    printf("Test on training rate after mining: %0.2f%%\n", rate * 100);

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
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true, &rate, &fp);
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

    Timer timer;
    // Feature extraction
    Mat feats;
    printf("Extracting features...\n");
    timer.Start();
    hog_extractor_.Extract(images, &feats);
    float t1 = timer.Snapshot();
    printf("Time for extration: %0.3fs\n", t1);

    // Prediction
    printf("Predicting with SVM...\n");
    svm_classifier_.Predict(feats, labels);
    float t2 = timer.Snapshot();
    printf("Time for classification: %0.3fs\n", t2 - t1);
    printf("Prediction done! Total time for %d images: %0.3fs\n",
            feats.rows, t2);

    return true;
}

bool HogSignClassifier::MiningHardSample(const Dataset &dataset,
        size_t neg_num, Size image_size, Mat *neg_feats)
{
    if (neg_feats == nullptr)
    {
        return false;
    }

    vector<size_t> image_idxs;
    for (size_t i = 0; i < dataset.GetFullImageNum(); ++i)
    {
        image_idxs.push_back(i);
    }
    std::random_shuffle(image_idxs.begin(), image_idxs.end());

    neg_feats->resize(0);
    const int step = 10;
    for (auto idx: image_idxs)
    {
        Mat full_image;
        if (!dataset.GetFullImage(idx, &full_image))
        {
            continue;
        }
        
        for (int x = 0; x < full_image.cols; x += step)
            for (int y = 0; y < full_image.rows; y += step)
                for (size_t size_idx = 0; size_idx < SIZE_LIST.size(); ++size_idx)
                {
                    int size = SIZE_LIST[size_idx];
                    if (x + size >= full_image.cols || y + size >= full_image.rows)
                    {
                        break;
                    }

                    Rect rect(x, y, size, size);
                    if (dataset.IsNegativeImage(idx, rect))
                    {
                        Mat image = full_image(rect).clone();
                        cv::resize(image, image, image_size);
                        cv::cvtColor(image, image, CV_BGR2GRAY);
                        Mat feat_row;
                        if (!hog_extractor_.ExtractFeat(image, &feat_row))
                        {
                            continue;
                        }

                        // False positive
                        int res = svm_classifier_.PredictSample(feat_row);
                        // cout << res << ",";
                        if (res != 0)
                        {
                            neg_feats->push_back(feat_row);
                            if (neg_feats->rows >= static_cast<int>(neg_num))
                            {
                                // cout << endl;
                                return true;
                            }
                        }
                    }
                }

    }
    return true;
}
}  // namespace ghk
