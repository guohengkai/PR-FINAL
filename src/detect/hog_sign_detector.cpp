/*************************************************************************
    > File Name: src/detect/hog_sign_detector.cpp
    > Author: Guo Hengkai
    > Description: HOG sign detector class implementation
    > Created Time: Wed 17 Jun 2015 11:20:30 AM CST
 ************************************************************************/
#include "hog_sign_detector.h"
#include "dataset.h"
#include "test_util.h"

namespace ghk
{
bool HogSignDetector::Save(const string &model_name) const
{
    return classifier_.Save(model_name + "_cl");
}

bool HogSignDetector::Load(const string &model_name)
{
    return classifier_.Load(model_name + "_cl");
}

bool HogSignDetector::Train(const Dataset &dataset)
{
    // Prepare positive training data
    vector<Mat> images;
    vector<int> labels;
    printf("Preparing postive training data...\n");
    dataset.GetDetectPosImage(image_size_, &images, &labels, true);
    
    // Train the classifier
    if (!classifier_.Train(dataset, images, labels))
    {
        return false;
    }

    return true;
}

bool HogSignDetector::Test(const Dataset &dataset)
{
    vector<vector<Rect>> rects, rects_truth;
    vector<vector<int>> labels, labels_truth;
    size_t n = dataset.GetDetectNum(false);
    
    for (size_t i = 0; i < n; ++i)
    {
        // Get ground truth
        vector<Rect> res_rects;
        vector<int> res_labels;
        if (!dataset.GetDetectRects(false, i, &res_rects))
        {
            return false;
        }
        if (!dataset.GetDetectLabels(false, i, &res_labels))
        {
            return false;
        }
        rects_truth.push_back(res_rects);
        labels_truth.push_back(res_labels);

        // Get image
        Mat image;
        if (!dataset.GetDetectImage(false, i, &image))
        {
            return false;
        }
        cv::cvtColor(image, image, CV_BGR2GRAY);

        // Get detection result
        DetectSingle(image, &res_rects, &res_labels);
        rects.push_back(res_rects);
        labels.push_back(res_labels);
    }

    // EvaluateDetect(rects_truth, labels_truth, rects, labels);
    return true;
}

bool HogSignDetector::Detect(const vector<Mat> &images,
        vector<vector<Rect>> *rects, vector<vector<int>> *labels)
{
    if (rects == nullptr || labels == nullptr)
    {
        return false;
    }

    rects->clear();
    labels->clear();
    for (auto image: images)
    {
        
    }

    return true;
}
}  // namespace ghk
