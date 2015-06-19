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
    return false;
}

bool HogSignDetector::Detect(const vector<Mat> &images,
        vector<vector<Rect>> *rects, vector<int> *labels)
{
    return false;
}
}  // namespace ghk
