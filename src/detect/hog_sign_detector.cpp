/*************************************************************************
    > File Name: src/detect/hog_sign_detector.cpp
    > Author: Guo Hengkai
    > Description: HOG sign detector class implementation
    > Created Time: Wed 17 Jun 2015 11:20:30 AM CST
 ************************************************************************/
#include "hog_sign_detector.h"
#include "dataset.h"
#include "sign_detector.h"
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

    // Test the detector
    th_ = 0.0f;
    return Test(dataset);
}

bool HogSignDetector::Test(const Dataset &dataset)
{
    vector<vector<Rect>> rects, rects_truth;
    vector<vector<int>> labels, labels_truth;
    vector<vector<float>> probs;
    size_t n = dataset.GetDetectNum(false);
    int pos_num = 0;
    
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
        pos_num += static_cast<int>(res_rects.size());

        // Get image
        Mat image;
        if (!dataset.GetDetectImage(false, i, &image))
        {
            return false;
        }
        cv::cvtColor(image, image, CV_BGR2GRAY);

        // Get detection result
        vector<Mat> image_vec(1, image);
        vector<vector<Rect>> rect_vec;
        vector<vector<int>> label_vec;
        vector<vector<float>> prob_vec;
        if (!Detect(image_vec, &rect_vec, &label_vec, &prob_vec))
        {
            return false;
        }

        // Merge detect results
        vector<size_t> idx;
        MergeRects(rect_vec[0], label_vec[0], prob_vec[0], &idx);
        res_rects.clear();
        res_labels.clear();
        vector<float> res_prob;
        for (auto k: idx)
        {
            res_rects.push_back(rect_vec[0][k]);
            res_labels.push_back(label_vec[0][k]);
            res_prob.push_back(prob_vec[0][k]);
        }
        rects.push_back(res_rects);
        labels.push_back(res_labels);
        probs.push_back(res_prob);
    }

    // Evaluate the rectangles
    vector<bool> results;
    vector<float> scores;
    for (size_t i = 0; i < n; ++i)
    {
        vector<bool> temp_res(labels[i].size(), false);
        for (size_t j = 0; j < labels[i].size(); ++j)
        {
            scores.push_back(probs[i][j]);
        }

        for (size_t k = 0; k < labels_truth[i].size(); ++k)
        {
            Rect rect = rects_truth[i][k];
            size_t idx = 0;
            float max_p = -1;
            for (size_t j = 0; j < labels[i].size(); ++j)
            {
                if (labels[i][j] == labels_truth[i][k] && static_cast<float>(
                    (rect & rects[i][j]).area())
                 / ((rect | rects[i][j]).area()) >= 0.5 &&
                 probs[i][j] > max_p)
                {
                    max_p = probs[i][j];
                    idx = j;
                }
            }
            if (max_p >= 0)
            {
                temp_res[idx] = true;
            }
        }

        results.insert(results.end(), temp_res.begin(), temp_res.end());
    }
    float rate = UpdateThreshold(results, scores,
            "./result/dect_curve.txt", pos_num, &th_);
    printf("Accuray under 10^-4 FPPW: %0.2f%%\n", rate * 100);
    return true;
}

bool HogSignDetector::Detect(const vector<Mat> &images,
        vector<vector<Rect>> *rects, vector<vector<int>> *labels)
{
    vector<vector<float>> probs;
    return Detect(images, rects, labels, &probs);
}

bool HogSignDetector::Detect(const vector<Mat> &images,
        vector<vector<Rect>> *rects, vector<vector<int>> *labels,
        vector<vector<float>> *probs)
{
    if (rects == nullptr || labels == nullptr)
    {
        return false;
    }

    rects->clear();
    labels->clear();
    probs->clear();
    for (auto image: images)
    {
        vector<Rect> res_rects;
        vector<int> res_labels;
        vector<float> res_probs;
        for (auto size: SIZE_LIST)
        {
            for (int x = 0; x < image.cols; x += DETECT_STEP)
                for (int y = 0; y < image.rows; y += DETECT_STEP)
                {
                    if (x + size >= image.cols || y + size >= image.rows)
                    {
                        continue;
                    }

                    Rect rect(x, y, size, size);
                    vector<Mat> image_vec(1, image(rect));
                    vector<int> label_vec;
                    vector<float> prob_vec;
                    if (!classifier_.Predict(image_vec, &label_vec, &prob_vec))
                    {
                        return false;
                    }
                    if (label_vec[0] > 0)  // Positive response
                    {
                        res_rects.push_back(rect);
                        res_labels.push_back(label_vec[0]);
                        res_probs.push_back(prob_vec[0]);
                    }
                }
        }

        rects->push_back(res_rects);
        labels->push_back(res_labels);
        probs->push_back(res_probs);
    }

    return true;
}
}  // namespace ghk
