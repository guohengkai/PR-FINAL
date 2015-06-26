/*************************************************************************
    > File Name: src/detect/hog_sign_detector.cpp
    > Author: Guo Hengkai
    > Description: HOG sign detector class implementation
    > Created Time: Wed 17 Jun 2015 11:20:30 AM CST
 ************************************************************************/
#include "hog_sign_detector.h"
#include "dataset.h"
#include "file_util.h"
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

    /*
    for (size_t i = 0; i < images.size(); ++i)
    {
        stringstream ss;
        ss << labels[i] << "/" << i;
        cv::imwrite("/home/ghk/Src/PR-HW/PR-FINAL/data/detect/" + ss.str() + ".jpg", images[i]);
    }
    */
    
    // Train the classifier
    if (!classifier_.Train(dataset, images, labels))
    {
        return false;
    }

    // Test the detector
    th_ = 0.0f;
    // return Test(dataset);
    return true;
}

bool HogSignDetector::Test(const Dataset &dataset)
{
    string suffix = "_rf_with";
    vector<vector<Rect>> rects, rects_truth;
    vector<vector<int>> labels, labels_truth;
    vector<vector<float>> probs;
    size_t n = dataset.GetDetectNum(false);
    int pos_num = 0;
    int win_num = 0; 
    printf("Start to detect on %zu images...\n", n);
    for (size_t i = 0; i < n; ++i)
    {
        if (i % 100 == 0)
        {
            printf("%zu, \n", i);
            fflush(stdout);
        }
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

        // Get detection result
        vector<Mat> image_vec(1, image);
        vector<vector<Rect>> rect_vec;
        vector<vector<int>> label_vec;
        vector<vector<float>> prob_vec;
        int temp_num;
        if (!Detect(image_vec, &rect_vec, &label_vec, &prob_vec, &temp_num))
        {
            return false;
        }
        win_num += temp_num;

        rects.push_back(rect_vec[0]);
        labels.push_back(label_vec[0]);
        probs.push_back(prob_vec[0]);
    }
    printf("\nTotal detected: %zu\n", rects.size());

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
                 probs[i][j] > max_p && !temp_res[j])
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

    // Find the threshold for SVM
    float rate = UpdateThreshold(results, scores,
            "./result/dect_curve" + suffix + ".txt", pos_num, win_num, &th_);
    printf("Accuray under 10^-4 FPPW: %0.2f%%\n", rate * 100);

    // Calculate confuse matrix
    Mat confuse_mat = Mat::zeros(CLASS_NUM, CLASS_NUM, CV_32F);
    Mat class_sum = Mat::zeros(CLASS_NUM, 1, CV_32F);
    for (size_t i = 0; i < n; ++i)
    {
        vector<bool> temp_res(labels[i].size(), false);

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
                 probs[i][j] > max_p && !temp_res[j])
                {
                    max_p = probs[i][j];
                    idx = j;
                }
            }
            if (max_p >= 0)
            {
                temp_res[idx] = true;
            }
            else
            {
                ++confuse_mat.at<float>(labels_truth[i][k], 0);
            }
            ++class_sum.at<float>(labels_truth[i][k], 0);
        }

        for (size_t j = 0; j < labels[i].size(); ++j)
        {
            if (temp_res[j])
            {
                ++confuse_mat.at<float>(labels[i][j], labels[i][j]);
            }
            else
            {
                ++confuse_mat.at<float>(0, labels[i][j]);
                ++class_sum.at<float>(0, 0);
            }
        }
    }
    confuse_mat = confuse_mat.mul(cv::repeat(1.0f / class_sum, 1, CLASS_NUM));
    SaveMat("./result/det_detect" + suffix, confuse_mat);
    
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
        vector<vector<float>> *probs, int *win_num)
{
    if (rects == nullptr || labels == nullptr)
    {
        return false;
    }
    // th_ = 0.95;
    rects->clear();
    labels->clear();
    probs->clear();
    if (win_num != nullptr)
    {
        *win_num = 0;
    }
    for (auto image: images)
    {
        cout << "Searching image patches..." << endl;
        Mat gray;
        cvtColor(image, gray, CV_BGR2GRAY);
        vector<Mat> image_vec;
        vector<Rect> all_rects;
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
                    Mat image_patch(gray(rect).clone());
                    resize(image_patch, image_patch, image_size_);
                    all_rects.push_back(rect);
                    image_vec.push_back(image_patch);
                }
        }

        if (win_num != nullptr)
        {
            *win_num += static_cast<int>(all_rects.size());
        }

        cout << "Predicting using SVM..." << endl;
        vector<int> label_vec;
        vector<float> prob_vec;
        if (!classifier_.Predict(image_vec, &label_vec, &prob_vec))
        {
            return false;
        }

        vector<Rect> res_rects;
        vector<int> res_labels;
        vector<float> res_probs;
        for (size_t i = 0; i < all_rects.size(); ++i)
        {
            if (label_vec[i] > 0 && prob_vec[i] > th_)  // Positive response
            {
                res_rects.push_back(all_rects[i]);
                res_labels.push_back(label_vec[i]);
                res_probs.push_back(prob_vec[i]);
            }
        }
        cout << "Totally " << res_rects.size() << " positions detected." << endl;

        // Merge detect results
        MergeRects(res_rects, res_labels, res_probs, 0.667f);
        cout << "Totally " << res_rects.size() << " positions after merging." << endl;

        rects->push_back(res_rects);
        labels->push_back(res_labels);
        probs->push_back(res_probs);
    }

    return true;
}
}  // namespace ghk
