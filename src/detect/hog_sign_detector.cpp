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
    // labels = vector<int>(labels.size(), 1);
    if (!classifier_.Train(dataset, images, labels))
    {
        return false;
    }

    return true;
}

bool HogSignDetector::Test(const Dataset &dataset)
{
    string suffix = "_rf_without";
    vector<vector<Rect>> rects, rects_truth;
    vector<vector<int>> labels, labels_truth;
    vector<vector<float>> probs;
    size_t n = dataset.GetDetectNum(false);
    // size_t n = dataset.GetDetectNum(true);
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
        // if (!dataset.GetDetectRects(true, i, &res_rects))
        {
            return false;
        }
        if (!dataset.GetDetectLabels(false, i, &res_labels))
        // if (!dataset.GetDetectLabels(true, i, &res_labels))
        {
            return false;
        }
        rects_truth.push_back(res_rects);
        labels_truth.push_back(res_labels);
        // labels_truth.push_back(vector<int>(res_labels.size(), 1));
        pos_num += static_cast<int>(res_rects.size());

        // Get image
        Mat image;
        if (!dataset.GetDetectImage(false, i, &image))
        // if (!dataset.GetDetectImage(true, i, &image))
        {
            return false;
        }

        // Get detection result
        vector<Mat> image_vec(1, image);
        vector<vector<Rect>> rect_vec;
        vector<vector<int>> label_vec;
        vector<vector<float>> prob_vec;
        int temp_num;
        if (!Detect(image_vec, &rect_vec, &label_vec, &prob_vec, &temp_num, false))
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
    Mat rate;
    bool is_set = false;
    for (float th = 0.0f; th <= 1.0f; th += 0.02f)
    {
        cout << th << ", " << flush;
        float pos = 0;
        float neg = 0;

        for (size_t i = 0; i < n; ++i)
        {
            vector<Rect> res_rects;
            vector<int> res_labels;
            vector<float> res_probs;
            for (size_t j = 0; j < rects[i].size(); ++j)
            {
                if (probs[i][j] >= th)
                {
                    res_rects.push_back(rects[i][j]);
                    res_labels.push_back(labels[i][j]);
                    res_probs.push_back(probs[i][j]);
                }
            }
            // MergeRects(res_rects, res_labels, res_probs, 0.667f);

            vector<bool> temp_res(res_labels.size(), false);
            for (size_t k = 0; k < labels_truth[i].size(); ++k)
            {
                Rect rect = rects_truth[i][k];
                size_t idx = 0;
                float max_p = -1;
                for (size_t j = 0; j < res_labels.size(); ++j)
                {
                    if (res_labels[j] == labels_truth[i][k] && static_cast<float>(
                        (rect & res_rects[j]).area())
                     / ((rect | res_rects[j]).area()) >= 0.5 &&
                     res_probs[j] > max_p && !temp_res[j])
                    {
                        max_p = res_probs[j];
                        idx = j;
                    }
                }
                if (max_p >= 0)
                {
                    temp_res[idx] = true;
                }
            }

            for (size_t k = 0; k < res_labels.size(); ++k)
            {
                if (temp_res[k])
                {
                    ++pos;
                }
                else
                {
                    ++neg;
                }
            }
        }

        Mat rate_row = Mat::zeros(1, 2, CV_32F);
        rate_row.at<float>(0, 0) = 1 - pos / pos_num;
        rate_row.at<float>(0, 1) = neg / win_num;
        rate.push_back(rate_row);

        if (rate_row.at<float>(0, 1) < 1e-2 && !is_set)
        {
            th_ = th;
            is_set = true;
        }
    }
    cout << endl;
    SaveMat("./result/dect_curve" + suffix, rate);
    if (!is_set)
    {
        th_ = 1.0f;
    }

    /*
    // Find the threshold for SVM
    float rate = UpdateThreshold(results, scores,
            "./result/dect_curve" + suffix + ".txt", pos_num, win_num, &th_);
    printf("Accuray under 10^-4 FPPW: %0.2f%%\n", rate * 100); 

    // Calculate confuse matrix
    Mat confuse_mat = Mat::zeros(CLASS_NUM, CLASS_NUM, CV_32F);
    Mat class_sum = Mat::zeros(CLASS_NUM, 1, CV_32F);
    for (size_t i = 0; i < n; ++i)
    {
        vector<Rect> res_rects(rects[i]);
        vector<int> res_labels(labels[i]);
        vector<float> res_probs(probs[i]);
        rects[i].clear();
        labels[i].clear();
        probs[i].clear();
        for (size_t j = 0; j < res_rects.size(); ++j)
        {
            if (res_probs[j] >= th_)
            {
                rects[i].push_back(res_rects[j]);
                labels[i].push_back(res_labels[j]);
                probs[i].push_back(res_probs[j]);
            }
        }

        MergeRects(rects[i], labels[i], probs[i], 0.667f);
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
    SaveMat("./result/det_detect" + suffix, confuse_mat);*/
    
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
        vector<vector<float>> *probs, int *win_num, bool is_merge)
{
    if (rects == nullptr || labels == nullptr)
    {
        return false;
    }
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

        cout << "Predicting..." << endl;
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

        if (is_merge)
        {
            // Merge detect results
            MergeRects(res_rects, res_labels, res_probs, 0.667f);
            cout << "Totally " << res_rects.size() << " positions after merging." << endl;
        }

        rects->push_back(res_rects);
        labels->push_back(res_labels);
        probs->push_back(res_probs);
    }

    return true;
}
}  // namespace ghk
