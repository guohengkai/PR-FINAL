/*************************************************************************
    > File Name: test_util.cpp
    > Author: Guo Hengkai
    > Description: Test function implementation
    > Created Time: Sat 16 May 2015 04:48:16 PM CST
 ************************************************************************/
#include "test_util.h"
#include "file_util.h"

namespace ghk
{
void EvaluateClassify(const vector<int> &ground_truth,
                      const vector<int> &predict_value,
                      int class_num, bool is_open,
                      float *tpr, float *fpr, Mat *result_mat)
{
    Mat result = Mat::zeros(class_num, class_num, CV_32F);
    Mat class_sum = Mat::zeros(class_num, 1, CV_32F);
    int tp_num = 0;
    int fp_num = 0;
    for (size_t i = 0; i < ground_truth.size(); ++i)
    {
        ++result.at<float>(ground_truth[i], predict_value[i]);
        ++class_sum.at<float>(ground_truth[i], 0);
        if (ground_truth[i] == 0)
        {
            if (is_open && predict_value[i] != 0)
            {
                ++fp_num;
            }
        }
        else
        {
            if (ground_truth[i] == predict_value[i])
            {
                ++tp_num;
            }
            else
            {
                ++fp_num;
            }
        }
    }

    *tpr = static_cast<float>(tp_num) / cv::sum(class_sum.rowRange(1, class_num))[0];
    *fpr = static_cast<float>(fp_num);
    if (is_open)
    {
        *fpr /= ((class_num - 1) * cv::sum(class_sum)[0]);
    }
    else
    {
        *fpr /= ((class_num - 2) * cv::sum(class_sum.rowRange(1, class_num))[0]);
    }

    if (result_mat != nullptr)
    {
        *result_mat = Mat(result.mul(cv::repeat(1.0f / class_sum, 1, class_num)));
    }
}

struct Result
{
    bool res;
    float score;
};

bool ResultComp(const Result &lhs, const Result &rhs)
{
    return lhs.score < rhs.score;
}

float UpdateThreshold(const vector<bool> &results, const vector<float> &scores,
        const string &file_name, int pos_num, int win_num, float *th, float fppw)
{
    if (results.empty())
    {
        return 0.0f;
    }
    vector<Result> res_vec;
    float pos = 0;
    float neg = 0;
    for (size_t i = 0; i < results.size(); ++i)
    {
        res_vec.push_back(Result{results[i], scores[i]});
        if (results[i])
        {
            ++pos;
        }
        else
        {
            ++neg;
        }
    }

    // Sort the score from high to low
    sort(res_vec.begin(), res_vec.end(), ResultComp);

    // Calculate all the rates with different threshold
    Mat rate;
    float accuracy = -1;
    bool is_set = false;
    for (size_t i = 0; i <= res_vec.size(); ++i)
    {
        Mat rate_row = Mat::zeros(1, 2, CV_32F);
        rate_row.at<float>(0, 0) = 1 - pos / pos_num;
        float temp = neg / win_num;
        rate_row.at<float>(0, 1) = temp;
        bool flag = true;

        if (i == 0 || i == res_vec.size()
                || fabs(res_vec[i].score - res_vec[i - 1].score) > 1e-5)
        {
            rate.push_back(rate_row);
        }
        else
        {
            flag = false;
        }

        if (i == res_vec.size())
        {
            break;
        }

        if (!is_set && temp <= fppw && flag)
        {
            accuracy = 1 - rate_row.at<float>(0, 0);
            *th = res_vec[i].score - 1e-5;
            is_set = true;
            if (file_name.empty())
            {
                return accuracy;
            }
        }

        if (res_vec[i].res)
        {
            --pos;
        }
        else
        {
            --neg;
        }
    }

    if (!file_name.empty())
    {
        SaveMat(file_name, rate);
    }

    if (!is_set)
    {
        accuracy = 1 - rate.at<float>(rate.rows - 1, 0);
        *th = res_vec.back().score + 1e-5;
    }
    return accuracy;
}
}  // namespace ghk
