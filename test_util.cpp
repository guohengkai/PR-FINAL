/*************************************************************************
    > File Name: test_util.cpp
    > Author: Guo Hengkai
    > Description: Test function implementation
    > Created Time: Sat 16 May 2015 04:48:16 PM CST
 ************************************************************************/
#include "test_util.h"

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

    *tpr = static_cast<float>(tp_num) / cv::sum(class_sum.colRange(1, class_num))[0];
    *fpr = static_cast<float>(fp_num);
    if (is_open)
    {
        *fpr /= ((class_num - 1) * cv::sum(class_sum)[0]);
    }
    else
    {
        *fpr /= ((class_num - 2) * cv::sum(class_sum.colRange(1, class_num))[0]);
    }

    if (result_mat != nullptr)
    {
        *result_mat = Mat(result.mul(cv::repeat(class_sum, 1, class_num)));
    }
}
}  // namespace ghk
