/*************************************************************************
    > File Name: test_util.h
    > Author: Guo Hengkai
    > Description: Test function definition
    > Created Time: Sat 16 May 2015 04:35:29 PM CST
 ************************************************************************/
#ifndef FINAL_TEST_UTIL_H_
#define FINAL_TEST_UTIL_H_

#include "common.h"

namespace ghk
{
void EvaluateClassify(const vector<int> &ground_truth,
                      const vector<int> &predict_value,
                      int class_num, bool is_open,
                      float *tpr, float *fpr, Mat *result_mat = nullptr);
}  // namespace ghk
#endif  // FINAL_TEST_UTIL_H_
