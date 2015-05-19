/*************************************************************************
    > File Name: test_class_util.h
    > Author: Guo Hengkai
    > Description: Test class function definition
    > Created Time: Tue 19 May 2015 04:49:57 PM CST
 ************************************************************************/
#ifndef FINAL_TEST_CLASS_UTIL_
#define FINAL_TEST_CLASS_UTIL_

#include "common.h"
#include "classifier.h"

namespace ghk
{
void Generate2dRandomData(int train_cnt, int test_cnt, Mat *data);
void TestClassifier(Classifier *classifier, const string &tmp_dir);
}  // namespace ghk

#endif  // FINAL_TEST_CLASS_UTIL_
