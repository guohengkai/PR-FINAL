/*************************************************************************
    > File Name: main.cpp
    > Author: Guo Hengkai
    > Description: Main program for traffic sign classification
    > Created Time: Fri 15 May 2015 04:27:05 PM CST
 ************************************************************************/
#include "common.h"
#include "dataset.h"
#include "test_class_util.h"
#include "knn_classifier.h"
#include "svm_classifier.h"
#include "sign_classifier.h"
#include "knn_sign_classifier.h"

using namespace ghk;

const string root_dir = "/home/ghk/Src/PR-HW/PR-FINAL";
const string model_dir = "/model";

void TestDataset()
{
    Dataset dataset(root_dir);
    TestDataset(dataset);
}

void TestClassifier()
{
    // KnnClassifier classifier(10);
    SvmClassifier classifier;
    TestClassifier(&classifier, root_dir + model_dir);
}

void TrainSignClassifier(SignClassifier *classifier)
{
    Dataset dataset(root_dir);
    classifier->Train(dataset);
}

int main(int argc, char **argv)
{
    // TestDataset();
    // TestClassifier();
    KnnSignClassifier classifier(false, 5);
    TrainSignClassifier(&classifier);

    return 0;
}
