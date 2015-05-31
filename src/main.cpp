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
#include "hog_sign_classifier.h"

using namespace ghk;

const string root_dir = "/home/ghk/Src/PR-HW/PR-FINAL";
const string model_dir = "/model";
const string result_dir = "/result";

void TestDataset()
{
    Dataset dataset(root_dir);
    TestDataset(dataset);
}

void TestClassifier()
{
    KnnClassifier classifier(10);
    // SvmClassifier classifier;
    TestClassifier(&classifier, root_dir + model_dir);
}

void TrainSignClassifier(SignClassifier *classifier, const string &model_name)
{
    Dataset dataset(root_dir);
    classifier->Train(dataset);
    classifier->Save(root_dir + model_dir + '/' + model_name);
    classifier->Load(root_dir + model_dir + '/' + model_name);
    classifier->Test(dataset);
}

void FullTest(SignClassifier *classifier)
{
    Dataset dataset(root_dir);
    classifier->FullTest(dataset, root_dir + result_dir);
}

int main(int argc, char **argv)
{
    // TestDataset();
    // TestClassifier();
    KnnSignClassifier classifier(true, 5, 100, 50, false);
    // HogSignClassifier classifier(125, 100);
    // TrainSignClassifier(&classifier, "hog_neg");
    FullTest(&classifier);
    return 0;
}
