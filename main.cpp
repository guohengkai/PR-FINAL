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

using namespace ghk;

const string root_dir = "/home/ghk/Src/PR-HW/PR-FINAL";
const string model_dir = "/model";

void TestClassifier()
{
    // KnnClassifier classifier(10);
    SvmClassifier classifier;
    TestClassifier(&classifier, root_dir + model_dir);
}

int main(int argc, char **argv)
{
    /*
    Dataset dataset(root_dir);

    const size_t idx = 72;
    Mat test_image;
    vector<Rect> rects;
    vector<int> labels;
    
    dataset.GetFullImage(idx, &test_image);
    dataset.GetDetectLabels(idx, &labels);
    dataset.GetDetectRects(idx, &rects);

    dataset.DrawRectAndLabel(rects, labels, &test_image);

    cv::imshow("", test_image);
    cv::waitKey(); */

    TestClassifier();

    return 0;
}
