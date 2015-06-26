/*************************************************************************
    > File Name: main.cpp
    > Author: Guo Hengkai
    > Description: Main program for traffic sign classification
    > Created Time: Fri 15 May 2015 04:27:05 PM CST
 ************************************************************************/
#include "common.h"
#include "dataset.h"
#include "test_class_util.h"
#include "forest_classifier.h"
#include "knn_classifier.h"
#include "svm_classifier.h"
#include "sign_classifier.h"
#include "knn_sign_classifier.h"
#include "hog_sign_classifier.h"
#include "hog_sign_detector.h"

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
    // KnnClassifier classifier(10);
    // SvmClassifier classifier;
    ForestClassifier classifier(10, 10);
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

void TrainDetector(const string &model_name)
{
    Dataset dataset(root_dir);
    HogSignDetector detector(4, 4, 100, 30, false);
    detector.Train(dataset);
    detector.Save(root_dir + model_dir + '/' + model_name);
    
    // detector.Load(root_dir + model_dir + '/' + model_name);
    /*
    vector<Mat> image(1);
    size_t idx = 100;
    vector<int> labels;
    vector<Rect> rects;
    // dataset.GetDetectImage(false, idx, &image[0]);
    // dataset.GetDetectRects(false, idx, &rects);
    // dataset.GetDetectLabels(false, idx, &labels);
    dataset.GetDetectImage(true, idx, &image[0]);
    dataset.GetDetectRects(true, idx, &rects);
    dataset.GetDetectLabels(true, idx, &labels);

    vector<vector<Rect>> res_rects;
    vector<vector<int>> res_labels;
    vector<vector<float>> probs;
    int win_num;
    detector.Detect(image, &res_rects, &res_labels, &probs, &win_num);
    cout << "Total number of windows: " << win_num << endl;
    dataset.DrawRectAndLabel(res_rects[0], res_labels[0], &image[0]);
    cv::imshow("", image[0]);
    cv::waitKey();*/

    detector.Test(dataset);
}

int main(int argc, char **argv)
{
    // TestDataset();
    // TestClassifier();
    // KnnSignClassifier classifier(true, 5, 180, 20, false);
    // HogSignClassifier classifier(4, 4, 100, 50);
    // TrainSignClassifier(&classifier, "hog_neg");
    // FullTest(&classifier);

    TrainDetector("hog_detector_mining_rf2");
    // TestDetectorFunc();
    return 0;
}
