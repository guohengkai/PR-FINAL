/*************************************************************************
    > File Name: test_class_util.cpp
    > Author: Guo Hengkai
    > Description: 
    > Created Time: Tue 19 May 2015 04:53:34 PM CST
 ************************************************************************/
#include "test_class_util.h"
#include "file_util.h"
#include "mat_util.h"
#include "sign_detector.h"
#include "test_util.h"

namespace ghk
{
void Generate2dRandomData(int train_cnt, int test_cnt, Mat *data)
{
    *data = Mat(train_cnt + test_cnt, 3, CV_32F);

    // Construct the points
    randn(data->rowRange(0, train_cnt / 2), 200, 50);
    randn(data->rowRange(train_cnt / 2, train_cnt + test_cnt / 2), 300, 50);
    randn(data->rowRange(train_cnt + test_cnt / 2, train_cnt + test_cnt), 200, 50);

    // Construct the responses
    data->col(2).setTo(1);
    data->col(2).rowRange(train_cnt / 2, train_cnt + test_cnt / 2).setTo(2);
}

void TestClassifier(Classifier *classifier, const string &tmp_dir)
{
    MakePath(tmp_dir);

    int train_cnt = 2000;
    int test_cnt = 200;
    Mat data;
    Generate2dRandomData(train_cnt, test_cnt, &data);
    vector<int> train_labels, test_labels;
    Mat2Vec(data.rowRange(0, train_cnt).col(2), &train_labels);
    Mat2Vec(data.rowRange(train_cnt, train_cnt + test_cnt).col(2),
            &test_labels);

    string model_name = tmp_dir + "/classier";

    if (!classifier->Train(data.rowRange(0, train_cnt).colRange(0, 2),
            train_labels))
    {
        printf("Fail to train the model.\n");
        return;
    }
    printf("Finish training.\n");

    if (!classifier->Save(model_name))
    {
        printf("Fail to save the model.\n");
        return;
    }
    printf("Finish saving.\n");
    if (!classifier->Load(model_name))
    {
        printf("Fail to load the model.\n");
        return;
    }
    printf("Model loaded.\n");

    vector<int> responses;
    if (!classifier->Predict(data.rowRange(train_cnt, train_cnt + test_cnt).
            colRange(0, 2), &responses))
    {
        printf("Fail to predict.\n");
        return;
    }
    printf("Finsih predicting.\n");
    
    float rate, fp;
    EvaluateClassify(test_labels, responses, 3, false, &rate, &fp);
    printf("Predict accuracy: %0.2f%%\n", rate * 100);

    // Visualization of classification result
    Mat img = Mat::zeros(500, 500, CV_8UC3);
    img.setTo(255);

    circle(img, Point(200, 200), 50, CV_RGB(255, 0, 0));
    circle(img, Point(300, 300), 50, CV_RGB(0, 0, 255));  
    
    for (int i = train_cnt; i < train_cnt + test_cnt; ++i)
    {
        int x = static_cast<int>(data.at<float>(i, 0));
        int y = static_cast<int>(data.at<float>(i, 1));
        int res = responses[i - train_cnt];
        
        circle(img, Point(x, y), 2, res == 1 ? CV_RGB(255, 0, 0) : CV_RGB(0, 0, 255), CV_FILLED);
        if (res != test_labels[i - train_cnt])
        {
            circle(img, Point(x, y), 4, CV_RGB(0, 255, 0));
        }
    }
    
    cv::imshow("", img);
    cv::waitKey(0);
}

void TestDataset(Dataset &dataset)
{
    const size_t idx = 72;
    Mat test_image;
    vector<Rect> rects;
    vector<int> labels;
    
    dataset.GetFullImage(idx, &test_image);
    dataset.GetDetectLabels(idx, &labels);
    dataset.GetDetectRects(idx, &rects);

    dataset.DrawRectAndLabel(rects, labels, &test_image);

    cv::imshow("", test_image);
    cv::waitKey();

    dataset.GetClassifyImage(true, idx, &test_image);
    cv::imshow("", test_image);
    cv::waitKey();

    dataset.GetClassifyImage(true, idx, &test_image, Size(200, 200));
    cv::imshow("", test_image);
    cv::waitKey();

    vector<Mat> neg_images;
    dataset.GetRandomNegImage(100, Size(100, 100), &neg_images, false);

    int n = 0;
    for (auto image: neg_images)
    {
        ++n;
        stringstream ss;
        ss << n;
        cv::imwrite("/home/ghk/Src/PR-HW/PR-FINAL/data/self-neg/"
                + ss.str() + ".jpg", image);
    }

    vector<Mat> images;
    dataset.GetDetectPosImage(Size(50, 50), &images, &labels, true);
    cout << "Load " << images.size() <<  " images for detection." << endl;
    n = 0;
    for (auto image: images)
    {
        ++n;
        if (n > 100) break;
        stringstream ss;
        ss << n;
        cv::imwrite("/home/ghk/Src/PR-HW/PR-FINAL/data/self-neg/"
                + ss.str() + ".jpg", image);
    }
}

void TestDetectorFunc()
{
    cout << "Test UpdateThreshold function..." << endl;
    vector<bool> results{
        true, false, true, false, true, false, false, false, true, false
    };
    vector<float> scores{
        0.5f, 0.6f, 0.7f, 0.5f, 0.1f, 0.2f, 1.0f, 0.2f, 1.0f, 0.3f
    };

    float th;
    cout << UpdateThreshold(results, scores, "./result/test_func",
            5, 100, &th, 0.01f) << endl;
    cout << th << endl;

    cout << "Test MergeRects function..." << endl;
    vector<Rect> rects{
        Rect(0, 0, 2, 2), Rect(0, 0, 2, 3), Rect(1, 1, 2, 2), Rect(0, 0, 3, 3),
        Rect(0, 0, 2, 2), Rect(0, 0, 2, 3), Rect(1, 1, 2, 2), Rect(0, 0, 3, 3),
        Rect(0, 0, 2, 2), Rect(0, 0, 2, 3), Rect(1, 1, 2, 2), Rect(0, 0, 3, 3),
        Rect(0, 0, 2, 2), Rect(0, 0, 2, 3), Rect(1, 1, 2, 2), Rect(0, 0, 3, 3),
        Rect(0, 0, 2, 2)
    };
    vector<int> labels{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1
    };
    vector<float> probs{
        0.8f, 0.7f, 0.9f, 0.6f, 1.0f,
        0.8f, 0.7f, 0.9f, 0.6f,
        0.8f, 0.7f, 0.9f, 0.6f,
        0.8f, 0.7f, 0.9f, 1.0f
    };
    vector<size_t> idx;
    MergeRects(rects, labels, probs, 0.667f);
    for (size_t i = 0; i < rects.size(); ++i)
    {
        cout << rects[i] << " " << labels[i] << " " << probs[i] << endl;
    }
}
}  // namespace ghk
