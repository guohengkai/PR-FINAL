/*************************************************************************
    > File Name: hog_sign_classifier.cpp
    > Author: Guo Hengkai
    > Description: HOG sign classifier class implementation using SVM
    > Created Time: Mon 25 May 2015 03:37:00 PM CST
 ************************************************************************/
#include "hog_sign_classifier.h"
#include "dataset.h"
#include "hog_extractor.h"
#include "svm_classifier.h"
#include "mat_util.h"
#include "math_util.h"
#include "file_util.h"
#include "test_util.h"
#include "timer.h"

namespace ghk
{
bool HogSignClassifier::Save(const string &model_name) const
{
    vector<float> param(1, img_size_);
    if (!SaveMat(model_name + "_para", Mat(), param))
    {
        printf("Fail to save the parameter.\n");
        return false;
    }
    if (!svm_classifier_.Save(model_name + "_svm"))
    {
        printf("Fail to save SVM.\n");
        return false;
    }
    return true;
}

bool HogSignClassifier::Load(const string &model_name)
{
    vector<float> param;
    Mat tmp;
    if (!LoadMat(model_name + "_para", &tmp, &param))
    {
        printf("Fail to load the parameter.\n");
        return false;
    }
    if (!svm_classifier_.Load(model_name + "_svm"))
    {
        printf("Fail to load SVM.\n");
        return false;
    }
    return true;
}

bool HogSignClassifier::Train(const Dataset &dataset,
        vector<Mat> &images, vector<int> &labels)
{
    // Find random negative sample
    srand(time(NULL));
    vector<Mat> neg_images;
    Size img_size(img_size_, img_size_);
    auto neg_num = static_cast<int>(images.size() * 1.5);
    printf("Randomly getting negative samples...\n");
    if (!dataset.GetRandomNegImage(neg_num, img_size, &neg_images))
    {
        printf("Fail to get negative samples.\n");
        return false;
    }
    images.insert(images.end(), neg_images.begin(), neg_images.end());
    vector<int> neg_labels(neg_images.size(), 0);
    labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());

    Timer timer;
    // Feature extraction
    Mat feats;
    printf("Extracting features...\n");
    timer.Start();
    hog_extractor_.Extract(images, &feats);
    float t1 = timer.Snapshot();
    printf("Time for extraction: %0.3fs\n", t1);

    // Train the SVM classifier
    printf("Training SVM classifier...\n");
    svm_classifier_.Train(feats, labels);
    float t2 = timer.Snapshot();
    printf("Time for training SVM: %0.3fs\n", t2 - t1);
    labels.erase(labels.begin() + labels.size() - neg_images.size(), labels.end());
    feats.resize(labels.size());

    // Test on training before mining
    vector<int> predict_labels;
    svm_classifier_.Predict(feats, &predict_labels);
    float rate, fp;
    EvaluateClassify(labels, predict_labels, CLASS_NUM, false, &rate, &fp);
    printf("Test on training rate before mining: %0.2f%%\n", rate * 100);

    // Mining hard negative sample
    timer.Start();
    printf("Mining hard negative samples...\n");
    Mat neg_feats;
    if (!MiningHardSample(dataset, neg_num, img_size, &neg_feats))
    {
        printf("Fail to retrain SVM.\n");
        return true;  // Because the original model can be used
    }
    float t3 = timer.Snapshot();
    printf("Time for mining: %0.3fs\n", t3);
    neg_labels.resize(neg_feats.rows, 0);
    labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());
    feats.push_back(neg_feats);

    // Retrain SVM classifier
    printf("Retraining SVM classifier...\n");
    svm_classifier_.Train(feats, labels);
    float t4 = timer.Snapshot();
    printf("Time for retrain SVM: %0.3fs\n", t4 - t3);
    labels.erase(labels.begin() + labels.size() - neg_feats.rows, labels.end());
    feats.resize(labels.size());
    printf("Total time: %0.3fs\n", t4 + t2);

    // Test on training again
    svm_classifier_.Predict(feats, &predict_labels);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, false, &rate, &fp);
    printf("Test on training rate after mining: %0.2f%%\n", rate * 100);

    return true;
}

bool HogSignClassifier::Train(const Dataset &dataset)
{
    // Get training data
    vector<Mat> images;
    vector<int> labels;
    size_t n = dataset.GetClassifyNum(true);
    Size img_size(img_size_, img_size_);
    printf("Preparing training data...\n");
    for (size_t i = 0; i < n; ++i)
    {
        Mat image;
        dataset.GetClassifyImage(true, i, &image, img_size);
        cv::cvtColor(image, image, CV_BGR2GRAY);
        images.push_back(image);
        labels.push_back(dataset.GetClassifyLabel(true, i));

        // Augment using rotation
        for (int j = 0; j < AUGMENT_TIMES; ++j)
        {
            Mat rot_img = image.clone();
            RotateImage(rot_img, Random(AUGMENT_ROTATE * 2 + 1)
                    - AUGMENT_ROTATE);
            images.push_back(rot_img);
            labels.push_back(labels[labels.size() - 1]);
        }
    }
    
    if (!Train(dataset, images, labels))
    {
        return false;
    }

    return true;
}

bool HogSignClassifier::Test(const Dataset &dataset)
{
    // Get test data
    vector<Mat> images;
    vector<int> labels;
    size_t n = dataset.GetClassifyNum(false);
    Size img_size(img_size_, img_size_);
    printf("Preparing testing data...\n");
    for (size_t i = 0; i < n; ++i)
    {
        Mat image;
        dataset.GetClassifyImage(false, i, &image, img_size);
        cv::cvtColor(image, image, CV_BGR2GRAY);
        images.push_back(image);
        labels.push_back(dataset.GetClassifyLabel(false, i));
    }

    // Test on test dataset
    vector<int> predict_labels;
    Predict(images, &predict_labels);
    float rate, fp;
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true, &rate, &fp);
    printf("Test rate: %0.2f%%\n", rate * 100);

    return true;
}

bool HogSignClassifier::Predict(const vector<Mat> &images,
        vector<int> *labels)
{
    vector<float> probs;
    return Predict(images, labels, &probs);
}

bool HogSignClassifier::Predict(const vector<Mat> &images,
        vector<int> *labels, vector<float> *probs)
{
    if (labels == nullptr)
    {
        return false;
    }

    Timer timer;
    // Feature extraction
    Mat feats;
    printf("Extracting features...\n");
    timer.Start();
    hog_extractor_.Extract(images, &feats);
    float t1 = timer.Snapshot();
    printf("Time for extration: %0.3fs\n", t1);

    // Prediction
    printf("Predicting with SVM...\n");
    svm_classifier_.Predict(feats, labels, probs);
    float t2 = timer.Snapshot();
    printf("Time for classification: %0.3fs\n", t2 - t1);
    printf("Prediction done! Total time for %d images: %0.3fs\n",
            feats.rows, t2);

    return true;
}

bool HogSignClassifier::FullTest(const Dataset &dataset,
        const string &dir)
{
    // Get training data
    vector<Mat> images;
    vector<int> labels;
    size_t n = dataset.GetClassifyNum(true);
    Size image_size(img_size_, img_size_);
    printf("Preparing training data...\n");
    Mat rotate_angle(n, AUGMENT_TIMES, CV_32F);
    for (size_t i = 0; i < n; ++i)
    {
        Mat image;
        dataset.GetClassifyImage(true, i, &image, image_size);
        cv::cvtColor(image, image, CV_BGR2GRAY);
        images.push_back(image);
        labels.push_back(dataset.GetClassifyLabel(true, i));

        // Augment using rotation
        for (int j = 0; j < AUGMENT_TIMES; ++j)
        {
            Mat rot_img = image.clone();
            rotate_angle.at<float>(i, j) = 
                Random(AUGMENT_ROTATE * 2 + 1) - AUGMENT_ROTATE;
            RotateImage(rot_img, rotate_angle.at<float>(i, j));
            images.push_back(rot_img);
            labels.push_back(labels[labels.size() - 1]);
        }
    }

    // Find random negative sample
    vector<Mat> neg_images;
    auto neg_num = n / (CLASS_NUM - 1) * 2;
    printf("Randomly getting negative samples...\n");
    if (!dataset.GetRandomNegImage(neg_num, image_size, &neg_images))
    {
        printf("Fail to get negative samples.\n");
        return false;
    }
    vector<int> neg_labels(neg_images.size(), 0);

    // Get test data
    vector<Mat> test_images;
    vector<int> test_labels;
    size_t test_n = dataset.GetClassifyNum(false);
    printf("Preparing testing data...\n");
    for (size_t i = 0; i < test_n; ++i)
    {
        Mat image;
        dataset.GetClassifyImage(false, i, &image, image_size);
        cv::cvtColor(image, image, CV_BGR2GRAY);
        test_images.push_back(image);
        test_labels.push_back(dataset.GetClassifyLabel(false, i));
    }

    // Test for different number of orientation and size of cell
    Mat hog_result(0, 4, CV_32F);
    printf("Training for different HOG parameters...\n");
    // for (int size = 10; size <= 150; size += 10)
    for (int size = 50; size <= 50; size += 10)
    {
        vector<Mat> size_train_img;
        vector<Mat> size_test_img;
        vector<Mat> size_neg_img;
        Size img_size(size, size);
        for (size_t i = 0; i < n; ++i)
        {
            Mat image;
            dataset.GetClassifyImage(true, i, &image, img_size);
            cv::cvtColor(image, image, CV_BGR2GRAY);
            size_train_img.push_back(image);

            // Augment using rotation
            for (int j = 0; j < AUGMENT_TIMES; ++j)
            {
                Mat rot_img = image.clone();
                RotateImage(rot_img, rotate_angle.at<float>(i, j));
                size_train_img.push_back(rot_img);
            }
        }
        for (size_t i = 0; i < test_n; ++i)
        {
            Mat image;
            dataset.GetClassifyImage(false, i, &image, img_size);
            cv::cvtColor(image, image, CV_BGR2GRAY);
            size_test_img.push_back(image);
        }
        for (auto img: neg_images)
        {
            Mat new_img;
            cv::resize(img, new_img, img_size);
            size_neg_img.push_back(new_img);
        }

        for (int num_orient = 3; num_orient <= 10; ++num_orient)
        {
            hog_extractor_.set_num_orient(num_orient);
            for (int cell_size = 4; cell_size <= 12; cell_size += 2)
            {
                printf("(%d %d %d), ", num_orient, cell_size, size);
                fflush(stdout);

                size_train_img.insert(size_train_img.end(),
                        size_neg_img.begin(), size_neg_img.end());
                neg_labels.resize(neg_images.size(), 0);
                labels.insert(labels.end(), neg_labels.begin(),
                        neg_labels.end());

                Mat feats;
                hog_extractor_.set_cell_size(cell_size);
                hog_extractor_.Extract(size_train_img, &feats);
                svm_classifier_.Train(feats, labels);
                labels.erase(labels.begin() + labels.size()
                        - neg_images.size(), labels.end());
                feats.resize(labels.size());

                Mat neg_feats;
                MiningHardSample(dataset, neg_num, img_size, &neg_feats);
                neg_labels.resize(neg_feats.rows, 0);
                labels.insert(labels.end(), neg_labels.begin(),
                        neg_labels.end());
                feats.push_back(neg_feats);
                svm_classifier_.Train(feats, labels);
                labels.erase(labels.begin() + labels.size() - neg_feats.rows,
                        labels.end());
                feats.resize(labels.size());

                vector<int> predict_labels;
                svm_classifier_.Predict(feats, &predict_labels);
                Mat result_row(1, 4, CV_32F);
                EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
                        &result_row.at<float>(0, 0),
                        &result_row.at<float>(0, 1));
                hog_extractor_.Extract(size_test_img, &feats);
                svm_classifier_.Predict(feats, &predict_labels);
                EvaluateClassify(test_labels, predict_labels, CLASS_NUM, true,
                        &result_row.at<float>(0, 2),
                        &result_row.at<float>(0, 3));
                hog_result.push_back(result_row);
                SaveMat(dir + "/hog_result_para", hog_result);
            }
        }
    }
    printf("Done! Saving...\n");
    SaveMat(dir + "/hog_result_para", hog_result);

    // Test for different penalty coefficients for SVM
    Mat svm_result_penal;
    printf("Training for different penalty coefficients...\n");
    images.insert(images.end(), neg_images.begin(), neg_images.end());
    Mat ori_feats;
    hog_extractor_.Extract(images, &ori_feats);
    for (float c = 0.0001; c < 1000000; c *= 10)
    {
        printf("%f, ", c);
        fflush(stdout);

        Mat feats = ori_feats.clone();
        svm_classifier_.set_c(c);
        labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());
        svm_classifier_.Train(feats, labels);
        labels.erase(labels.begin() + labels.size()
                - neg_images.size(), labels.end());
        feats.resize(labels.size());

        Mat neg_feats;
        MiningHardSample(dataset, neg_num, image_size, &neg_feats);
        neg_labels.resize(neg_feats.rows, 0);
        labels.insert(labels.end(), neg_labels.begin(),
                neg_labels.end());
        feats.push_back(neg_feats);
        svm_classifier_.Train(feats, labels);
        labels.erase(labels.begin() + labels.size() - neg_feats.rows,
                labels.end());
        feats.resize(labels.size());

        vector<int> predict_labels;
        Mat result_row(1, 4, CV_32F);
        svm_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
                &result_row.at<float>(0, 0),
                &result_row.at<float>(0, 1));
        hog_extractor_.Extract(test_images, &feats);
        svm_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(test_labels, predict_labels, CLASS_NUM, true,
                &result_row.at<float>(0, 2),
                &result_row.at<float>(0, 3));
        svm_result_penal.push_back(result_row);
        SaveMat(dir + "/svm_result_penal", svm_result_penal);
    }
    printf("\n");
    printf("Done! Saving...\n");
    SaveMat(dir + "/svm_result_penal", svm_result_penal);

    // Best results
    Timer timer;
    timer.Start();
    printf("Training for best result for HOG SVM...\n");
    images.insert(images.end(), neg_images.begin(), neg_images.end());
    labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());

    Mat feats;
    hog_extractor_.Extract(images, &feats);
    svm_classifier_.Train(feats, labels);
    labels.erase(labels.begin() + labels.size()
            - neg_images.size(), labels.end());
    feats.resize(labels.size());
    float t1 = timer.Snapshot();
    printf("Time for training without mining: %0.3fs\n", t1);
    vector<int> predict_labels;
    vector<float> best_result_before_train(2, 0);
    vector<float> best_result_before_test(2, 0);
    Mat mat_before_train, mat_before_test;
    svm_classifier_.Predict(feats, &predict_labels);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
            &best_result_before_train[0],
            &best_result_before_train[1],
            &mat_before_train);
    Mat test_feats;
    hog_extractor_.Extract(test_images, &test_feats);
    svm_classifier_.Predict(test_feats, &predict_labels);
    EvaluateClassify(test_labels, predict_labels, CLASS_NUM, true,
            &best_result_before_test[0],
            &best_result_before_test[1],
            &mat_before_test);
    float t2 = timer.Snapshot();
    printf("Time for testing without mining: %0.3fs\n", t2 - t1);

    Mat neg_feats;
    MiningHardSample(dataset, neg_num, image_size, &neg_feats);
    neg_labels.resize(neg_feats.rows, 0);
    labels.insert(labels.end(), neg_labels.begin(),
            neg_labels.end());
    feats.push_back(neg_feats);
    svm_classifier_.Train(feats, labels);
    labels.erase(labels.begin() + labels.size() - neg_feats.rows,
            labels.end());
    feats.resize(labels.size());
    float t3 = timer.Snapshot();
    printf("Time for training with mining: %0.3fs\n", t1 + (t3 - t2));

    Mat mat_after_train, mat_after_test;
    vector<float> best_result_after_train(2, 0);
    vector<float> best_result_after_test(2, 0);
    svm_classifier_.Predict(feats, &predict_labels);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
            &best_result_after_train[0],
            &best_result_after_train[1],
            &mat_after_train);
    hog_extractor_.Extract(test_images, &feats);
    svm_classifier_.Predict(feats, &predict_labels);
    EvaluateClassify(test_labels, predict_labels, CLASS_NUM, true,
            &best_result_after_test[0],
            &best_result_after_test[1],
            &mat_after_test);
    float t4 = timer.Snapshot();
    printf("Time for testing with mining: %0.3fs\n", (t2 - t1) + (t4 - t3));
    SaveMat(dir + "/best_hog_before_train",
                mat_before_train, best_result_before_train);
    SaveMat(dir + "/best_hog_before_test",
                mat_before_test, best_result_before_test);
    SaveMat(dir + "/best_hog_after_train",
                mat_after_train, best_result_after_train);
    SaveMat(dir + "/best_hog_after_test",
                mat_after_test, best_result_after_test);
    return true;
}

bool HogSignClassifier::MiningHardSample(const Dataset &dataset,
        size_t neg_num, Size image_size, Mat *neg_feats)
{
    if (neg_feats == nullptr)
    {
        return false;
    }

    vector<size_t> image_idxs;
    for (size_t i = 0; i < dataset.GetDetectNum(true); ++i)
    {
        image_idxs.push_back(i);
    }
    std::random_shuffle(image_idxs.begin(), image_idxs.end());

    neg_feats->resize(0);
    const int step = 10;
    for (auto idx: image_idxs)
    {
        Mat full_image;
        if (!dataset.GetDetectImage(true, idx, &full_image))
        {
            continue;
        }
        
        for (int x = 0; x < full_image.cols; x += step)
            for (int y = 0; y < full_image.rows; y += step)
                for (size_t size_idx = 0; size_idx < SIZE_LIST.size(); ++size_idx)
                {
                    int size = SIZE_LIST[size_idx];
                    if (x + size >= full_image.cols || y + size >= full_image.rows)
                    {
                        break;
                    }

                    Rect rect(x, y, size, size);
                    if (dataset.IsNegativeImage(true, idx, rect))
                    {
                        Mat image = full_image(rect).clone();
                        cv::resize(image, image, image_size);
                        cv::cvtColor(image, image, CV_BGR2GRAY);
                        
                        vector<Mat> image_vec(AUGMENT_TIMES + 1, image);
                        for (int i = 0; i < AUGMENT_TIMES; ++i)
                        {
                            RotateImage(image_vec[i + 1],
                                    Random(AUGMENT_ROTATE * 2 + 1)
                                    - AUGMENT_ROTATE);
                        }

                        Mat feat_row;
                        if (!hog_extractor_.Extract(image_vec, &feat_row))
                        {
                            return false;
                        }

                        // False positive
                        vector<int> labels;
                        if (!svm_classifier_.Predict(feat_row, &labels))
                        {
                            return false;
                        }

                        for (size_t i = 0; i < labels.size(); ++i)
                        {
                            if (labels[i] != 0)
                            {
                                neg_feats->push_back(feat_row.row(i));
                                if (neg_feats->rows >= static_cast<int>(neg_num))
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }

    }
    return true;
}
}  // namespace ghk
