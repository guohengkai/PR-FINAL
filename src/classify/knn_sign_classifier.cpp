/*************************************************************************
    > File Name: knn_sign_classifier.cpp
    > Author: Guo Hengkai
    > Description: KNN sign classifier class implemenation using eigenface or fisherface features
    > Created Time: Sat 23 May 2015 10:51:47 PM CST
 ************************************************************************/
#include "knn_sign_classifier.h"
#include "dataset.h"
#include "extractor.h"
#include "eigen_extractor.h"
#include "fisher_extractor.h"
#include "knn_classifier.h"
#include "mat_util.h"
#include "math_util.h"
#include "file_util.h"
#include "test_util.h"
#include "timer.h"

namespace ghk
{
KnnSignClassifier::KnnSignClassifier(bool use_fisher, int near_num,
            int eigen_feat_num, int img_size, bool use_threshold):
            use_fisher_(use_fisher), eigen_extractor_(eigen_feat_num),
            knn_classifier_(near_num), img_size_(img_size),
            threshold_(FLT_MAX), use_threshold_(use_threshold), neg_num_(2000)
{
   if (use_fisher_) 
   {
       extractor_ = &fisher_extractor_;
   }
   else
   {
       extractor_ = &eigen_extractor_;
   }
}

bool KnnSignClassifier::Save(const string &model_name) const
{
    vector<float> param{Bool2Float(use_fisher_),
                        static_cast<float>(img_size_),
                        threshold_,
                        Bool2Float(use_threshold_)};
    if (!SaveMat(model_name + "_para", Mat(), param))
    {
        printf("Fail to save the parameter.\n");
        return false;
    }

    if (!extractor_->Save(model_name + "_ext"))
    {
        printf("Fail to save the extractor.\n");
        return false;
    }
    if (!knn_classifier_.Save(model_name + "_c"))
    {
        printf("Fail to save the classifier.\n");
        return false;
    }
    return true;
}

bool KnnSignClassifier::Load(const string &model_name)
{
    vector<float> param;
    Mat tmp;
    if (!LoadMat(model_name + "_para", &tmp, &param))
    {
        printf("Fail to load the parameter.\n");
        return false;
    }
    use_fisher_ = Float2Bool(param[0]);
    img_size_ = param[1];
    threshold_ = param[2];
    use_threshold_ = Float2Bool(param[3]);

    if (use_fisher_)
    {
        extractor_ = &fisher_extractor_;
    }
    else
    {
        extractor_ = &eigen_extractor_;
    }

    if (!extractor_->Load(model_name + "_ext"))
    {
        printf("Fail to load the extractor.\n");
        return false;
    }
    if (!knn_classifier_.Load(model_name + "_c"))
    {
        printf("Fail to load the classifier.\n");
        return false;
    }
    return true;
}

bool KnnSignClassifier::Train(const Dataset &dataset)
{
    // Get training data
    vector<Mat> images;
    vector<Mat> neg_images;
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
    if (!use_threshold_)
    {
        // Find negative sample
        srand(time(NULL));
        printf("Randomly getting negative samples...\n");
        if (!dataset.GetRandomNegImage(neg_num_ / 5, img_size, &neg_images))
        {
            printf("Fail to get negative samples.\n");
            return false;
        }
        images.insert(images.end(), neg_images.begin(), neg_images.end());

        // Negative labels
        vector<int> neg_labels(neg_images.size(), 0);
        labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());
    }

    Timer timer;
    // Train the extractor
    printf("Training extractor...\n");
    timer.Start();
    extractor_->Train(images, labels);
    float t1 = timer.Snapshot();
    printf("Time for training extractor: %0.3fs\n", t1);

    // Feature extraction
    Mat feats;
    printf("Extracting features...\n");
    extractor_->Extract(images, &feats);
    float t2 = timer.Snapshot();
    printf("Time for extraction: %0.3fs\n", t2 - t1);

    // Train the KNN classifier
    printf("Training KNN classifier...\n");
    knn_classifier_.Train(feats, labels);
    float t3 = timer.Snapshot();
    printf("Time for training KNN: %0.3fs\n", t3 - t2);

    if (use_threshold_)
    {
        // Train the threshold
        printf("Training the threshold...\n");
        if (!TrainThreshold(dataset))
        {
            printf("Fail to train the threshold.\n");
            return false;
        }
    }
    else
    {
        labels.erase(labels.begin() + neg_images.size(), labels.end());
        feats.resize(labels.size());
    }
    float t4 = timer.Snapshot();
    printf("Training done! Total %0.3fs. Now testing...\n", t4);

    // Test on training
    vector<int> predict_labels;
    knn_classifier_.Predict(feats, &predict_labels);
    float rate, fp;
    EvaluateClassify(labels, predict_labels, CLASS_NUM, false, &rate, &fp);
    printf("Test on training rate: %0.2f%%\n", rate * 100);

    return true;
}

bool KnnSignClassifier::Test(const Dataset &dataset)
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

bool KnnSignClassifier::Predict(const vector<Mat> &images,
        vector<int> *labels)
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
    extractor_->Extract(images, &feats);
    float t1 = timer.Snapshot();
    printf("Time for extraction: %0.3fs\n", t1);

    // Prediction
    vector<float> distance;
    printf("Predicting with KNN...\n");
    knn_classifier_.Predict(feats, labels, &distance);
    float t2 = timer.Snapshot();
    printf("Time for classification: %0.3fs\n", t2 - t1);
    if (use_threshold_)
    {
        for (size_t i = 0; i < distance.size(); ++i)
        {
            if (distance[i] > threshold_)
            {
                (*labels)[i] = 0;
            }
        }
    }
    float t3 = timer.Snapshot();
    printf("Prediction done! Total time for %d images: %0.3fs\n",
            feats.rows, t3);

    return true;
}

bool KnnSignClassifier::FullTest(const Dataset &dataset, const string &dir)
{
    // Backup parameters
    bool use_fisher_backup = use_fisher_;
    bool use_threshold_backup = use_threshold_;
    int img_size_backup = img_size_;

    // Get training data
    vector<Mat> images;
    vector<Mat> neg_images;
    vector<int> labels;
    size_t n = dataset.GetClassifyNum(true);
    Size img_size(img_size_, img_size_);
    printf("Preparing training data...\n");
    Mat rotate_angle(n, AUGMENT_TIMES, CV_32F);
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
            rotate_angle.at<float>(i, j) = 
                Random(AUGMENT_ROTATE * 2 + 1) - AUGMENT_ROTATE;
            RotateImage(rot_img, rotate_angle.at<float>(i, j));
            images.push_back(rot_img);
            labels.push_back(labels[labels.size() - 1]);
        }
    }

    // Find negative sample
    srand(time(NULL));
    printf("Randomly getting negative samples...\n");
    if (!dataset.GetRandomNegImage(neg_num_ / 5, img_size, &neg_images))
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
        dataset.GetClassifyImage(false, i, &image, img_size);
        cv::cvtColor(image, image, CV_BGR2GRAY);
        test_images.push_back(image);
        test_labels.push_back(dataset.GetClassifyLabel(false, i));
    }

    /*
    // Test of component number for eigen feature (best: 180)
    set_use_fisher(false);
    printf("Training for different component number with eigen...\n");
    extractor_->Train(images, labels);
    Mat num_result_eigen(0, 4, CV_32F);
    const int step = 10;
    for (int i = step; i <= img_size_ * img_size_; i += step)
    {
        printf("%d, ", i);
        fflush(stdout);

        Mat feats;
        extractor_->set_feat_dim(i);
        extractor_->Extract(images, &feats);
        knn_classifier_.Train(feats, labels);

        Mat result_row(1, 4, CV_32F);
        vector<int> predict_labels;
        knn_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 0), &result_row.at<float>(0, 1));
        extractor_->Extract(test_images, &feats);
        knn_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(test_labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 2), &result_row.at<float>(0, 3));
        num_result_eigen.push_back(result_row);
    }
    printf("\n");
    printf("Done! Saving...\n");
    SaveMat(dir + "/num_result_eigen", num_result_eigen);
    
    // Test of nearest neighbour number for eigen feature (best: 5)
    set_use_fisher(false);
    printf("Training for different neighbour number with eigen...\n");
    extractor_->Train(images, labels);
    Mat feats;
    extractor_->Extract(images, &feats);
    Mat test_feats;
    extractor_->Extract(test_images, &test_feats);
    Mat nearest_result_eigen(0, 4, CV_32F);
    for (int i = 1; i < 102; i += 2)
    {
        printf("%d, ", i);
        fflush(stdout);

        knn_classifier_.set_near_num(i);
        knn_classifier_.Train(feats, labels);

        Mat result_row(1, 4, CV_32F);
        vector<int> predict_labels;
        knn_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 0), &result_row.at<float>(0, 1));
        knn_classifier_.Predict(test_feats, &predict_labels);
        EvaluateClassify(test_labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 2), &result_row.at<float>(0, 3));
        nearest_result_eigen.push_back(result_row);
    }
    printf("\n");
    printf("Done! Saving...\n");
    SaveMat(dir + "/nearest_result_eigen", nearest_result_eigen);

    // Test of nearest neighbour number for fisher feature (best: 7)
    set_use_fisher(true);
    printf("Training for different neighbour number with fisher...\n");
    extractor_->Train(images, labels);
    extractor_->Extract(images, &feats);
    extractor_->Extract(test_images, &test_feats);
    Mat nearest_result_fisher(0, 4, CV_32F);
    for (int i = 1; i < 102; i += 2)
    {
        printf("%d, ", i);
        fflush(stdout);

        knn_classifier_.set_near_num(i);
        knn_classifier_.Train(feats, labels);

        Mat result_row(1, 4, CV_32F);
        vector<int> predict_labels;
        knn_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 0), &result_row.at<float>(0, 1));
        knn_classifier_.Predict(test_feats, &predict_labels);
        EvaluateClassify(test_labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 2), &result_row.at<float>(0, 3));
        nearest_result_fisher.push_back(result_row);
    }
    printf("\n");
    printf("Done! Saving...\n");
    SaveMat(dir + "/nearest_result_fisher", nearest_result_fisher);

    // Test of image size
    printf("Training for different image size...");
    vector<Mat> size_train_img;
    vector<Mat> size_test_img;
    Mat size_result[2];
    for (int size = 10; size <= 150; size += 10)
    {
        printf("%d, ", size);
        fflush(stdout);

        size_train_img.clear();
        size_test_img.clear();
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

        Mat feats;
        Mat result_row(1, 4, CV_32F);
        vector<int> predict_labels;

        set_use_fisher(false);
        extractor_->Train(size_train_img, labels);
        extractor_->Extract(size_train_img, &feats);
        knn_classifier_.Train(feats, labels);

        knn_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 0), &result_row.at<float>(0, 1));
        extractor_->Extract(size_test_img, &feats);
        knn_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(test_labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 2), &result_row.at<float>(0, 3));
        size_result[0].push_back(result_row);

        set_use_fisher(true);
        extractor_->Train(size_train_img, labels);
        extractor_->Extract(size_train_img, &feats);
        knn_classifier_.Train(feats, labels);

        knn_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 0), &result_row.at<float>(0, 1));
        extractor_->Extract(size_test_img, &feats);
        knn_classifier_.Predict(feats, &predict_labels);
        EvaluateClassify(test_labels, predict_labels, CLASS_NUM, false,
                &result_row.at<float>(0, 2), &result_row.at<float>(0, 3));
        size_result[1].push_back(result_row);
    }
    printf("\n");
    printf("Done! Saving...\n");
    SaveMat(dir + "/size_result_eigen", size_result[0]);
    SaveMat(dir + "/size_result_fisher", size_result[1]);

    // Test of open-set methods for eigen feature using threshold
    Mat th_result_eigen(0, 4, CV_32F);
    printf("Training for different threshold with eigen...\n");
    set_use_fisher(false);
    extractor_->Train(images, labels);
    extractor_->Extract(images, &feats);
    knn_classifier_.Train(feats, labels);

    Mat neg_feats;
    extractor_->Extract(neg_images, &neg_feats);
    vector<float> neg_distance;
    vector<int> neg_train_labels;
    knn_classifier_.Predict(neg_feats, &neg_train_labels, &neg_distance);
    float mean = 0, deviation = 0;
    for (auto dis: neg_distance)
    {
        mean += dis;
        deviation += dis * dis;
    }
    mean /= neg_distance.size();
    deviation = sqrt(deviation / neg_distance.size() - mean * mean);

    extractor_->Extract(test_images, &test_feats);

    Mat result_row(1, 4, CV_32F);
    vector<int> predict_labels;
    vector<float> train_dis;
    knn_classifier_.Predict(feats, &predict_labels, &train_dis);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
            &result_row.at<float>(0, 0), &result_row.at<float>(0, 1));
    vector<int> test_predict_labels;
    vector<float> test_dis;
    knn_classifier_.Predict(test_feats, &test_predict_labels, &test_dis);
    EvaluateClassify(test_labels, test_predict_labels, CLASS_NUM, true,
            &result_row.at<float>(0, 2), &result_row.at<float>(0, 3));
    th_result_eigen.push_back(result_row);

    for (float rate = 0; rate <= 5; rate += 0.5)
    {
        printf("%0.1f, ", rate);
        fflush(stdout);

        threshold_ = mean + rate * deviation;
        vector<int> temp_labels;
        for (size_t i = 0; i < train_dis.size(); ++i)
        {
            if (train_dis[i] > threshold_)
            {
                temp_labels.push_back(0);
            }
            else
            {
                temp_labels.push_back(predict_labels[i]);
            }
        }
        EvaluateClassify(labels, temp_labels, CLASS_NUM, true,
                &result_row.at<float>(0, 0), &result_row.at<float>(0, 1));
        temp_labels.clear();
        for (size_t i = 0; i < test_dis.size(); ++i)
        {
            if (test_dis[i] > threshold_)
            {
                temp_labels.push_back(0);
            }
            else
            {
                temp_labels.push_back(test_predict_labels[i]);
            }
        }
        EvaluateClassify(test_labels, temp_labels, CLASS_NUM, true,
                &result_row.at<float>(0, 2), &result_row.at<float>(0, 3));
        th_result_eigen.push_back(result_row);
    }
    printf("\n");
    printf("Done! Saving...\n");
    SaveMat(dir + "/th_result_eigen", th_result_eigen);

    // Test of open-set methods for fisher feature using threshold
    Mat th_result_fisher(0, 4, CV_32F);
    printf("Training for different threshold with fisher...\n");
    set_use_fisher(true);
    extractor_->Train(images, labels);
    extractor_->Extract(images, &feats);
    knn_classifier_.Train(feats, labels);

    extractor_->Extract(neg_images, &neg_feats);
    knn_classifier_.Predict(neg_feats, &neg_train_labels, &neg_distance);
    mean = 0;
    deviation = 0;
    for (auto dis: neg_distance)
    {
        mean += dis;
        deviation += dis * dis;
    }
    mean /= neg_distance.size();
    deviation = sqrt(deviation / neg_distance.size() - mean * mean);

    extractor_->Extract(test_images, &test_feats);

    knn_classifier_.Predict(feats, &predict_labels, &train_dis);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
            &result_row.at<float>(0, 0), &result_row.at<float>(0, 1));
    knn_classifier_.Predict(test_feats, &test_predict_labels, &test_dis);
    EvaluateClassify(test_labels, test_predict_labels, CLASS_NUM, true,
            &result_row.at<float>(0, 2), &result_row.at<float>(0, 3));
    th_result_fisher.push_back(result_row);

    for (float rate = 0; rate <= 5; rate += 0.5)
    {
        printf("%0.1f, ", rate);
        fflush(stdout);

        threshold_ = mean + rate * deviation;
        vector<int> temp_labels;
        for (size_t i = 0; i < train_dis.size(); ++i)
        {
            if (train_dis[i] > threshold_)
            {
                temp_labels.push_back(0);
            }
            else
            {
                temp_labels.push_back(predict_labels[i]);
            }
        }
        EvaluateClassify(labels, temp_labels, CLASS_NUM, true,
                &result_row.at<float>(0, 0), &result_row.at<float>(0, 1));
        temp_labels.clear();
        for (size_t i = 0; i < test_dis.size(); ++i)
        {
            if (test_dis[i] > threshold_)
            {
                temp_labels.push_back(0);
            }
            else
            {
                temp_labels.push_back(test_predict_labels[i]);
            }
        }
        EvaluateClassify(test_labels, temp_labels, CLASS_NUM, true,
                &result_row.at<float>(0, 2), &result_row.at<float>(0, 3));
        th_result_fisher.push_back(result_row);
    }
    printf("\n");
    printf("Done! Saving...\n");
    SaveMat(dir + "/th_result_fisher", th_result_fisher);

    // Test of open-set methods for eigen feature using negative class
    images.insert(images.end(), neg_images.begin(), neg_images.end());
    labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());
    printf("Training for negative class with eigen...\n");
    set_use_fisher(false);
    extractor_->Train(images, labels);
    extractor_->Extract(images, &feats);
    knn_classifier_.Train(feats, labels);
    labels.erase(labels.begin() + neg_images.size(), labels.end());
    feats.resize(labels.size());
    extractor_->Extract(test_images, &test_feats);

    Mat neg_result_eigen(1, 4, CV_32F);
    knn_classifier_.Predict(feats, &predict_labels);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
            &neg_result_eigen.at<float>(0, 0),
            &neg_result_eigen.at<float>(0, 1));
    knn_classifier_.Predict(test_feats, &test_predict_labels);
    EvaluateClassify(test_labels, test_predict_labels, CLASS_NUM, true,
            &neg_result_eigen.at<float>(0, 2),
            &neg_result_eigen.at<float>(0, 3));
    printf("Done! Saving...\n");
    SaveMat(dir + "/neg_result_eigen", neg_result_eigen);

    // Test of open-set methods for fisher feature using negative class
    images.insert(images.end(), neg_images.begin(), neg_images.end());
    labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());
    printf("Training for negative class with fisher...\n");
    set_use_fisher(true);
    extractor_->Train(images, labels);
    extractor_->Extract(images, &feats);
    knn_classifier_.Train(feats, labels);
    labels.erase(labels.begin() + neg_images.size(), labels.end());
    feats.resize(labels.size());
    extractor_->Extract(test_images, &test_feats);

    Mat neg_result_fisher(1, 4, CV_32F);
    knn_classifier_.Predict(feats, &predict_labels);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
            &neg_result_fisher.at<float>(0, 0),
            &neg_result_fisher.at<float>(0, 1));
    knn_classifier_.Predict(test_feats, &test_predict_labels);
    EvaluateClassify(test_labels, test_predict_labels, CLASS_NUM, true,
            &neg_result_fisher.at<float>(0, 2),
            &neg_result_fisher.at<float>(0, 3));
    printf("Done! Saving...\n");
    SaveMat(dir + "/neg_result_fisher", neg_result_fisher);
    */

    Mat feats, neg_feats, test_feats;
    vector<int> neg_train_labels, predict_labels, test_predict_labels;
    vector<float> neg_distance;
    float mean, deviation;
    vector<float> train_dis;
    vector<float> test_dis;
    // Best results for eigen feature
    Timer timer;
    timer.Start();
    printf("Training for best result with eigen...\n");
    set_use_fisher(false);
    knn_classifier_.set_near_num(5);
    extractor_->Train(images, labels);
    extractor_->Extract(images, &feats);
    knn_classifier_.Train(feats, labels);

    extractor_->Extract(neg_images, &neg_feats);
    knn_classifier_.Predict(neg_feats, &neg_train_labels, &neg_distance);
    mean = 0;
    deviation = 0;
    for (auto dis: neg_distance)
    {
        mean += dis;
        deviation += dis * dis;
    }
    mean /= neg_distance.size();
    deviation = sqrt(deviation / neg_distance.size() - mean * mean);
    threshold_ = mean + 2.5 * deviation;
    float t1 = timer.Snapshot();
    printf("Time for training PCA: %0.3fs\n", t1);

    extractor_->Extract(test_images, &test_feats);
    knn_classifier_.Predict(feats, &predict_labels, &train_dis);
    for (size_t i = 0; i < train_dis.size(); ++i)
    {
        if (train_dis[i] > threshold_)
        {
            predict_labels[i] = 0;
        }
    }
    Mat result_mat_train_eigen;
    vector<float> best_result_train_eigen(2, 0);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
            &best_result_train_eigen[0],
            &best_result_train_eigen[1],
            &result_mat_train_eigen);
    knn_classifier_.Predict(test_feats, &test_predict_labels, &test_dis);
    for (size_t i = 0; i < test_dis.size(); ++i)
    {
        if (test_dis[i] > threshold_)
        {
            test_predict_labels[i] = 0;
        }
    }
    Mat result_mat_test_eigen;
    vector<float> best_result_test_eigen(2, 0);
    EvaluateClassify(test_labels, test_predict_labels, CLASS_NUM, true,
            &best_result_test_eigen[0],
            &best_result_test_eigen[1],
            &result_mat_test_eigen);
    float t2 = timer.Snapshot();
    printf("Time for testing PCA: %0.3fs\n", t2 - t1);
    printf("Done! Saving...\n");
    SaveMat(dir + "/best_result_train_eigen",
            result_mat_train_eigen, best_result_train_eigen);
    SaveMat(dir + "/best_result_test_eigen",
            result_mat_test_eigen, best_result_test_eigen);
    
    // Best results for fisher feature
    t1 = timer.Snapshot();
    printf("Training for best result with fisher...\n");
    set_use_fisher(true);
    knn_classifier_.set_near_num(7);
    extractor_->Train(images, labels);
    extractor_->Extract(images, &feats);
    knn_classifier_.Train(feats, labels);
    extractor_->Extract(test_images, &test_feats);

    extractor_->Extract(neg_images, &neg_feats);
    knn_classifier_.Predict(neg_feats, &neg_train_labels, &neg_distance);
    mean = 0;
    deviation = 0;
    for (auto dis: neg_distance)
    {
        mean += dis;
        deviation += dis * dis;
    }
    mean /= neg_distance.size();
    deviation = sqrt(deviation / neg_distance.size() - mean * mean);
    threshold_ = mean + 2.5 * deviation;
    t2 = timer.Snapshot();
    printf("Time for training Fisher: %0.3fs\n", t2 - t1);

    knn_classifier_.Predict(feats, &predict_labels, &train_dis);
    for (size_t i = 0; i < train_dis.size(); ++i)
    {
        if (train_dis[i] > threshold_)
        {
            predict_labels[i] = 0;
        }
    }
    Mat result_mat_train_fisher;
    vector<float> best_result_train_fisher(2, 0);
    EvaluateClassify(labels, predict_labels, CLASS_NUM, true,
            &best_result_train_fisher[0],
            &best_result_train_fisher[1],
            &result_mat_train_fisher);
    knn_classifier_.Predict(test_feats, &test_predict_labels, &test_dis);
    for (size_t i = 0; i < test_dis.size(); ++i)
    {
        if (test_dis[i] > threshold_)
        {
            test_predict_labels[i] = 0;
        }
    }
    Mat result_mat_test_fisher;
    vector<float> best_result_test_fisher(2, 0);
    EvaluateClassify(test_labels, test_predict_labels, CLASS_NUM, true,
            &best_result_test_fisher[0],
            &best_result_test_fisher[1],
            &result_mat_test_fisher);
    t1 = timer.Snapshot();
    printf("Time for testing Fisher: %0.3fs\n", t1 - t2);
    printf("Done! Saving...\n");
    SaveMat(dir + "/best_result_train_fisher",
            result_mat_train_fisher, best_result_train_fisher);
    SaveMat(dir + "/best_result_test_fisher",
            result_mat_test_fisher, best_result_test_fisher);

    // Restore parameters
    set_use_fisher(use_fisher_backup);
    use_threshold_ = use_threshold_backup;
    img_size_ = img_size_backup;

    return true;
}

void KnnSignClassifier::set_use_fisher(bool use_fisher)
{
    if (use_fisher != use_fisher_)
    {
        use_fisher_ = use_fisher;
        if (use_fisher_)
        {
            extractor_ = &fisher_extractor_;
        }
        else
        {
            extractor_ = &eigen_extractor_;
        }
    }
}

bool KnnSignClassifier::TrainThreshold(const Dataset &dataset)
{
    srand(time(NULL));
    vector<Mat> neg_images;
    Size img_size(img_size_, img_size_);
    printf("Randomly getting negative samples...\n");
    if (!dataset.GetRandomNegImage(neg_num_, img_size, &neg_images))
    {
        printf("Fail to get negative samples.\n");
        return false;
    }

    Mat feats;
    printf("Extracting features...\n");
    extractor_->Extract(neg_images, &feats);

    vector<float> distance;
    vector<int> labels;
    printf("Getting distance with KNN...\n");
    knn_classifier_.Predict(feats, &labels, &distance);

    float mean = 0, deviation = 0;
    for (auto dis: distance)
    {
        mean += dis;
        deviation += dis * dis;
    }
    mean /= distance.size();
    deviation = sqrt(deviation / distance.size() - mean * mean);
    printf("Mean: %0.4f, Std: %0.4f\n", mean, deviation);
    threshold_ = mean + 2 * deviation;
    
    return true;
}
}  // namespace ghk
