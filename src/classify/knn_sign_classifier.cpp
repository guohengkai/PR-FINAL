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
