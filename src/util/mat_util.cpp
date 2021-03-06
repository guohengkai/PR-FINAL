/*************************************************************************
    > File Name: mat_util.cpp
    > Author: Guo Hengkai
    > Description: Matrix and vector function implementation
    > Created Time: Tue 19 May 2015 03:06:09 PM CST
 ************************************************************************/
#include "mat_util.h"

namespace ghk
{
void Normalize(const Mat &normA, const Mat &normB,
        const Mat &feats, Mat *feats_norm)
{
    int n = feats.rows;
    *feats_norm = (feats - cv::repeat(normA, n, 1))
        / cv::repeat(normB, n, 1);
}

void TrainNormalize(const Mat &feats, Mat *normA, Mat *normB)
{
    cv::reduce(feats, *normA, 0, CV_REDUCE_MIN);
    cv::reduce(feats, *normB, 0, CV_REDUCE_MAX);
    *normB -= *normA;
}

template <typename T>
void Mat2Vec(const Mat &mat, vector<T> *vec)
{
    vec->clear();
    for (int i = 0; i < mat.rows; ++i)
    {
        vec->push_back(static_cast<T>(mat.at<float>(i, 0)));
    }
}
template void Mat2Vec<int>(const Mat &mat, vector<int> *vec);
template void Mat2Vec<float>(const Mat &mat, vector<float> *vec);

template <typename T>
void Vec2Mat(const vector<T> &vec, Mat *mat)
{
    *mat = Mat::zeros(0, 1, CV_32F);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        mat->push_back(static_cast<float>(vec[i]));
    }
}
template void Vec2Mat<int>(const vector<int> &vec, Mat *mat);
template void Vec2Mat<float>(const vector<float> &vec, Mat *mat);

/*
void Vec2Mat(const vector<float> &vec, Mat *mat)
{
    *mat = Mat::zeros(0, 1, CV_32F);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        mat->push_back(vec[i]);
    }
}*/

bool Image2Vec(const vector<Mat> &images, Mat *image_vecs)
{
    if (images.empty() || image_vecs == nullptr)
    {
        return false;
    }

    // Suppose sizes of image are same
    size_t m = static_cast<size_t>(images[0].rows * images[0].cols);
    *image_vecs = Mat(0, m, CV_32F);
    for (auto image: images)
    {
        Mat image_gray(image);
        if (image_gray.channels() > 1)
        {
            cv::cvtColor(image, image_gray, CV_BGR2GRAY);
        }
        image_gray.convertTo(image_gray, CV_32F);
        if (image_gray.total() != m)
        {
            return false;
        }
        image_vecs->push_back(image_gray.reshape(0, 1));
    }
    
    return true;
}

int GetUniqueClassNum(const vector<int> &labels)
{
    set<int> label_set;
    for (auto label: labels)
    {
        label_set.insert(label);
    }
    return static_cast<int>(label_set.size());
}

void RotateImage(Mat &image, float angle_degree)
{
    int row = image.rows;
    int col = image.cols;
    Mat mat = cv::getRotationMatrix2D(cv::Point2f((col - 1.0) / 2,
                                                  (row - 1.0) / 2),
                                      angle_degree, 1.0);
    cv::warpAffine(image, image, mat, image.size());
}
}  // namespace ghk
