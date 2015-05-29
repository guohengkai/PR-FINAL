/*************************************************************************
    > File Name: mat_util.h
    > Author: Guo Hengkai
    > Description: Matrix and vector function definition
    > Created Time: Tue 19 May 2015 03:07:35 PM CST
 ************************************************************************/
#ifndef FINAL_MAT_UTIL_H_
#define FINAL_MAT_UTIL_H_

#include "common.h"

namespace ghk
{
void Normalize(const Mat &normA, const Mat &normB,
        const Mat &feats, Mat *feats_norm);
void TrainNormalize(const Mat &feats, Mat *normA, Mat *normB);
template <typename T>
void Mat2Vec(const Mat &mat, vector<T> *vec);
template <typename T>
void Vec2Mat(const vector<T> &vec, Mat *mat);
bool Image2Vec(const vector<Mat> &images, Mat *image_vecs);
int GetUniqueClassNum(const vector<int> &labels);
void RotateImage(Mat &image, float angle_degree);
}

#endif  // FINAL_MAT_UTIL_H_
