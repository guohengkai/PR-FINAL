/*************************************************************************
    > File Name: mat_util.h
    > Author: Guo Hengkai
    > Description: Matrix function definition
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
void Mat2Vec(const Mat &mat, vector<int> *vec);
void Vec2Mat(const vector<int> &vec, Mat *mat);
}

#endif  // FINAL_MAT_UTIL_H_
