/*************************************************************************
    > File Name: file_util.h
    > Author: Guo Hengkai
    > Description: File function definition
    > Created Time: Tue 19 May 2015 02:32:01 PM CST
 ************************************************************************/
#ifndef FINAL_FILE_UTIL_H_
#define FINAL_FILE_UTIL_H_

#include "common.h"

namespace ghk
{
const string FILE_EXT = ".txt";
bool LoadMat(const string &file_name, Mat *mat,
        vector<float> *param = nullptr);
bool SaveMat(const string &file_name, const Mat &mat,
        const vector<float> &param);
bool SaveMat(const string& file_name, const Mat& mat);
bool MakePath(const string path_name, mode_t mode);
}  // namespace ghk

#endif  // FINAL_FILE_UTIL_H_
