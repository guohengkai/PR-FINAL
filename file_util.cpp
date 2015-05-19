/*************************************************************************
    > File Name: file_util.cpp
    > Author: Guo Hengkai
    > Description: File function implementation
    > Created Time: Tue 19 May 2015 02:36:26 PM CST
 ************************************************************************/
#include "file_util.h"
#include <sys/stat.h>

namespace ghk
{
bool LoadMat(const string& file_name, Mat* mat, vector<float>* param)
{
    if (mat == nullptr)
    {
        return false;
    }

    FILE *in_file;
    if ((in_file = fopen((file_name + FILE_EXT).c_str(), "r")) == nullptr)
    {
        return false;
    }

    int num_param;
    fscanf(in_file, "%d", &num_param);
    if (param != nullptr)
    {
        param->clear();
    }
    for (int i = 0; i < num_param; ++i)
    {
        float para;
        fscanf(in_file, "%f", &para);
        if (param != nullptr)
        {
            param->push_back(para);
        }
    }

    int row, col;
    fscanf(in_file, "%d %d", &row, &col);
    *mat = Mat(row, col, CV_32F);
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j) {
            fscanf(in_file, "%f", &(mat->at<float>(i, j)));
        }

    fclose(in_file);

    return true;
}

bool SaveMat(const string& file_name, const Mat& mat,
             const vector<float>& param)
{
    FILE *out_file;
    if ((out_file = fopen((file_name + FILE_EXT).c_str(), "w")) == nullptr)
    {
        return false;
    }

    fprintf(out_file, "%d\n", static_cast<int>(param.size()));
    for (size_t i = 0; i < param.size(); ++i)
    {
        fprintf(out_file, "%f ", param[i]);
    }
    fprintf(out_file, "\n");

    fprintf(out_file, "%d %d\n", mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            fprintf(out_file, "%f ", mat.at<float>(i, j));
        }
        fprintf(out_file, "\n");
    }

    fclose(out_file);

    return true;
}

bool SaveMat(const string& file_name, const Mat& mat)
{
    vector<float> empty_param;
    return SaveMat(file_name, mat, empty_param);
}

bool MakePath(const string path_name, mode_t mode)
{
    size_t pre = 0;
    size_t pos;
    string dir;
    int mdret;
    string path = path_name;

    if (path[path.size() - 1] != '/')
    {
        path += '/';
    }
    while ((pos = path.find_first_of('/', pre)) != string::npos)
    {
        dir = path.substr(0, pos++);
        pre = pos;
        if (dir.size() == 0) continue;
        if ((mdret = mkdir(dir.c_str(), mode)) && errno != EEXIST)
        {
            return false;
        }
    }

    return true;
}
}  // namespace ghk
