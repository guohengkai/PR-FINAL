/*************************************************************************
    > File Name: dataset.h
    > Author: Guo Hengkai
    > Description: Dataset class definition to handle sign dataset
    > Created Time: Fri 15 May 2015 03:43:53 PM CST
 ************************************************************************/
#ifndef FINAL_DATASET_H_
#define FINAL_DATASET_H_

#include "common.h"

#define INDEX(x) (x?0:1)

namespace ghk
{
const int CLASS_NUM = 11;  // 0 for open-set test
const int POS_C_TEST_NUM = 40;  // number of test images for each class

const string FILE_LIST_NAME = "filelist.txt";
const string DETECT_ANNOTE_NAME = "annotations.txt";
const string LABEL_LIST_NAME = "labelname.txt";
const string CLASSIFY_DIR = "/data";
const string DETECT_DIR = "/origin";
const string NEG_DIR = "/neg";

const int MAX_LINE = 255;
const int DETECT_NAME_LENGTH = 25;  // length of image name for detection

class Dataset
{
public:
    Dataset(const string &base_dir);

    int GetClassifyLabel(bool is_train, size_t idx) const;
    bool GetClassifyImage(bool is_train, size_t idx,
            Mat *image, Size img_size = Size()) const;

    bool GetDetectLabels(size_t idx, vector<int> *labels) const;
    bool GetDetectRects(size_t idx, vector<Rect> *rects) const;
    bool GetFullImage(size_t idx, Mat *image) const;

    void DrawRectAndLabel(const vector<Rect> &rects, const vector<int> &labels,
            Mat *image) const;

    inline size_t GetClassifyNum(bool is_train) const
    {
        return c_label_[INDEX(is_train)].size();
    }
    inline size_t GetFullImageNum() const
    {
        return d_rect_.size();
    }

private:
    bool LoadLabelNames(const string &list_name);
    bool LoadClassifyImages(const string &data_dir, int label, int test_num);
    bool LoadDetectLists(const string &data_dir);

    string base_dir_;

    vector<int> c_label_[2];
    vector<Mat> c_image_[2];
    map<string, int> label_name_map_;
    vector<string> label_name_;

    vector<vector<Rect>> d_rect_;
    vector<vector<int>> d_label_;
    vector<string> d_name_list_;
};

}  // namespace ghk

#endif  // FINAL_DATASET_H_
