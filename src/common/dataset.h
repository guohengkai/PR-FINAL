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
const size_t DETECT_TEST_NUM = 500;  // number of test images for detection
const vector<int> SIZE_LIST = vector<int>{
    20, 30, 50, 70, 90, 100, 150, 200, 250, 300};  // box size
const int AUGMENT_TIMES = 4;
const int AUGMENT_ROTATE = 30;  // Max rotation degree

const string FILE_LIST_NAME = "filelist.txt";
const string DETECT_ANNOTE_NAME = "annotations.txt";
const string LABEL_LIST_NAME = "labelname.txt";
const string CLASSIFY_DIR = "/data";
const string DETECT_DIR = "/origin";
const string NEG_DIR = "/neg";

const int MAX_LINE = 255;
const int DETECT_NAME_LENGTH = 25;  // length of image name for detection
const float INTERSECT_UNION_RATE = 0.5;
const float INTERSECT_UNION_RATE_POS = 0.7;
const int DETECT_STEP = 10;

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
    bool GetDetectLabels(bool is_train, size_t idx, vector<int> *labels) const;
    bool GetDetectRects(bool is_train, size_t idx, vector<Rect> *rects) const;
    bool GetDetectImage(bool is_train, size_t idx, Mat *image) const;

    bool GetRandomNegImage(size_t neg_num, Size image_size,
            vector<Mat> *images, bool is_augment = true) const;
    bool GetDetectPosImage(Size image_size, vector<Mat> *images,
            vector<int> *labels, bool is_augment = true) const;
    bool IsNegativeImage(bool is_train, size_t idx, const Rect &rect) const;
    int IsPositiveImage(bool is_train, size_t idx, const Rect &rect) const;

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
    inline size_t GetDetectNum(bool is_train) const
    {
        return is_train ? d_rect_.size() - DETECT_TEST_NUM : DETECT_TEST_NUM;
    }

private:
    bool LoadLabelNames(const string &list_name);
    bool LoadClassifyImages(const string &data_dir, int label, int test_num);
    bool LoadDetectLists(const string &data_dir);
    inline size_t GetDetectIdx(bool is_train, size_t idx) const
    {
        return is_train ? idx : idx + GetDetectNum(true);
    }

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
