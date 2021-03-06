/*************************************************************************
    > File Name: dataset.cpp
    > Author: Guo Hengkai
    > Description: Dataset class implementation to handle sign dataset
    > Created Time: Fri 15 May 2015 04:37:28 PM CST
 ************************************************************************/
#include "dataset.h"
#include <sstream>
#include "common.h"
#include "mat_util.h"
#include "math_util.h"
#include "file_util.h"

namespace ghk
{
Dataset::Dataset(const string &base_dir): base_dir_(base_dir)
{
    // Load class names
    label_name_.clear();
    label_name_map_.clear();
    string list_name = base_dir_ + CLASSIFY_DIR + "/" + LABEL_LIST_NAME;
    if (!LoadLabelNames(list_name))
    {
        return;
    }

    // Load positive classification data
    for (int i = 0; i < 2; ++i)
    {
        c_label_[i].clear();
        c_image_[i].clear();
    }
    for (int i = 1; i <= 10; ++i)
    {
        std::stringstream ss;
        ss << i;
        string data_dir = base_dir_ + CLASSIFY_DIR + "/" + ss.str() + "/";
        if (!LoadClassifyImages(data_dir, i, POS_C_TEST_NUM))
        {
            return;
        }
    }

    // Load negative classification data
    string data_dir = base_dir + CLASSIFY_DIR + NEG_DIR + "/";
    if (!LoadClassifyImages(data_dir, 0, -1))
    {
        return;
    }

    printf("Load %zu train images and %zu test images for classification.\n",
            GetClassifyNum(true), GetClassifyNum(false));

    // Load detection data
    d_name_list_.clear();
    d_rect_.clear();
    d_label_.clear();
    string detect_dir = base_dir_ + DETECT_DIR + "/";
    if (!LoadDetectLists(detect_dir))
    {
        return;
    }

    printf("Load %zu image names for detection.\n", GetFullImageNum());
}


int Dataset::GetClassifyLabel(bool is_train, size_t idx) const
{
    if (idx >= GetClassifyNum(is_train))
    {
        return -1;
    }
    return c_label_[INDEX(is_train)][idx];
}

bool Dataset::GetClassifyImage(bool is_train, size_t idx,
        Mat *image, Size img_size) const
{
    if (idx >= GetClassifyNum(is_train))
    {
        return false;
    }
    if (img_size.area() == 0)
    {
        *image = Mat(c_image_[INDEX(is_train)][idx]);
    }
    else
    {
        cv::resize(c_image_[INDEX(is_train)][idx], *image, img_size);
    }
    return true;
}


bool Dataset::GetDetectLabels(size_t idx, vector<int> *labels) const
{
    if (idx >= GetFullImageNum())
    {
        return false;
    }
    *labels = vector<int>(d_label_[idx]);
    return true;
}

bool Dataset::GetDetectRects(size_t idx, vector<Rect> *rects) const
{
    if (idx >= GetFullImageNum())
    {
        return false;
    }
    *rects = vector<Rect>(d_rect_[idx]);
    return true;
}

bool Dataset::GetFullImage(size_t idx, Mat *image) const
{
    if (idx >= GetFullImageNum())
    {
        return false;
    }
    *image = cv::imread(d_name_list_[idx], 1);
    if (image->empty())
    {
        printf("Fail to load %s.\n", d_name_list_[idx].c_str());
        return false;
    }
    return true;
}

bool Dataset::GetDetectLabels(bool is_train, size_t idx, vector<int> *labels) const
{
    return GetDetectLabels(GetDetectIdx(is_train, idx), labels);
}

bool Dataset::GetDetectRects(bool is_train, size_t idx, vector<Rect> *rects) const
{
    return GetDetectRects(GetDetectIdx(is_train, idx), rects);
}

bool Dataset::GetDetectImage(bool is_train, size_t idx, Mat *image) const
{
    return GetFullImage(GetDetectIdx(is_train, idx), image);
}

bool Dataset::GetRandomNegImage(size_t neg_num, Size image_size,
        vector<Mat> *images, bool is_augment) const
{
    if (images == nullptr)
    {
        return false;
    }

    int iter = 0;
    const int max_iter = 1000000;
    if (is_augment)
    {
        neg_num *= (AUGMENT_TIMES + 1);
    }
    images->clear();
    while (images->size() < neg_num && iter <= max_iter)
    {
        auto idx = Random(GetDetectNum(true));
        Mat image;
        GetDetectImage(true, idx, &image);

        auto size_idx = Random(SIZE_LIST.size());
        auto size = SIZE_LIST[size_idx];
        auto pos_x = Random(image.cols - size + 1);
        auto pos_y = Random(image.rows - size + 1);
        
        Rect rect(pos_x, pos_y, size, size);
        if (IsNegativeImage(true, idx, rect))
        {
            image = image(rect);
            cv::resize(image, image, image_size);
            cv::cvtColor(image, image, CV_BGR2GRAY);
            images->push_back(image);

            if (is_augment)
            {
                for (int i = 0; i < AUGMENT_TIMES; ++i)
                {
                    Mat rot_img = image.clone();
                    RotateImage(rot_img, Random(AUGMENT_ROTATE * 2 + 1)
                            - AUGMENT_ROTATE);
                    images->push_back(rot_img);
                }
            }
        }

        ++iter;
    }

    if (iter > max_iter)
    {
        printf("Only %zu negative images.\n", images->size());
        return false;
    }
    else
    {
        return true;
    }
}

bool Dataset::GetDetectPosImage(Size image_size,
        vector<Mat> *images, vector<int> *labels, bool is_augment) const
{
    if (images == nullptr)
    {
        return false;
    }

    images->clear();
    labels->clear();
    for (size_t i = 0; i < GetDetectNum(true); ++i)
    {
        if (i % 100 == 0)
        {
            printf("%zu(%zu), ", i, images->size());
            fflush(stdout);
        }
        Mat image;
        GetDetectImage(true, i, &image);

        for (auto size: SIZE_LIST)
        {
            for (int x = 0; x < image.cols; x += DETECT_STEP)
                for (int y = 0; y < image.rows; y += DETECT_STEP)
                {
                    if (x + size >= image.cols || y + size >= image.rows)
                    {
                        continue;
                    }
                    Rect rect(x, y, size, size);
                    int label;
                    if ((label = IsPositiveImage(true, i, rect)) > 0)
                    {
                        Mat temp = image(rect).clone();
                        Mat resize_temp;
                        cv::resize(temp, resize_temp, image_size);
                        cv::cvtColor(resize_temp, resize_temp, CV_BGR2GRAY);
                        images->push_back(resize_temp);
                        labels->push_back(label);
                        if (is_augment)
                        {
                            for (int i = 0; i < AUGMENT_TIMES; ++i)
                            {
                                Mat rot_img = temp.clone();
                                RotateImage(rot_img, Random(AUGMENT_ROTATE * 2 + 1)
                                        - AUGMENT_ROTATE);
                                cv::resize(rot_img, rot_img, image_size);
                                cv::cvtColor(rot_img, rot_img, CV_BGR2GRAY);
                                images->push_back(rot_img);
                                labels->push_back(label);
                            }
                        }
                    }
                }
        }
    }
    printf("\n");
    return true;
}

bool Dataset::IsNegativeImage(bool is_train, size_t idx, const Rect &rect) const
{
    idx = GetDetectIdx(is_train, idx);
    for (auto pos: d_rect_[idx])
    {
        if (static_cast<float>((pos & rect).area()) / (pos | rect).area()
                >= INTERSECT_UNION_RATE)
        {
            return false;
        }
    }
    return true;
}

int Dataset::IsPositiveImage(bool is_train, size_t idx, const Rect &rect) const
{
    idx = GetDetectIdx(is_train, idx);
    for (size_t i = 0; i < d_rect_[idx].size(); ++i)
    {
        auto pos = d_rect_[idx][i];
        if (static_cast<float>((pos & rect).area()) / (pos | rect).area()
                >= INTERSECT_UNION_RATE_POS)
        {
            return d_label_[idx][i];
        }
    }
    return 0;
}

bool Dataset::LoadLabelNames(const string &list_name)
{
    FILE *in_file;
    if ((in_file = fopen(list_name.c_str(), "r")) == nullptr)
    {
        printf("Fail to open %s.\n", list_name.c_str());
        return false;
    }
    
    char name[MAX_LINE + 1];
    int label = 0;
    while (fgets(name, MAX_LINE, in_file) != NULL)
    {
        if (name[strlen(name) - 1] == '\n')
        {
            name[strlen(name) - 1] = '\0';
        }
        label_name_map_[string(name)] = label++;
        label_name_.push_back(string(name));
    }

    fclose(in_file);
    return true;
}

bool Dataset::LoadClassifyImages(const string &data_dir,
        int label, int test_num)
{
    string list_name = data_dir + FILE_LIST_NAME;
    FILE *in_file;
    if ((in_file = fopen(list_name.c_str(), "r")) == nullptr)
    {
        printf("Fail to open %s.\n", list_name.c_str());
        return false;
    }

    // Test images
    char name[MAX_LINE + 1];
    for (int i = 0; i < test_num; ++i)
    {
        if (fgets(name, MAX_LINE, in_file) == NULL)
        {
            printf("No enough data.\n");
            return false;
        }

        ClipString(name);
        Mat image = cv::imread(data_dir + name, 1);
        if (image.empty())
        {
            printf("Fail to load %s.\n", (data_dir + name).c_str());
            cout << data_dir + name << endl;
            return false;
        }
        c_image_[1].push_back(image);
        c_label_[1].push_back(label);
    }

    // Train images or test images
    int idx = 0;
    if (test_num < 0)
    {
        idx = 1;
    }
    while (fgets(name, MAX_LINE, in_file) != NULL)
    {
        ClipString(name);
        Mat image = cv::imread(data_dir + name, 1);
        if (image.empty())
        {
            printf("Fail to load %s.\n", (data_dir + name).c_str());
            return false;
        }
        c_image_[idx].push_back(image);
        c_label_[idx].push_back(label);
    }
    fclose(in_file);
    return true;
}

bool Dataset::LoadDetectLists(const string &data_dir)
{
    string list_name = data_dir + DETECT_ANNOTE_NAME;
    FILE *in_file;
    if ((in_file = fopen(list_name.c_str(), "r")) == nullptr)
    {
        printf("Fail to open %s.\n", list_name.c_str());
        return false;
    }

    char image_name[DETECT_NAME_LENGTH + 1];
    char line[MAX_LINE + 1];
    int ch;

    int max_size = 0, min_size = 10000;
    while (1)
    {
        // Read image name
        for (int i = 0; i < DETECT_NAME_LENGTH; ++i)
        {
            ch = fgetc(in_file);
            if (ch == EOF)
            {
                break;
            }
            image_name[i] = ch;
        }
        if (ch == EOF)
        {
            break;
        }
        image_name[DETECT_NAME_LENGTH] = '\0';
        d_name_list_.push_back(data_dir + image_name);

        // Read annotations
        vector<Rect> rects;
        vector<int> labels;
        fgetc(in_file);  // ':'
        fgets(line, MAX_LINE, in_file);
        if (strlen(line) > 2)  // has rects
        {
            int n;
            sscanf(line, "%d", &n);
            char *token_ptr = strtok(line, ";");
            char label_name[MAX_LINE + 1];
            for (int i = 0; i < n; ++i)
            {
                token_ptr = strtok(NULL, ";");
                // printf("%s\n", token_ptr);
                int a, b, c, d;
                sscanf(token_ptr, "%[^,],%d,%d,%d,%d", label_name,
                        &a, &b, &c, &d);
                if (c - a < SIZE_LIST[0] || d - b < SIZE_LIST[0])
                {
                    continue;
                }
                // printf("%s\n", label_name);
                // printf("%d %d %d %d\n", a, b, c, d);
                rects.push_back(Rect(a, b, c - a, d - b));
                max_size = std::max(max_size, d - b);
                max_size = std::max(max_size, c - a);
                min_size = std::min(min_size, d - b);
                min_size = std::min(min_size, c - a);
                labels.push_back(label_name_map_[string(label_name)]);
            }
        }
        d_rect_.push_back(rects);
        d_label_.push_back(labels);
    }
    printf("Max/min size: %d, %d\n", max_size, min_size);
    fclose(in_file);
    return true;
}

void Dataset::DrawRectAndLabel(const vector<Rect> &rects, const vector<int> &labels,
        Mat *image) const
{
    for (size_t i = 0; i < rects.size(); ++i)
    {
        cv::rectangle(*image, rects[i], CV_RGB(0, 0, 0));
        // string name = label_name_[labels[i]];
        std::stringstream ss;
        ss << labels[i];
        string name = ss.str();
        cv::putText(*image, name, rects[i].tl() + cv::Point(1, 11),
                cv::FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0, 0, 0));
    }
}
}  // namespace ghk
