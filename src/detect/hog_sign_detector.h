/*************************************************************************
    > File Name: src/detect/hog_sign_detector.h
    > Author: Guo Hengkai
    > Description: HOG sign detector class definition
    > Created Time: Wed 17 Jun 2015 11:03:31 AM CST
 ************************************************************************/
#ifndef FINAL_HOG_SIGN_DETECTOR_H_
#define FINAL_HOG_SIGN_DETECTOR_H_

#include "common.h"
#include "dataset.h"
#include "hog_sign_classifier.h"
#include "sign_detector.h"

namespace ghk
{
class HogSignDetector: public SignDetector
{
public:
    HogSignClassifier(int num_orient = 8, int cell_size = 8,
            float c = 125, int img_size = 100):
        classifier_(num_orient, cell_size, c, image_size),
        image_size_(Size(img_size, img_size)) {}
    virtual bool Save(const string &model_name) const;
    virtual bool Load(const string &model_name);
    
    virtual bool Train(const Dataset &dataset);
    virtual bool Test(const Dataset &dataset);
    virtual bool Detect(const vector<Mat> &images,
            vector<vector<Rect>> *rects, vector<int> *labels);

private:
    HogSignClassifier classifier_;
    Size image_size_;
}
}  // namespace ghk

#endif  // FINAL_HOG_SIGN_DETECTOR_H_

