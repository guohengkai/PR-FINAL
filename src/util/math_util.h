/*************************************************************************
    > File Name: src/util/math_util.h
    > Author: Guo Hengkai
    > Description: Math util function definitions
    > Created Time: Thu 28 May 2015 04:16:13 PM CST
 ************************************************************************/
#ifndef FINAL_MATH_UTIL_H_
#define FINAL_MATH_UTIL_H_

#include "common.h"

namespace ghk
{
template <typename T>
T Random(T n);
float Bool2Float(bool flag);
bool Float2Bool(float flag);
}  // namespace ghk

#endif  // FINAL_MATH_UTIL_H_

