/*************************************************************************
    > File Name: src/util/math_util.cpp
    > Author: Guo Hengkai
    > Description: Math util function implementations
    > Created Time: Thu 28 May 2015 04:18:55 PM CST
 ************************************************************************/
#include "math_util.h"

namespace ghk
{
template <typename T>
T Random(T n)
{
    int result = rand();
    return result % n;
}
template int Random<int>(int n);
template size_t Random<size_t>(size_t n);

float Bool2Float(bool flag)
{
    if (flag)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

bool Float2Bool(float flag)
{
    return (flag > 0);
}
}  // namespace ghk
