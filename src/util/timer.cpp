/*************************************************************************
    > File Name: src/util/timer.cpp
    > Author: Guo Hengkai
    > Description: Timer class implementation using gettimeofday for Linux
    > Created Time: Fri 29 May 2015 03:54:39 PM CST
 ************************************************************************/
#include "timer.h"
#include <cstdlib>
#include <sys/time.h>

namespace ghk
{
Timer::Timer()
{
}

void Timer::Start()
{
    gettimeofday(&start_time_, NULL);
    time_use_ = 0;
}

float Timer::End()
{
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    time_use_ = end_time.tv_sec - start_time_.tv_sec
        + (end_time.tv_usec - start_time_.tv_usec) / 1000000.0f;
    return time_use();
}
}  // namespace ghk
