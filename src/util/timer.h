/*************************************************************************
    > File Name: src/util/timer.h
    > Author: Guo Hengkai
    > Description: Timer class definition using gettimeofday for Linux
    > Created Time: Fri 29 May 2015 03:49:47 PM CST
 ************************************************************************/
#ifndef FINAL_TIMER_H_
#define FINAL_TIMER_H_

#include <sys/time.h>

namespace ghk
{
class Timer
{
public:
    Timer();
    void Start();
    float End();

    inline float time_use() { return time_use_; }

private:
    struct timeval start_time_;
    float time_use_;
};
}  // namespace ghk

#endif  // FINAL_TIMER_H_

