/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include "base/Thread.hpp"
#include "base/Deque.hpp"

namespace FW
{
//------------------------------------------------------------------------
// Launch a batch of tasks and wait:
//   MulticoreLauncher().push(myTaskFunc, &myTaskData, 0, numTasks);
//
// Display progress indicator:
//   MulticoreLauncher().push(myTaskFunc, &myTaskData, 0, numTasks).popAll("Processing...");
//
// Grab results:
// {
//     MulticoreLauncher launcher;
//     launcher.push(...);
//     while (launcher.getNumTasks())
//         doSomething(launcher.pop().result);
// }
//
// Asynchronous operation:
// {
//     MulticoreLauncher launcher;
//     launcher.push(...);
//     while (launcher.getNumTasks())
//     {
//         spendSomeIdleTime();
//         while (launcher.getNumFinished())
//             doSomething(launcher.pop().result);
//     }
// }
//
// Create dependent tasks dynamically:
//
// void myTaskFunc(MulticoreLauncher::Task& task)
// {
//     ...
//     task.launcher->push(...);
//     ...
// }
//
// Per-thread temporary data:
//
// void myDeinitFunc(void* threadData)
// {
//     ...
// }
//
// void myTaskFunc(MulticoreLauncher::Task& task)
// {
//     void* threadData = Thread::getCurrent()->getUserData("myID");
//     if (!threadData)
//     {
//         threadData = ...;
//         Thread::getCurrent()->setUserData("myID", threadData, myDeinitFunc);
//     }
//     ...
// }
//------------------------------------------------------------------------

class MulticoreLauncher
{
public:
    struct Task;
    typedef void (*TaskFunc)(Task& task);

    struct Task
    {
        MulticoreLauncher*  launcher;
        TaskFunc            func;
        void*               data;
        int                 idx;
        void*               result; // Potentially written by TaskFunc.
    };

public:
                            MulticoreLauncher   (void);
                            ~MulticoreLauncher  (void);

    MulticoreLauncher&      push                (TaskFunc func, void* data, int firstIdx = 0, int numTasks = 1);
    Task                    pop                 (void);         // Blocks until at least one task has finished.

    int                     getNumTasks         (void) const;   // Tasks that have been pushed but not popped.
    int                     getNumFinished      (void) const;   // Tasks that can be popped without blocking.

    void                    popAll              (void)          { while (getNumTasks()) pop(); }
    void                    popAll              (const String& progressMessage);

    static int              getNumCores         (void);
    static void             setNumThreads       (int numThreads);

private:
    static void             applyNumThreads     (void);
    static void             threadFunc          (void* param);

private:
                            MulticoreLauncher   (const MulticoreLauncher&); // forbidden
    MulticoreLauncher&      operator=           (const MulticoreLauncher&); // forbidden

private:
    static Spinlock         s_lock;
    static S32              s_numInstances;
    static S32              s_desiredThreads;

    static Monitor*         s_monitor;
    static Deque<Task>      s_pending;
    static S32              s_numThreads;

    S32                     m_numTasks;
    Deque<Task>             m_finished;
};

//------------------------------------------------------------------------
}
