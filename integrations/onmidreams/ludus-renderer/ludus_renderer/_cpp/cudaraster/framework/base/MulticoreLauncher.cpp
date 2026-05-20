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

#include "base/MulticoreLauncher.hpp"
#include "base/Timer.hpp"

using namespace FW;

//------------------------------------------------------------------------

Spinlock                        MulticoreLauncher::s_lock;
S32                             MulticoreLauncher::s_numInstances   = 0;
S32                             MulticoreLauncher::s_desiredThreads = -1;

Monitor*                        MulticoreLauncher::s_monitor        = NULL;
Deque<MulticoreLauncher::Task>  MulticoreLauncher::s_pending;
S32                             MulticoreLauncher::s_numThreads     = 0;

//------------------------------------------------------------------------

MulticoreLauncher::MulticoreLauncher(void)
:   m_numTasks  (0)
{
    s_lock.enter();

    // First time => query the number of cores.

    if (s_desiredThreads == -1)
        s_desiredThreads = getNumCores();

    // First instance => static init.

    if (s_numInstances++ == 0)
        s_monitor = new Monitor;

    s_lock.leave();
}

//------------------------------------------------------------------------

MulticoreLauncher::~MulticoreLauncher(void)
{
    popAll();
    s_lock.enter();

    // Last instance => static deinit.

    if (--s_numInstances == 0)
    {
        // Kill threads.

        int old = s_desiredThreads;
        s_desiredThreads = 0;
        s_monitor->enter();
        applyNumThreads();
        s_monitor->leave();
        s_desiredThreads = old;

        // Deinit static members.

        delete s_monitor;
        s_monitor = NULL;
        s_pending.reset();
    }

    s_lock.leave();
}

//------------------------------------------------------------------------

MulticoreLauncher& MulticoreLauncher::push(TaskFunc func, void* data, int firstIdx, int numTasks)
{
    FW_ASSERT(func != NULL);
    FW_ASSERT(numTasks >= 0);

    if (numTasks <= 0)
        return *this;

    s_monitor->enter();

    for (int i = 0; i < numTasks; i++)
    {
        Task& task = s_pending.addLast();
        task.launcher = this;
        task.func     = func;
        task.data     = data;
        task.idx      = firstIdx + i;
        task.result   = NULL;
    }
    m_numTasks += numTasks;

    applyNumThreads();
    s_monitor->notifyAll();
    s_monitor->leave();
    return *this;
}

//------------------------------------------------------------------------

MulticoreLauncher::Task MulticoreLauncher::pop(void)
{
    FW_ASSERT(getNumTasks());
    s_monitor->enter();

    // Wait for a task to finish.

    while (!getNumFinished())
        s_monitor->wait();

    // Pop from the queue.

    m_numTasks--;
    Task task = m_finished.removeFirst();
    s_monitor->leave();
    return task;
}

//------------------------------------------------------------------------

int MulticoreLauncher::getNumTasks(void) const
{
    return m_numTasks;
}

//------------------------------------------------------------------------

int MulticoreLauncher::getNumFinished(void) const
{
    return m_finished.getSize();
}

//------------------------------------------------------------------------

void MulticoreLauncher::popAll(const String& progressMessage)
{
    Timer timer;
    F32 progress = 0.0f;
    while (getNumTasks())
    {
        if (timer.getElapsed() > 0.1f)
        {
            printf("\r%s %d%%", progressMessage.getPtr(), (int)progress);
            timer.start();
        }
        pop();
        progress += (100.0f - progress) / (F32)(getNumTasks() + 1);
    }
    printf("\r%s 100%%\n", progressMessage.getPtr());
}

//------------------------------------------------------------------------

int MulticoreLauncher::getNumCores(void)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}

//------------------------------------------------------------------------

void MulticoreLauncher::setNumThreads(int numThreads)
{
    FW_ASSERT(numThreads > 0);
    s_lock.enter();

    s_desiredThreads = numThreads;
    if (s_numThreads != 0)
    {
        s_monitor->enter();
        applyNumThreads();
        s_monitor->leave();
    }

    s_lock.leave();
}

//------------------------------------------------------------------------

void MulticoreLauncher::applyNumThreads(void) // Must have the monitor.
{
    // Start new threads.

    while (s_numThreads < s_desiredThreads)
    {
        (new Thread)->start(threadFunc, NULL);
        s_numThreads++;
    }

    // Kill excess threads.

    if (s_numThreads > s_desiredThreads)
    {
        s_monitor->notifyAll();
        while (s_numThreads > s_desiredThreads)
            s_monitor->wait();
    }
}

//------------------------------------------------------------------------

void MulticoreLauncher::threadFunc(void* param)
{
    FW_UNREF(param);
    Thread::getCurrent()->setPriority(Thread::Priority_Min);
    s_monitor->enter();

    while (s_numThreads <= s_desiredThreads)
    {
        // No pending tasks => wait.

        if (!s_pending.getSize())
        {
            s_monitor->wait();
            continue;
        }

        // Pick a task.

        Task task = s_pending.removeFirst();
        MulticoreLauncher* launcher = task.launcher;

        // Execute.

        s_monitor->leave();
        task.func(task);
        failIfError();
        s_monitor->enter();

        // Mark as finished.

        launcher->m_finished.addLast(task);
        s_monitor->notifyAll();
    }

    s_numThreads--;
    delete Thread::getCurrent();
    s_monitor->notifyAll();
    s_monitor->leave();
}

//------------------------------------------------------------------------
