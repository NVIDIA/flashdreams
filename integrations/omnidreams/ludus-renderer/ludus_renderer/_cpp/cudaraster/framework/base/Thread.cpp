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

#include "base/Thread.hpp"

using namespace FW;

//------------------------------------------------------------------------

Spinlock            Thread::s_lock;
Hash<U32, Thread*>  Thread::s_threads;
Thread*             Thread::s_mainThread = NULL;

//------------------------------------------------------------------------

Spinlock::Spinlock(void)
{
    InitializeCriticalSection(&m_critSect);
}

//------------------------------------------------------------------------

Spinlock::~Spinlock(void)
{
    DeleteCriticalSection(&m_critSect);
}

//------------------------------------------------------------------------

void Spinlock::enter(void)
{
    EnterCriticalSection(&m_critSect);
}

//------------------------------------------------------------------------

void Spinlock::leave(void)
{
    LeaveCriticalSection(&m_critSect);
}

//------------------------------------------------------------------------

Semaphore::Semaphore(int initCount, int maxCount)
{
    m_handle = CreateSemaphore(NULL, initCount, maxCount, NULL);
    if (!m_handle)
        failWin32Error("CreateSemaphore");
}

//------------------------------------------------------------------------

Semaphore::~Semaphore(void)
{
    CloseHandle(m_handle);
}

//------------------------------------------------------------------------

bool Semaphore::acquire(int millis)
{
    DWORD res = WaitForSingleObject(m_handle, (millis >= 0) ? millis : INFINITE);
    if (res == WAIT_FAILED)
        failWin32Error("WaitForSingleObject");
    return (res == WAIT_OBJECT_0);
}

//------------------------------------------------------------------------

void Semaphore::release(void)
{
    if (!ReleaseSemaphore(m_handle, 1, NULL))
        failWin32Error("ReleaseSemaphore");
}

//------------------------------------------------------------------------

Monitor::Monitor(void)
:   m_ownerSem      (1, 1),
    m_waitSem       (0, 1),
    m_notifySem     (0, 1),
    m_ownerThread   (0),
    m_enterCount    (0),
    m_waitCount     (0)
{
    InitializeCriticalSection(&m_mutex);
    if (isAvailable_InitializeConditionVariable())
        InitializeConditionVariable(&m_condition);
}

//------------------------------------------------------------------------

Monitor::~Monitor(void)
{
    DeleteCriticalSection(&m_mutex);
}

//------------------------------------------------------------------------

void Monitor::enter(void)
{
    if (isAvailable_InitializeConditionVariable())
    {
        EnterCriticalSection(&m_mutex);
    }
    else
    {
    U32 currThread = Thread::getID();

    m_lock.enter();
    if (m_ownerThread != currThread || !m_enterCount)
    {
        m_lock.leave();
        m_ownerSem.acquire();
        m_lock.enter();
    }

    m_ownerThread = currThread;
    m_enterCount++;
    m_lock.leave();
}
}

//------------------------------------------------------------------------

void Monitor::leave(void)
{
    if (isAvailable_InitializeConditionVariable())
    {
        LeaveCriticalSection(&m_mutex);
    }
    else
    {
    FW_ASSERT(m_ownerThread == Thread::getID() && m_enterCount);
    m_enterCount--;
    if (!m_enterCount)
        m_ownerSem.release();
}
}

//------------------------------------------------------------------------

void Monitor::wait(void)
{
    if (isAvailable_InitializeConditionVariable())
    {
        SleepConditionVariableCS(&m_condition, &m_mutex, INFINITE);
    }
    else
    {
    FW_ASSERT(m_ownerThread == Thread::getID() && m_enterCount);
    U32 currThread = m_ownerThread;
    int enterCount = m_enterCount;

    m_waitCount++;
    m_enterCount = 0;
    m_ownerSem.release();

    m_waitSem.acquire();
    m_waitCount--;
    m_notifySem.release();

    m_ownerSem.acquire();
    m_lock.enter();
    m_ownerThread = currThread;
    m_enterCount = enterCount;
    m_lock.leave();
}
}

//------------------------------------------------------------------------

void Monitor::notify(void)
{
    if (isAvailable_InitializeConditionVariable())
    {
        WakeConditionVariable(&m_condition);
    }
    else
    {
    FW_ASSERT(m_ownerThread == Thread::getID() && m_enterCount);
    if (m_waitCount)
    {
        m_waitSem.release();
        m_notifySem.acquire();
    }
}
}

//------------------------------------------------------------------------

void Monitor::notifyAll(void)
{
    if (isAvailable_InitializeConditionVariable())
    {
        WakeAllConditionVariable(&m_condition);
    }
    else
    {
    FW_ASSERT(m_ownerThread == Thread::getID() && m_enterCount);
    while (m_waitCount)
    {
        m_waitSem.release();
        m_notifySem.acquire();
    }
}
}

//------------------------------------------------------------------------

Thread::Thread(void)
:   m_refCount  (0),
    m_id        (0),
    m_handle    (NULL),
    m_priority  (Priority_Normal)
{
}

//------------------------------------------------------------------------

Thread::~Thread(void)
{
    // Wait and exit.

    if (this != getCurrent())
        join();
    else
    {
        failIfError();
        refer();
        m_exited = true;
        unrefer();
    }

    // Deinit user data.

    for (int i = m_userData.firstSlot(); i != -1; i = m_userData.nextSlot(i))
    {
        const UserData& data = m_userData.getSlot(i).value;
        if (data.deinitFunc)
            data.deinitFunc(data.data);
    }
}

//------------------------------------------------------------------------

void Thread::start(ThreadFunc func, void* param)
{
    m_startLock.enter();
    join();

    StartParams params;
    params.thread       = this;
    params.userFunc     = func;
    params.userParam    = param;
    params.ready.acquire();

    if (!CreateThread(NULL, 0, threadProc, &params, 0, NULL))
        failWin32Error("CreateThread");

    params.ready.acquire();
    m_startLock.leave();
}

//------------------------------------------------------------------------

Thread* Thread::getCurrent(void)
{
    s_lock.enter();
    Thread** found = s_threads.search(getID());
    Thread* thread = (found) ? *found : NULL;
    s_lock.leave();

    if (!thread)
    {
        thread = new Thread;
        thread->started();
    }
    return thread;
}

//------------------------------------------------------------------------

Thread* Thread::getMain(void)
{
    getCurrent();
    return s_mainThread;
}

//------------------------------------------------------------------------

bool Thread::isMain(void)
{
    Thread* curr = getCurrent();
    return (curr == s_mainThread);
}

//------------------------------------------------------------------------

U32 Thread::getID(void)
{
    return GetCurrentThreadId();
}

//------------------------------------------------------------------------

void Thread::sleep(int millis)
{
    Sleep(millis);
}

//------------------------------------------------------------------------

void Thread::yield(void)
{
    SwitchToThread();
}

//------------------------------------------------------------------------

int Thread::getPriority(void)
{
    refer();
    if (m_handle)
        m_priority = GetThreadPriority(m_handle);
    unrefer();

    if (m_priority == THREAD_PRIORITY_ERROR_RETURN)
        failWin32Error("GetThreadPriority");
    return m_priority;
}

//------------------------------------------------------------------------

void Thread::setPriority(int priority)
{
    refer();
    m_priority = priority;
    if (m_handle && !SetThreadPriority(m_handle, priority))
        failWin32Error("SetThreadPriority");
    unrefer();
}

//------------------------------------------------------------------------

bool Thread::isAlive(void)
{
    bool alive = false;
    refer();

    if (m_handle)
    {
        DWORD exitCode;
        if (!GetExitCodeThread(m_handle, &exitCode))
            failWin32Error("GetExitCodeThread");

        if (exitCode == STILL_ACTIVE)
            alive = true;
        else
            m_exited = true;
    }

    unrefer();
    return alive;
}

//------------------------------------------------------------------------

void Thread::join(void)
{
    FW_ASSERT(this != getMain());
    FW_ASSERT(this != getCurrent());

    refer();
    if (m_handle && WaitForSingleObject(m_handle, INFINITE) == WAIT_FAILED)
        failWin32Error("WaitForSingleObject");
    m_exited = true;
    unrefer();
}

//------------------------------------------------------------------------

void* Thread::getUserData(const String& id)
{
    m_lock.enter();
    UserData* found = m_userData.search(id);
    void* data = (found) ? found->data : NULL;
    m_lock.leave();
    return data;
}

//------------------------------------------------------------------------

void Thread::setUserData(const String& id, void* data, DeinitFunc deinitFunc)
{
    UserData oldData;
    oldData.data = NULL;
    oldData.deinitFunc = NULL;

    UserData newData;
    newData.data = data;
    newData.deinitFunc = deinitFunc;

    // Replace data.

    m_lock.enter();

    UserData* found = m_userData.search(id);
    if (found)
    {
        oldData = *found;
        *found = newData;
    }

    if ((found != NULL) != (data != NULL || deinitFunc != NULL))
    {
        if (found)
            m_userData.remove(id);
        else
            m_userData.add(id, newData);
    }

    m_lock.leave();

    // Deinit old data.

    if (oldData.deinitFunc)
        oldData.deinitFunc(oldData.data);
}

//------------------------------------------------------------------------

void Thread::suspendAll(void)
{
    s_lock.enter();
    for (int i = s_threads.firstSlot(); i != -1; i = s_threads.nextSlot(i))
    {
        Thread* thread = s_threads.getSlot(i).value;
        thread->refer();
        if (thread->m_handle && thread->m_id != getID())
            SuspendThread(thread->m_handle);
        thread->unrefer();
    }
    s_lock.leave();
}

//------------------------------------------------------------------------

void Thread::refer(void)
{
    m_lock.enter();
    m_refCount++;
    m_lock.leave();
}

//------------------------------------------------------------------------

void Thread::unrefer(void)
{
    m_lock.enter();
    if (--m_refCount == 0 && m_exited)
    {
        m_exited = false;
        exited();
    }
    m_lock.leave();
}

//------------------------------------------------------------------------

void Thread::started(void)
{
    m_id = getID();
    HANDLE process = GetCurrentProcess();
    if (!DuplicateHandle(process, GetCurrentThread(), process, &m_handle, THREAD_ALL_ACCESS, FALSE, 0))
        failWin32Error("DuplicateHandle");

    s_lock.enter();

    if (!s_mainThread)
        s_mainThread = this;

    if (!s_threads.contains(m_id))
        s_threads.add(m_id, this);

    s_lock.leave();
}

//------------------------------------------------------------------------

void Thread::exited(void)
{
    if (!m_handle)
        return;

    s_lock.enter();

    if (this == s_mainThread)
        s_mainThread = NULL;

    if (s_threads.contains(m_id))
        s_threads.remove(m_id);

    if (!s_threads.getSize())
        s_threads.reset();

    s_lock.leave();

    if (m_handle)
        CloseHandle(m_handle);
    m_id = 0;
    m_handle = NULL;
}

//------------------------------------------------------------------------

DWORD WINAPI Thread::threadProc(LPVOID lpParameter)
{
    StartParams*    params      = (StartParams*)lpParameter;
    Thread*         thread      = params->thread;
    ThreadFunc      userFunc    = params->userFunc;
    void*           userParam   = params->userParam;

    // Initialize.

    thread->started();
    thread->setPriority(thread->m_priority);
    params->ready.release();

    // Execute.

    userFunc(userParam);

    // Check whether the thread object still exists,
    // as userFunc() may have deleted it.

    s_lock.enter();
    bool exists = s_threads.contains(getID());
    s_lock.leave();

    // Still exists => deinit.

    if (exists)
    {
        failIfError();
        thread->getPriority();

        thread->refer();
        thread->m_exited = true;
        thread->unrefer();
    }
    return 0;
}

//------------------------------------------------------------------------
