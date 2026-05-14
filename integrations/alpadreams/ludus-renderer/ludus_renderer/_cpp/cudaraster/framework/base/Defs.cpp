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

#include "base/Defs.hpp"
#include "base/String.hpp"
#include "base/Thread.hpp"
#include "base/Timer.hpp"
#include "gui/Window.hpp"
#include "io/File.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <malloc.h>

using namespace FW;

//------------------------------------------------------------------------

#define FW_MEM_DEBUG    0

//------------------------------------------------------------------------

struct AllocHeader
{
    AllocHeader*    prev;
    AllocHeader*    next;
    size_t          size;
    const char*     ownerID;
};

//------------------------------------------------------------------------

struct ProfileTimer
{
    String          id;
    Timer           timer;
    S32             parent;
    Array<S32>      children;
};

//------------------------------------------------------------------------
// Hack to guard against the fact that s_lock may be accessed by malloc()
// and free() before it has been initialized and/or after it has been
// destroyed.

class SafeSpinlock : public Spinlock
{
public:
                SafeSpinlock    (void) { s_inited = true; }
                ~SafeSpinlock   (void) { s_inited = false; }

    void        enter           (void) { if (s_inited) Spinlock::enter(); }
    void        leave           (void) { if (s_inited) Spinlock::leave(); }

private:
    static bool s_inited;
};

bool SafeSpinlock::s_inited = false;

//------------------------------------------------------------------------

static SafeSpinlock                     s_lock;
static size_t                           s_memoryUsed        = 0;
static bool                             s_hasFailed         = false;
static int                              s_nestingLevel      = 0;
static bool                             s_discardEvents     = false;
static String                           s_emptyString;

static Array<File*>                     s_logFiles;
static Array<BufferedOutputStream*>     s_logStreams;

#if FW_MEM_DEBUG
static bool                             s_memPushingOwner   = false;
static AllocHeader                      s_memAllocs         = { &s_memAllocs, &s_memAllocs, 0, NULL };
static Hash<U32, Array<const char*> >   s_memOwnerStacks;
#endif

static bool                             s_profileStarted    = false;
static Hash<const char*, S32>           s_profilePointerToToken;
static Hash<String, S32>                s_profileStringToToken;
static Hash<Vec2i, S32>                 s_profileTimerHash;         // (parentTimer, childToken) => childTimer
static Array<ProfileTimer>              s_profileTimers;
static Array<S32>                       s_profileStack;

//------------------------------------------------------------------------

static void deinitString(void* str)
{
    delete (String*)str;
}

//------------------------------------------------------------------------

void* FW::malloc(size_t size)
{
    FW_ASSERT(size >= 0);

#if FW_MEM_DEBUG
    s_lock.enter();

    AllocHeader* alloc = (AllocHeader*)::malloc(sizeof(AllocHeader) + size);
    if (!alloc)
        fail("Out of memory!");

    void* ptr           = alloc + 1;
    alloc->prev         = s_memAllocs.prev;
    alloc->next         = &s_memAllocs;
    alloc->prev->next   = alloc;
    alloc->next->prev   = alloc;
    alloc->size         = size;
    alloc->ownerID      = "Uncategorized";
    s_memoryUsed        += size;

    if (!s_memPushingOwner)
    {
        U32 threadID = Thread::getID();
        if (s_memOwnerStacks.contains(threadID) && s_memOwnerStacks[threadID].getSize())
               alloc->ownerID = s_memOwnerStacks[threadID].getLast();
    }

    s_lock.leave();

#else
    void* ptr = ::malloc(size);
    if (!ptr)
        fail("Out of memory!");

    s_lock.enter();
    s_memoryUsed += _msize(ptr);
    s_lock.leave();
#endif

    return ptr;
}

//------------------------------------------------------------------------

void FW::free(void* ptr)
{
    if (!ptr)
        return;

#if FW_MEM_DEBUG
    s_lock.enter();

    AllocHeader* alloc = (AllocHeader*)ptr - 1;
    alloc->prev->next = alloc->next;
    alloc->next->prev = alloc->prev;
    s_memoryUsed -= alloc->size;
    ::free(alloc);

    s_lock.leave();

#else
    s_lock.enter();
    s_memoryUsed -= _msize(ptr);
    s_lock.leave();

    ::free(ptr);
#endif
}

//------------------------------------------------------------------------

void* FW::realloc(void* ptr, size_t size)
{
    FW_ASSERT(size >= 0);

    if (!ptr)
        return FW::malloc(size);

    if (!size)
    {
        FW::free(ptr);
        return NULL;
    }

#if FW_MEM_DEBUG
    void* newPtr = FW::malloc(size);
    memcpy(newPtr, ptr, min(size, ((AllocHeader*)ptr - 1)->size));
    FW::free(ptr);

#else
    size_t oldSize = _msize(ptr);
    void* newPtr = ::realloc(ptr, size);
    if (!newPtr)
        fail("Out of memory!");

    s_lock.enter();
    s_memoryUsed += _msize(newPtr) - oldSize;
    s_lock.leave();
#endif

    return newPtr;
}

//------------------------------------------------------------------------

void FW::printf(const char* fmt, ...)
{
    s_lock.enter();
    va_list args;
    va_start(args, fmt);

    vprintf(fmt, args);
    for (int i = 0; i < s_logFiles.getSize(); i++)
        s_logStreams[i]->writefv(fmt, args);

    va_end(args);
    s_lock.leave();
}

//------------------------------------------------------------------------

String FW::sprintf(const char* fmt, ...)
{
    String str;
    va_list args;
    va_start(args, fmt);
    str.setfv(fmt, args);
    va_end(args);
    return str;
}

//------------------------------------------------------------------------

void FW::setError(const char* fmt, ...)
{
    if (hasError())
        return;

    String* str = new String;
    va_list args;
    va_start(args, fmt);
    str->setfv(fmt, args);
    va_end(args);

    Thread::getCurrent()->setUserData("error", str, deinitString);
}

//------------------------------------------------------------------------

String FW::clearError(void)
{
    String old = getError();
    Thread::getCurrent()->setUserData("error", NULL);
    return old;
}

//------------------------------------------------------------------------

bool FW::restoreError(const String& old)
{
    bool had = hasError();
    Thread::getCurrent()->setUserData("error",
        (old.getLength()) ? new String(old) : NULL,
        (old.getLength()) ? deinitString : NULL);
    return had;
}

//------------------------------------------------------------------------

bool FW::hasError(void)
{
    return (Thread::getCurrent()->getUserData("error") != NULL);
}

//------------------------------------------------------------------------

const String& FW::getError(void)
{
    String* str = (String*)Thread::getCurrent()->getUserData("error");
    return (str) ? *str : s_emptyString;
}

//------------------------------------------------------------------------

void FW::fail(const char* fmt, ...)
{
    // Fail only once.

    s_lock.enter();
    bool alreadyFailed = s_hasFailed;
    s_hasFailed = true;
    s_lock.leave();
    if (alreadyFailed)
        return;

    // Print message.

    String tmp;
    va_list args;
    va_start(args, fmt);
    tmp.setfv(fmt, args);
    va_end(args);
    printf("\n%s\n", tmp.getPtr());

    // Try to prevent any user code from being executed.

    Thread::suspendAll();
    setDiscardEvents(true);

    // Display modal dialog.

    MessageBox(NULL, tmp.getPtr(), "Fatal error", MB_OK);

    // Running under a debugger => break here.

    if (IsDebuggerPresent())
        __debugbreak();

    // Kill the app.

    FatalExit(1);
}

//------------------------------------------------------------------------

void FW::failWin32Error(const char* funcName)
{
    DWORD err = GetLastError();
    LPTSTR msgBuf = NULL;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, err, 0, (LPTSTR)&msgBuf, 0, NULL);
    String msg(msgBuf);
    LocalFree(msgBuf);

    if (msg.getLength())
        fail("%s() failed!\n%s", funcName, msg.getPtr());
    else
        fail("%s() failed!\nError %d\n", funcName, err);
}

//------------------------------------------------------------------------

void FW::failIfError(void)
{
    if (hasError())
        fail("%s", getError().getPtr());
}

//------------------------------------------------------------------------

int FW::incNestingLevel(int delta)
{
    int old = s_nestingLevel;
    s_nestingLevel += delta;
    return old;
}

//------------------------------------------------------------------------

bool FW::setDiscardEvents(bool discard)
{
    bool old = s_discardEvents;
    s_discardEvents = discard;
    return old;
}

//------------------------------------------------------------------------

bool FW::getDiscardEvents(void)
{
    return s_discardEvents;
}

//------------------------------------------------------------------------

void FW::pushLogFile(const String& name, bool append)
{
    s_lock.enter();
    File* file = new File(name, (append) ? File::Modify : File::Create);
    file->seek(file->getSize());
    s_logFiles.add(file);
    s_logStreams.add(new BufferedOutputStream(*file, 1024, true, true));
    s_lock.leave();
}

//------------------------------------------------------------------------

void FW::popLogFile(void)
{
    s_lock.enter();
    if (s_logFiles.getSize())
    {
        s_logStreams.getLast()->flush();
        delete s_logFiles.removeLast();
        delete s_logStreams.removeLast();
        if (!s_logFiles.getSize())
        {
            s_logFiles.reset();
            s_logStreams.reset();
        }
    }
    s_lock.leave();
}

//------------------------------------------------------------------------

bool FW::hasLogFile(void)
{
    return (s_logFiles.getSize() != 0);
}

//------------------------------------------------------------------------

size_t FW::getMemoryUsed(void)
{
    return s_memoryUsed;
}

//------------------------------------------------------------------------

void FW::pushMemOwner(const char* id)
{
#if !FW_MEM_DEBUG
    FW_UNREF(id);
#else
    s_lock.enter();
    s_memPushingOwner = true;

    U32 threadID = Thread::getID();
    Array<const char*>* stack = s_memOwnerStacks.search(threadID);
    if (!stack)
    {
        stack = &s_memOwnerStacks.add(threadID);
        stack->clear();
    }
    stack->add(id);

    s_memPushingOwner = false;
    s_lock.leave();
#endif
}

//------------------------------------------------------------------------

void FW::popMemOwner(void)
{
#if FW_MEM_DEBUG
    U32 threadID = Thread::getID();
    Array<const char*>* stack = s_memOwnerStacks.search(threadID);
    if (stack)
    {
        stack->removeLast();
        if (!stack->getSize())
        {
            s_memOwnerStacks.remove(threadID);
            if (!s_memOwnerStacks.getSize())
                s_memOwnerStacks.reset();
        }
    }
#endif
}

//------------------------------------------------------------------------

void FW::printMemStats(void)
{
#if FW_MEM_DEBUG
    // Create snapshot of the alloc list.

    s_lock.enter();
    AllocHeader* first = NULL;
    for (AllocHeader* src = s_memAllocs.next; src != &s_memAllocs; src = src->next)
    {
        AllocHeader* alloc = (AllocHeader*)::malloc(sizeof(AllocHeader));
        *alloc = *src;
        alloc->next = first;
        first = alloc;
    }
    s_lock.leave();

    // Calculate total size per owner.

    Hash<String, S64> owners;
    for (AllocHeader* alloc = first; alloc;)
    {
        if (!owners.contains(alloc->ownerID))
            owners.add(alloc->ownerID, 0);
        owners[alloc->ownerID] += alloc->size;

        AllocHeader* next = alloc->next;
        ::free(alloc);
        alloc = next;
    }

    // Print.

    printf("\n");
    printf("%-32s%.2f\n", "Memory usage / megs", (F32)s_memoryUsed * exp2(-20));
    for (int slot = owners.firstSlot(); slot != -1; slot = owners.nextSlot(slot))
    {
        const HashEntry<String, S64>& entry = owners.getSlot(slot);
        printf("  %-30s%-12.2f%.0f%%\n",
            entry.key.getPtr(),
            (F32)entry.value * exp2(-20),
            (F32)entry.value / (F32)s_memoryUsed * 100.0f);
    }
    printf("\n");
#endif
}

//------------------------------------------------------------------------

void FW::profileStart(void)
{
    if (!Thread::isMain())
        fail("profileStart() can only be used in the main thread!");
    if (s_profileStarted)
        return;

    s_profileStarted = true;
    profilePush("Total time spent");
}

//------------------------------------------------------------------------

void FW::profilePush(const char* id)
{
    if (!s_profileStarted)
        return;
    if (!Thread::isMain())
        fail("profilePush() can only be used in the main thread!");

    // Find or create token.

    S32 token;
    S32* found = s_profilePointerToToken.search(id);
    if (found)
        token = *found;
    else
    {
        found = s_profileStringToToken.search(id);
        if (found)
            token = *found;
        else
        {
            token = s_profileStringToToken.getSize();
            s_profileStringToToken.add(id, token);
        }
        s_profilePointerToToken.add(id, token);
    }

    // Find or create timer.

    Vec2i timerKey(-1, token);
    if (s_profileStack.getSize())
        timerKey.x = s_profileStack.getLast();

    S32 timerIdx;
    found = s_profileTimerHash.search(timerKey);
    if (found)
        timerIdx = *found;
    else
    {
        timerIdx = s_profileTimers.getSize();
        s_profileTimerHash.add(timerKey, timerIdx);
        ProfileTimer& timer = s_profileTimers.add();
        timer.id = id;
        timer.parent = timerKey.x;
        if (timerKey.x != -1)
            s_profileTimers[timerKey.x].children.add(timerIdx);
    }

    // Push timer.

    if (s_profileStack.getSize() == 1)
        s_profileTimers[s_profileStack[0]].timer.start();
    s_profileStack.add(timerIdx);
    if (s_profileStack.getSize() > 1)
        s_profileTimers[timerIdx].timer.start();
}

//------------------------------------------------------------------------

void FW::profilePop(void)
{
    if (!s_profileStarted || s_profileStack.getSize() == 0)
        return;
    if (!Thread::isMain())
        fail("profilePop() can only be used in the main thread!");

    if (s_profileStack.getSize() > 1)
        s_profileTimers[s_profileStack.getLast()].timer.end();
    s_profileStack.removeLast();
    if (s_profileStack.getSize() == 1)
        s_profileTimers[s_profileStack.getLast()].timer.end();
}

//------------------------------------------------------------------------

void FW::profileEnd(bool printResults)
{
    if (!Thread::isMain())
        fail("profileEnd() can only be used in the main thread!");
    if (!s_profileStarted)
        return;

    // Pop remaining timers.

    while (s_profileStack.getSize())
        profilePop();

    // Recurse and print.

    if (printResults && s_profileTimers.getSize() > 1)
    {
        printf("\n");
        Array<Vec2i> stack(Vec2i(0, 0));
        while (stack.getSize())
        {
            Vec2i entry = stack.removeLast();
            const ProfileTimer& timer = s_profileTimers[entry.x];
            for (int i = timer.children.getSize() - 1; i >= 0; i--)
                stack.add(Vec2i(timer.children[i], entry.y + 2));

            printf("%*s%-*s%-8.3f",
                entry.y, "",
                32 - entry.y, timer.id.getPtr(),
                timer.timer.getTotal());

            printf("%.0f%%\n", timer.timer.getTotal() / s_profileTimers[0].timer.getTotal() * 100.0f);
        }
        printf("\n");
    }

    // Clean up.

    s_profileStarted = false;
    s_profilePointerToToken.reset();
    s_profileStringToToken.reset();
    s_profileTimerHash.reset();
    s_profileTimers.reset();
    s_profileStack.reset();
}

//------------------------------------------------------------------------
