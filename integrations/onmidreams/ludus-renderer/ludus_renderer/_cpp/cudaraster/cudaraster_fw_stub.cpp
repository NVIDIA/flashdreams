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

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal cross-platform replacement for framework/base/Defs.cpp. Provides
// just the public API surface from framework/base/Defs.hpp that the cudaraster
// build path actually references. The upstream Defs.cpp pulls in heavy
// Windows-only dependencies (Thread.hpp, Timer.hpp, Window.hpp, File.hpp) to
// implement features we don't need (memory tracking, log files, profiling,
// thread-local error storage, modal error dialogs) — none of which fit a
// PyTorch C++/CUDA extension. This file is compiled in place of Defs.cpp.

#include "base/Defs.hpp"
#include "base/String.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace FW
{

void* malloc(size_t size)
{
    void* ptr = ::malloc(size);
    if (!ptr)
        fail("Out of memory!");
    return ptr;
}

void free(void* ptr)
{
    ::free(ptr);
}

void* realloc(void* ptr, size_t size)
{
    if (!ptr)
        return malloc(size);
    if (!size)
    {
        free(ptr);
        return NULL;
    }
    void* newPtr = ::realloc(ptr, size);
    if (!newPtr)
        fail("Out of memory!");
    return newPtr;
}

void printf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

String sprintf(const char* fmt, ...)
{
    String str;
    va_list args;
    va_start(args, fmt);
    str.setfv(fmt, args);
    va_end(args);
    return str;
}

namespace
{
    thread_local String t_errorMessage;
    thread_local bool   t_errorSet = false;
    bool                s_discardEvents = false;
    int                 s_nestingLevel  = 0;
    bool                s_failed = false;
}

void setError(const char* fmt, ...)
{
    if (t_errorSet)
        return;
    va_list args;
    va_start(args, fmt);
    t_errorMessage.setfv(fmt, args);
    va_end(args);
    t_errorSet = true;
}

String clearError(void)
{
    String old = t_errorSet ? t_errorMessage : String();
    t_errorMessage.reset();
    t_errorSet = false;
    return old;
}

bool restoreError(const String& old)
{
    bool had = t_errorSet;
    if (old.getLength())
    {
        t_errorMessage = const_cast<String&>(old);
        t_errorSet = true;
    }
    else
    {
        t_errorMessage.reset();
        t_errorSet = false;
    }
    return had;
}

bool hasError(void)
{
    return t_errorSet;
}

const String& getError(void)
{
    static const String empty;
    return t_errorSet ? t_errorMessage : empty;
}

void fail(const char* fmt, ...)
{
    if (s_failed)
        std::abort();
    s_failed = true;

    String tmp;
    va_list args;
    va_start(args, fmt);
    tmp.setfv(fmt, args);
    va_end(args);

    std::fprintf(stderr, "\nFATAL: %s\n", tmp.getPtr());
    std::fflush(stderr);
    std::abort();
}

void failWin32Error(const char* funcName)
{
    fail("%s() failed!", funcName);
}

void failIfError(void)
{
    if (hasError())
        fail("%s", getError().getPtr());
}

int incNestingLevel(int delta)
{
    int old = s_nestingLevel;
    s_nestingLevel += delta;
    return old;
}

bool setDiscardEvents(bool discard)
{
    bool old = s_discardEvents;
    s_discardEvents = discard;
    return old;
}

bool getDiscardEvents(void)
{
    return s_discardEvents;
}

void pushLogFile(const String&, bool)         {}
void popLogFile(void)                          {}
bool hasLogFile(void)                          { return false; }

size_t getMemoryUsed(void)                     { return 0; }
void   pushMemOwner(const char*)               {}
void   popMemOwner(void)                       {}
void   printMemStats(void)                     {}

void profileStart(void)                        {}
void profilePush(const char*)                  {}
void profilePop(void)                          {}
void profileEnd(bool)                          {}

} // namespace FW
