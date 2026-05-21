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

#include "io/File.hpp"

using namespace FW;

//------------------------------------------------------------------------

File::AsyncOp::~AsyncOp(void)
{
    wait();
    CloseHandle(m_overlapped.hEvent);
}

//------------------------------------------------------------------------

bool File::AsyncOp::isDone(void)
{
    if (m_done)
        return true;

    if (!HasOverlappedIoCompleted(&m_overlapped))
        return false;

    wait();
    return true;
}

//------------------------------------------------------------------------

void File::AsyncOp::wait(void)
{
    if (m_done)
        return;

    DWORD numBytes = 0;
    if (!GetOverlappedResult(m_fileHandle, &m_overlapped, &numBytes, TRUE))
    {
        setError("GetOverlappedResult() failed!");
        failed();
    }
    else if ((int)numBytes != m_expectedBytes)
    {
        setError("GetOverlappedResult() returned %d bytes, expected %d!", numBytes, m_expectedBytes);
        failed();
    }
    else
    {
        done();
    }
}

//------------------------------------------------------------------------

File::AsyncOp::AsyncOp(HANDLE fileHandle)
:   m_offset        (0),
    m_numBytes      (0),
    m_expectedBytes (0),
    m_userBytes     (0),
    m_readPtr       (NULL),
    m_writePtr      (NULL),

    m_copyBytes     (0),
    m_copySrc       (NULL),
    m_copyDst       (NULL),
    m_freePtr       (NULL),

    m_fileHandle    (fileHandle),
    m_done          (false),
    m_failed        (false)
{
    memset(&m_overlapped, 0, sizeof(m_overlapped));

    // Create event object. Without one, GetOverlappedResult()
    // occassionally fails to wait long enough on WinXP.

    m_overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (!m_overlapped.hEvent)
        failWin32Error("CreateEvent");
}

//------------------------------------------------------------------------

void File::AsyncOp::done(void)
{
    if (m_done)
        return;
    m_done = true;

    if (m_copyBytes && !m_failed)
        memcpy(m_copyDst, m_copySrc, m_copyBytes);
    if (m_freePtr)
        free(m_freePtr);
}

//------------------------------------------------------------------------

File::File(const String& name, Mode mode, bool disableCache)
:   m_name          (name),
    m_mode          (mode),
    m_disableCache  (disableCache),
    m_handle        (NULL),
    m_align         (1),

    m_size          (0),
    m_offset        (0)
{
    static bool privilegeSet = false;
    if (!privilegeSet)
    {
        privilegeSet = true;

        HANDLE token;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &token))
        {
            TOKEN_PRIVILEGES tp;
            tp.PrivilegeCount = 1;
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

            if (LookupPrivilegeValue(NULL, SE_MANAGE_VOLUME_NAME, &tp.Privileges[0].Luid))
                AdjustTokenPrivileges(token, FALSE, &tp, 0, NULL, NULL);

            CloseHandle(token);
        }
    }

    // Select mode.

    const char* modeName;
    DWORD access;
    DWORD creation;

    switch (mode)
    {
    case Read:      modeName = "read"; access = GENERIC_READ; creation = OPEN_EXISTING; break;
    case Create:    modeName = "create"; access = GENERIC_READ | GENERIC_WRITE; creation = CREATE_ALWAYS; break;
    case Modify:    modeName = "modify"; access = GENERIC_READ | GENERIC_WRITE; creation = OPEN_ALWAYS; break;
    default:        FW_ASSERT(false); return;
    }

    // Open.

    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    if (disableCache)
        flags |= FILE_FLAG_NO_BUFFERING;

    m_handle = CreateFile(
        name.getPtr(),
        access,
        FILE_SHARE_READ,
        NULL,
        creation,
        flags,
        NULL);

    if (m_handle == INVALID_HANDLE_VALUE)
        setError("Cannot open file '%s' for %s!", m_name.getPtr(), modeName);

    // Get size.

    LARGE_INTEGER size;
    size.QuadPart = 0;
    if (m_handle && !GetFileSizeEx(m_handle, &size))
        setError("GetFileSizeEx() failed on '%s'!", m_name.getPtr());
    m_size = size.QuadPart;
    m_actualSize = size.QuadPart;

    // Get alignment.

    if (disableCache)
    {
        DWORD bytesPerSector = 1;
        if (!GetDiskFreeSpace(NULL, NULL, &bytesPerSector, NULL, NULL))
            failWin32Error("GetDiskFreeSpace");
        m_align = bytesPerSector;
    }
    FW_ASSERT((m_align & (m_align - 1)) == 0);
}

//------------------------------------------------------------------------

File::~File(void)
{
    if (!m_handle)
        return;

    fixSize();
    CancelIo(m_handle);
    CloseHandle(m_handle);
}

//------------------------------------------------------------------------

bool File::checkWritable(void) const
{
    if (m_mode != File::Read)
        return true;
    setError("File '%s' was opened as read-only!", m_name.getPtr());
    return false;
}

//------------------------------------------------------------------------

void File::seek(S64 ofs)
{
    if (ofs >= 0 && ofs <= m_size)
        m_offset = ofs;
    else
        setError("Tried to seek outside '%s'!", m_name.getPtr());
}

//------------------------------------------------------------------------

void File::setSize(S64 size)
{
    if (!checkWritable() || !m_handle)
        return;
    if (size >= 0)
        m_size = size;
    else
        setError("Tried to set negative size for '%s'!", m_name.getPtr());
}

//------------------------------------------------------------------------

void File::allocateSpace(S64 size)
{
    if (m_mode == Read || !m_handle || m_actualSize >= size)
        return;

    LARGE_INTEGER ofs;
    ofs.QuadPart = (size + m_align - 1) & -m_align;
    if (SetFilePointerEx(m_handle, ofs, NULL, FILE_BEGIN) && SetEndOfFile(m_handle))
    {
        SetFileValidData(m_handle, ofs.QuadPart);
        m_actualSize = ofs.QuadPart;
    }
}

//------------------------------------------------------------------------

int File::read(void* ptr, int size)
{
    AsyncOp* op = readAsync(ptr, size);
    op->wait();
    int numBytes = op->getNumBytes();
    delete op;
    return numBytes;
}

//------------------------------------------------------------------------

void File::write(const void* ptr, int size)
{
    delete writeAsync(ptr, size);
}

//------------------------------------------------------------------------

void File::flush(void)
{
    if (m_mode == Read || !m_handle)
        return;

    profilePush("Flush file");
    if (!FlushFileBuffers(m_handle))
        setError("FlushFileBuffers() failed on '%s'!", m_name.getPtr());
    profilePop();
    fixSize();
}

//------------------------------------------------------------------------

File::AsyncOp* File::readAsync(void* ptr, int size)
{
    FW_ASSERT(ptr || !size);
    profilePush("Read file");

    // Create AsyncOp.

    AsyncOp* op = new AsyncOp(m_handle);
    op->m_userBytes = max((int)min((S64)size, m_size - m_offset), 0);
    if (!m_handle)
        op->m_userBytes = 0;

    int mask = m_align - 1;
    op->m_numBytes = min((op->m_userBytes + mask) & ~mask, max(size, 0));

    // Aligned => read directly.

    if (!op->m_numBytes || (((UPTR)ptr & (UPTR)mask) == 0 && (op->m_numBytes & mask) == 0))
    {
        op->m_offset    = m_offset;
        op->m_readPtr   = ptr;
    }

    // Unaligned => read through temporary buffer.

    else
    {
        op->m_offset    = m_offset & ~(S64)mask;
        op->m_numBytes  = (((S32)m_offset & mask) + op->m_userBytes + mask) & ~(S64)mask;
        op->m_readPtr   = allocAligned(op->m_freePtr, op->m_numBytes);
        op->m_copyBytes = op->m_userBytes;
        op->m_copySrc   = (U8*)op->m_readPtr + ((S32)m_offset & mask);
        op->m_copyDst   = ptr;
    }

    // Execute the op.

    op->m_expectedBytes = (S32)min((S64)op->m_numBytes, m_actualSize - op->m_offset);
    startOp(op);
    if (!op->hasFailed())
        m_offset += op->m_userBytes;

    profilePop();
    return op;
}

//------------------------------------------------------------------------

File::AsyncOp* File::writeAsync(const void* ptr, int size)
{
    FW_ASSERT(ptr || !size);
    profilePush("Write file");

    // Write past the end of file => expand.

    int mask = m_align - 1;
    S64 sizeNeeded = (m_offset + size + mask) & ~(S64)mask;
    if (m_actualSize < sizeNeeded)
    {
        if (m_disableCache)
            allocateSpace(max(sizeNeeded, m_actualSize * MinimumExpandNum / MinimumExpandDenom));
        m_actualSize = max(m_actualSize, sizeNeeded);
    }

    // Create AsyncOp.

    AsyncOp* op = new AsyncOp(m_handle);
    op->m_userBytes = size;
    if (op->m_userBytes < 0 || !checkWritable() || !m_handle)
        op->m_userBytes = 0;

    // Aligned => write directly.

    if (!op->m_userBytes || (((UPTR)ptr & (UPTR)mask) == 0 && (op->m_userBytes & mask) == 0))
    {
        op->m_offset    = m_offset;
        op->m_numBytes  = op->m_userBytes;
        op->m_writePtr  = ptr;
    }

    // Unaligned => write through temporary buffer.

    else
    {
        S64 start       = m_offset & ~(S64)mask;
        S64 end         = (m_offset + op->m_userBytes + mask) & ~(S64)mask;
        op->m_offset    = start;
        op->m_numBytes  = (S32)(end - start);
        U8* buffer      = (U8*)allocAligned(op->m_freePtr, op->m_numBytes);
        op->m_writePtr  = buffer;

        // Read head.

        if (start != m_offset && start < m_size)
            if (!readAligned(start, buffer, m_align))
                op->failed();

        // Read tail.

        if (end != m_offset + op->m_userBytes && end - m_align < m_size && (start == m_offset || end > start + m_align))
            if (!readAligned(end - m_align, buffer + end - start - m_align, m_align))
                op->failed();

        // Copy body.

        memcpy(buffer + m_offset - start, ptr, op->m_userBytes);
    }

    op->m_expectedBytes = op->m_numBytes;
    startOp(op);
    if (!op->hasFailed())
    {
        m_offset += op->m_userBytes;
        m_size = max(m_size, m_offset);
    }

    profilePop();
    return op;
}

//------------------------------------------------------------------------

void File::fixSize(void)
{
    if (m_mode == Read || !m_handle || m_actualSize == m_size)
        return;

    // Size is not aligned properly => reopen with buffering.

    profilePush("Resize file");

    bool reopen = ((m_size & (m_align - 1)) != 0);
    if (reopen)
    {
        CloseHandle(m_handle);
        m_handle = CreateFile(
            m_name.getPtr(),
            GENERIC_WRITE,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL);

        if (!m_handle)
            setError("CreateFile() failed on '%s'!", m_name.getPtr());
    }

    // Set size.

    LARGE_INTEGER ofs;
    ofs.QuadPart = m_size;
    if (!SetFilePointerEx(m_handle, ofs, NULL, FILE_BEGIN))
        setError("SetFilePointerEx() failed on '%s'!", m_name.getPtr());
    else if (!SetEndOfFile(m_handle))
        setError("SetEndOfFile() failed on '%s'!", m_name.getPtr());
    else
        m_actualSize = m_size;

    // File was reopened => reopen without buffering.

    if (reopen)
    {
        CloseHandle(m_handle);
        m_handle = CreateFile(
            m_name.getPtr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING,
            NULL);

        if (!m_handle)
            setError("CreateFile() failed on '%s'!", m_name.getPtr());
    }

    profilePop();
}

//------------------------------------------------------------------------

void File::startOp(AsyncOp* op)
{
    // Backup parameters.

    FW_ASSERT(op);
    S64 offset          = op->m_offset;
    S32 numBytes        = op->m_numBytes;
    S32 expectedBytes   = op->m_expectedBytes;
    U8* readPtr         = (U8*)op->m_readPtr;
    U8* writePtr        = (U8*)op->m_writePtr;

    // Nothing to do => skip.

    if (!numBytes)
    {
        op->done();
        return;
    }

    // Loop over blocks corresponding to MaxBytesPerSysCall.
    // Only the last one is executed asynchronously.

    S32 pos = 0;
    for (;;)
    {
        // Setup AsyncOp.

        AsyncOp* blockOp = op;
        if (pos + MaxBytesPerSysCall < numBytes)
            blockOp = new AsyncOp(m_handle);

        blockOp->m_offset   = offset + pos;
        blockOp->m_numBytes = min(numBytes - pos, (S32)MaxBytesPerSysCall);
        blockOp->m_expectedBytes = min(expectedBytes - pos, blockOp->m_numBytes);
        blockOp->m_readPtr  = (readPtr)  ? readPtr + pos  : NULL;
        blockOp->m_writePtr = (writePtr) ? writePtr + pos : NULL;

        // Queue the op.

        BOOL ok;
        const char* funcName;
        blockOp->m_overlapped.Offset = (DWORD)blockOp->m_offset;
        blockOp->m_overlapped.OffsetHigh = (DWORD)(blockOp->m_offset >> 32);

        if (op->m_readPtr)
        {
            ok = ReadFile(m_handle, blockOp->m_readPtr, blockOp->m_numBytes, NULL, &blockOp->m_overlapped);
            funcName = "ReadFile";
        }
        else
        {
            ok = WriteFile(m_handle, blockOp->m_writePtr, blockOp->m_numBytes, NULL, &blockOp->m_overlapped);
            funcName = "WriteFile";
        }

        // Check result.

        if (ok)
            blockOp->done();
        else if (GetLastError() != ERROR_IO_PENDING)
        {
            setError("%s() failed on '%s'!", funcName, m_name.getPtr());
            blockOp->failed();
        }

        // Last op => done.

        if (blockOp == op)
            break;

        // Wait for the op to finish.

        pos += blockOp->m_numBytes;
        blockOp->wait();
        bool failed = blockOp->hasFailed();
        delete blockOp;

        if (failed)
        {
            op->failed();
            break;
        }
    }
}

//------------------------------------------------------------------------

void* File::allocAligned(void*& base, int size)
{
    base = (U8*)malloc(size + m_align - 1);
    U8* ptr = (U8*)base + m_align - 1;
    ptr -= (UPTR)ptr & (UPTR)(m_align - 1);
    return ptr;
}

//------------------------------------------------------------------------

bool File::readAligned(S64 ofs, void* ptr, int size)
{
    AsyncOp* op     = new AsyncOp(m_handle);
    op->m_offset    = ofs;
    op->m_numBytes  = size;
    op->m_readPtr   = ptr;

    op->m_expectedBytes = (S32)min((S64)size, m_actualSize - ofs);
    startOp(op);
    op->wait();
    bool ok = (!op->hasFailed());
    delete op;
    return ok;
}

//------------------------------------------------------------------------
