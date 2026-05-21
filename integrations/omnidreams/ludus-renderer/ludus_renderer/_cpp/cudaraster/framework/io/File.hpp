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
#include "io/Stream.hpp"
#include "base/DLLImports.hpp"

namespace FW
{

//------------------------------------------------------------------------

class File : public InputStream, public OutputStream
{
public:
    enum
    {
        MaxBytesPerSysCall  = 16 << 20,
        MinimumExpandNum    = 5,
        MinimumExpandDenom  = 4
    };

    enum Mode
    {
        Read,   // must exist - cannot be written
        Create, // created or truncated - can be read or written
        Modify  // opened or created - can be read or written
    };

    class AsyncOp
    {
        friend class File;

    public:
                            ~AsyncOp                (void);

        bool                isDone                  (void);
        bool                hasFailed               (void) const    { return m_failed; }
        void                wait                    (void);
        int                 getNumBytes             (void) const    { FW_ASSERT(m_done); return (m_failed) ? 0 : m_userBytes; }

    private:
                            AsyncOp                 (HANDLE fileHandle);

        void                done                    (void);
        void                failed                  (void)          { m_failed = true; done(); }

    private:
                            AsyncOp                 (const AsyncOp&); // forbidden
        AsyncOp&            operator=               (const AsyncOp&); // forbidden

    private:
        S64                 m_offset;
        S32                 m_numBytes;             // Number of bytes to read or write.
        S32                 m_expectedBytes;        // Number of bytes to expect.
        S32                 m_userBytes;            // Number of bytes to return to the user.
        void*               m_readPtr;              // Buffer to read to.
        const void*         m_writePtr;             // Buffer to write from.

        S32                 m_copyBytes;            // Number of bytes to copy afterwards.
        const void*         m_copySrc;              // Pointer to copy from.
        void*               m_copyDst;              // Pointer to copy to.
        void*               m_freePtr;              // Pointer to free afterwards.

        HANDLE              m_fileHandle;
        OVERLAPPED          m_overlapped;
        bool                m_done;
        bool                m_failed;
    };

public:
                            File                    (const String& name, Mode mode, bool disableCache = false);
    virtual                 ~File                   (void);

    const String&           getName                 (void) const    { return m_name; }
    Mode                    getMode                 (void) const    { return m_mode; }
    int                     getAlign                (void) const    { return m_align; } // 1 unless disableCache = true.
    bool                    checkWritable           (void) const;

    S64                     getSize                 (void) const    { return m_size; }
    S64                     getOffset               (void) const    { return m_offset; }
    void                    seek                    (S64 ofs);
    void                    setSize                 (S64 size);
    void                    allocateSpace           (S64 size);

    virtual int             read                    (void* ptr, int size);
    virtual void            write                   (const void* ptr, int size);
    virtual void            flush                   (void);

    AsyncOp*                readAsync               (void* ptr, int size);
    AsyncOp*                writeAsync              (const void* ptr, int size);

private:
                            File                    (const File&); // forbidden
    File&                   operator=               (const File&); // forbidden

    void                    fixSize                 (void);
    void                    startOp                 (AsyncOp* op);

    void*                   allocAligned            (void*& base, int size);
    bool                    readAligned             (S64 ofs, void* ptr, int size);

private:
    String                  m_name;
    Mode                    m_mode;
    bool                    m_disableCache;
    HANDLE                  m_handle;
    S32                     m_align;

    S64                     m_size;
    S64                     m_actualSize;
    S64                     m_offset;
};

//------------------------------------------------------------------------
}
