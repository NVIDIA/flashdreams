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
#include "base/Array.hpp"

#include <stdarg.h>

namespace FW
{
//------------------------------------------------------------------------

class String
{
public:
                    String      (void)                          {}
                    String      (char chr)                      { set(chr); }
                    String      (const char* chars)             { set(chars); }
                    String      (const char* start, const char* end )	{ set(start, end); }
                    String      (const String& other)           { set(other); }
                    String      (S32 value)                     { setf("%d", value); }
                    String      (F64 value)                     { setf("%g", value); }
                    ~String     (void)                          {}

    int             getLength   (void) const                    { return max(m_chars.getSize() - 1, 0); }
    char            getChar     (int idx) const                 { FW_ASSERT(idx < getLength()); return m_chars[idx]; }
    const char*     getPtr      (void) const                    { return (m_chars.getSize()) ? m_chars.getPtr() : ""; }

    String&         reset       (void)                          { m_chars.reset(); return *this; }
    String&         set         (char chr);
    String&         set         (const char* chars);
    String&         set         (const char* start, const char* end);
    String&         set         (const String& other)           { m_chars = other.m_chars; return *this; }
    String&         setf        (const char* fmt, ...);
    String&         setfv       (const char* fmt, va_list args);

    String          substring   (int start, int end) const;
    String          substring   (int start) const               { return substring(start, getLength()); }

	String			trimStart	(void) const;
	String			trimEnd		(void) const;
	String			trim		(void) const;

	void			split		(char chr, Array<String>& pieces, bool includeEmpty = false) const;

    String&         clear       (void)                          { m_chars.clear(); }
    String&         append      (char chr);
    String&         append      (const char* chars);
    String&         append      (const String& other);
    String&         appendf     (const char* fmt, ...);
    String&         appendfv    (const char* fmt, va_list args);
    String&         compact     (void)                          { m_chars.compact(); }

    int             indexOf     (char chr) const                { return m_chars.indexOf(chr); }
    int             indexOf     (char chr, int fromIdx) const   { return m_chars.indexOf(chr, fromIdx); }
    int             lastIndexOf (char chr) const                { return m_chars.lastIndexOf(chr); }
    int             lastIndexOf (char chr, int fromIdx) const   { return m_chars.lastIndexOf(chr, fromIdx); }

    String          toUpper     (void) const;
    String          toLower     (void) const;
    bool            startsWith  (const String& str) const;
    bool            endsWith    (const String& str) const;

    String          getFileName (void) const;
    String          getDirName  (void) const;

    char            operator[]  (int idx) const                 { return getChar(idx); }
    String&         operator=   (const String& other)           { set(other); return *this; }
    String&         operator+=  (char chr)                      { append(chr); return *this; }
    String&         operator+=  (const String& other)           { append(other); return *this; }
    String          operator+   (char chr) const                { return String(*this).append(chr); }
    String          operator+   (const String& other) const     { return String(*this).append(other); }
    bool            operator==  (const char* chars) const       { return (strcmp(getPtr(), chars) == 0); }
    bool            operator==  (const String& other) const     { return (strcmp(getPtr(), other.getPtr()) == 0); }
    bool            operator!=  (const char* chars) const       { return (strcmp(getPtr(), chars) != 0); }
    bool            operator!=  (const String& other) const     { return (strcmp(getPtr(), other.getPtr()) != 0); }
    bool            operator<   (const char* chars) const       { return (strcmp(getPtr(), chars) < 0); }
    bool            operator<   (const String& other) const     { return (strcmp(getPtr(), other.getPtr()) < 0); }
    bool            operator>   (const char* chars) const       { return (strcmp(getPtr(), chars) > 0); }
    bool            operator>   (const String& other) const     { return (strcmp(getPtr(), other.getPtr()) > 0); }
    bool            operator>=  (const char* chars) const       { return (strcmp(getPtr(), chars) <= 0); }
    bool            operator>=  (const String& other) const     { return (strcmp(getPtr(), other.getPtr()) <= 0); }
    bool            operator<=  (const char* chars) const       { return (strcmp(getPtr(), chars) >= 0); }
    bool            operator<=  (const String& other) const     { return (strcmp(getPtr(), other.getPtr()) >= 0); }

private:
    static int      strlen      (const char* chars);
    static int      strcmp      (const char* a, const char* b);

private:
    Array<char>     m_chars;
};

//------------------------------------------------------------------------

String  getDateString   (void);

bool    parseSpace      (const char*& ptr);
bool    parseChar       (const char*& ptr, char chr);
bool    parseLiteral    (const char*& ptr, const char* str);
bool    parseInt        (const char*& ptr, S32& value);
bool    parseInt        (const char*& ptr, S64& value);
bool    parseHex        (const char*& ptr, U32& value);
bool    parseFloat      (const char*& ptr, F32& value);

//------------------------------------------------------------------------
}
