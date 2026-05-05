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

#include "base/String.hpp"

#include <stdio.h>
#include <time.h>
#include <ctype.h>

using namespace FW;

//------------------------------------------------------------------------

String& String::set(char chr)
{
    m_chars.reset(2);
    m_chars[0] = chr;
    m_chars[1] = '\0';
    return *this;
}

//------------------------------------------------------------------------

String& String::set(const char* chars)
{
    int len = strlen(chars);
    if (!len)
        return reset();

    m_chars.set(chars, len + 1);
    return *this;
}

//------------------------------------------------------------------------

String& String::set(const char* start, const char* end)
{
    int len = int(end-start);
    if (!len)
        return reset();

    m_chars.set(start, len + 1);
	m_chars[len] = 0x00;
    return *this;
}

//------------------------------------------------------------------------
String& String::setf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    setfv(fmt, args);
    va_end(args);
    return *this;
}

//------------------------------------------------------------------------

String& String::setfv(const char* fmt, va_list args)
{
    int len = _vscprintf(fmt, args);
    if (!len)
        return reset();

    m_chars.reset(len + 1);
    vsprintf_s(m_chars.getPtr(), len + 1, fmt, args);
    return *this;
}

//------------------------------------------------------------------------

String String::substring(int start, int end) const
{
    FW_ASSERT(end <= getLength());

    String res;
    res.m_chars.reset(end - start + 1);
    Array<char>::copy(res.m_chars.getPtr(), m_chars.getPtr(start), end - start);
    res.m_chars[end - start] = '\0';
    return res;
}

//------------------------------------------------------------------------

String String::trimStart(void) const
{
	int len = getLength();
	for (int i=0; i < len; i++)
		if (!isspace(m_chars[i]))
			return substring(i, getLength());
	return "";
}

//------------------------------------------------------------------------

String String::trimEnd(void) const
{
	int len = getLength();
	for (int i=len-1; i >= 0; i--)
		if (!isspace(m_chars[i]))
			return substring(0, i+1);
	return "";
}

//------------------------------------------------------------------------

String String::trim(void) const
{
	int len = getLength();
	int idx = -1;
	
	for (int i=0; i < len; i++)
	{
		if (!isspace(m_chars[i]))
		{
			idx = i;
			break;
		}
	}
	
	if (idx == -1)
		return "";
	
	for (int i=len-1; i >= 0; i--)
		if (!isspace(m_chars[i]))
			return substring(idx, i+1);

	return ""; // unreachable
}

//------------------------------------------------------------------------

void String::split(char chr, Array<String>& pieces, bool includeEmpty) const
{
	int n = 0;
	while (n <= getLength())
	{
		int c = indexOf(chr, n);
		if (c < 0)
			c = getLength();
		if (c != n || includeEmpty)
			pieces.add(substring(n, c));
		n = c+1;
	}
}

//------------------------------------------------------------------------

String& String::append(char chr)
{
    int len = getLength();
    m_chars.resize(len + 2);
    m_chars[len] = chr;
    m_chars[len + 1] = '\0';
    return *this;
}

//------------------------------------------------------------------------

String& String::append(const char* chars)
{
    int lenA = getLength();
    int lenB = strlen(chars);
    m_chars.resize(lenA + lenB + 1);
    Array<char>::copy(m_chars.getPtr(lenA), chars, lenB);
    m_chars[lenA + lenB] = '\0';
    return *this;
}

//------------------------------------------------------------------------

String& String::append(const String& other)
{
    if (&other != this)
        return append(other.getPtr());

    String tmp = other;
    return append(tmp.getPtr());
}

//------------------------------------------------------------------------

String& String::appendf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    appendfv(fmt, args);
    va_end(args);
    return *this;
}

//------------------------------------------------------------------------

String& String::appendfv(const char* fmt, va_list args)
{
    int lenA = getLength();
    int lenB = _vscprintf(fmt, args);
    m_chars.resize(lenA + lenB + 1);
    vsprintf_s(m_chars.getPtr(lenA), lenB + 1, fmt, args);
    return *this;
}

//------------------------------------------------------------------------

String String::toUpper(void) const
{
    String str;
    str.m_chars.reset(m_chars.getSize());
    for (int i = 0; i < m_chars.getSize(); i++)
    {
        char c = m_chars[i];
        if (c >= 'a' && c <= 'z')
            c += 'A' - 'a';
        str.m_chars[i] = c;
    }
    return str;
}

//------------------------------------------------------------------------

String String::toLower(void) const
{
    String str;
    str.m_chars.reset(m_chars.getSize());
    for (int i = 0; i < m_chars.getSize(); i++)
    {
        char c = m_chars[i];
        if (c >= 'A' && c <= 'Z')
            c += 'a' - 'A';
        str.m_chars[i] = c;
    }
    return str;
}

//------------------------------------------------------------------------

bool String::startsWith(const String& str) const
{
    const char* a = getPtr();
    const char* b = str.getPtr();
    for (int ofs = 0; b[ofs]; ofs++)
        if (a[ofs] != b[ofs])
            return false;
    return true;
}

//------------------------------------------------------------------------

bool String::endsWith(const String& str) const
{
    int a = getLength();
    int b = str.getLength();
    if (a < b)
        return false;
    return (strcmp(getPtr() + a - b, str.getPtr()) == 0);
}

//------------------------------------------------------------------------

String String::getFileName(void) const
{
    int idx = max(lastIndexOf('/'), lastIndexOf('\\'));
    return (idx == -1) ? *this : substring(idx + 1, getLength());
}

//------------------------------------------------------------------------

String String::getDirName(void) const
{
    int idx = max(lastIndexOf('/'), lastIndexOf('\\'));
    return (idx == -1) ? "." : substring(0, idx);
}

//------------------------------------------------------------------------

int String::strlen(const char* chars)
{
    if (!chars)
        return 0;

    int len = 0;
    while (chars[len])
        len++;
    return len;
}

//------------------------------------------------------------------------

int String::strcmp(const char* a, const char* b)
{
    int ofs = 0;
    while (a[ofs] && a[ofs] == b[ofs])
        ofs++;
    return a[ofs] - b[ofs];
}

//------------------------------------------------------------------------

String FW::getDateString(void)
{
    // Query and format.

    char buffer[256];
    time_t currTime;
    time(&currTime);
    if (ctime_s(buffer, sizeof(buffer), &currTime) != 0)
        fail("ctime_s() failed!");

    // Strip linefeed.

    char* ptr = buffer;
    while (*ptr && *ptr != '\n' && *ptr != '\r')
        ptr++;
    *ptr = 0;
    return buffer;
}

//------------------------------------------------------------------------

bool FW::parseSpace(const char*& ptr)
{
    FW_ASSERT(ptr);
    while (*ptr == ' ' || *ptr == '\t')
        ptr++;
    return true;
}

//------------------------------------------------------------------------

bool FW::parseChar(const char*& ptr, char chr)
{
    FW_ASSERT(ptr);
    if (*ptr != chr)
        return false;
    ptr++;
    return true;
}

//------------------------------------------------------------------------

bool FW::parseLiteral(const char*& ptr, const char* str)
{
    FW_ASSERT(ptr && str);
    const char* tmp = ptr;

    while (*str && *tmp == *str)
    {
        tmp++;
        str++;
    }
    if (*str)
        return false;

    ptr = tmp;
    return true;
}

//------------------------------------------------------------------------

bool FW::parseInt(const char*& ptr, S32& value)
{
    const char* tmp = ptr;
    S32 v = 0;
    bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));
    if (*tmp < '0' || *tmp > '9')
        return false;
    while (*tmp >= '0' && *tmp <= '9')
        v = v * 10 + *tmp++ - '0';

    value = (neg) ? -v : v;
    ptr = tmp;
    return true;
}

//------------------------------------------------------------------------

bool FW::parseInt(const char*& ptr, S64& value)
{
    const char* tmp = ptr;
    S64 v = 0;
    bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));
    if (*tmp < '0' || *tmp > '9')
        return false;
    while (*tmp >= '0' && *tmp <= '9')
        v = v * 10 + *tmp++ - '0';

    value = (neg) ? -v : v;
    ptr = tmp;
    return true;
}

//------------------------------------------------------------------------

bool FW::parseHex(const char*& ptr, U32& value)
{
    const char* tmp = ptr;
    U32 v = 0;
    for (;;)
    {
        if (*tmp >= '0' && *tmp <= '9')         v = v * 16 + *tmp++ - '0';
        else if (*tmp >= 'A' && *tmp <= 'F')    v = v * 16 + *tmp++ - 'A' + 10;
        else if (*tmp >= 'a' && *tmp <= 'f')    v = v * 16 + *tmp++ - 'a' + 10;
        else                                    break;
    }

    if (tmp == ptr)
        return false;

    value = v;
    ptr = tmp;
    return true;
}

//------------------------------------------------------------------------

bool FW::parseFloat(const char*& ptr, F32& value)
{
    const char* tmp = ptr;
    bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));

    F32 v = 0.0f;
    int numDigits = 0;
    while (*tmp >= '0' && *tmp <= '9')
    {
        v = v * 10.0f + (F32)(*tmp++ - '0');
        numDigits++;
    }
    if (parseChar(tmp, '.'))
    {
        F32 scale = 1.0f;
        while (*tmp >= '0' && *tmp <= '9')
        {
            scale *= 0.1f;
            v += scale * (F32)(*tmp++ - '0');
            numDigits++;
        }
    }
    if (!numDigits)
        return false;

    ptr = tmp;
    if (*ptr == '#')
    {
        if (parseLiteral(ptr, "#INF"))
        {
            value = bitsToFloat((neg) ? 0xFF800000 : 0x7F800000);
            return true;
        }
        if (parseLiteral(ptr, "#SNAN"))
        {
            value = bitsToFloat((neg) ? 0xFF800001 : 0x7F800001);
            return true;
        }
        if (parseLiteral(ptr, "#QNAN"))
        {
            value = bitsToFloat((neg) ? 0xFFC00001 : 0x7FC00001);
            return true;
        }
        if (parseLiteral(ptr, "#IND"))
        {
            value = bitsToFloat((neg) ? 0xFFC00000 : 0x7FC00000);
            return true;
        }
    }

    S32 e = 0;
    if ((parseChar(tmp, 'e') || parseChar(tmp, 'E')) && parseInt(tmp, e))
    {
        ptr = tmp;
        if (e)
            v *= pow(10.0f, (F32)e);
    }
    value = (neg) ? -v : v;
    return true;
}

//------------------------------------------------------------------------
