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

#include "gui/Keys.hpp"
#include "base/String.hpp"
#include "base/DLLImports.hpp"

using namespace FW;

//------------------------------------------------------------------------

String FW::unicodeToKey(S32 cp)
{
    FW_ASSERT(cp >= 0 && cp <= 0x10FFFF);
    return sprintf("U+%04X", cp);
}

//------------------------------------------------------------------------

S32 FW::keyToUnicode(const String& key)
{
    const char* ptr = key.getPtr();
    S32 cp = 0;

    if (*ptr != 'U' || *++ptr != '+')
        return 0;

    while (*++ptr)
    {
        cp <<= 4;

        if (*ptr >= '0' && *ptr <= '9')
            cp += *ptr - '0';
        else if (*ptr >= 'A' && *ptr <= 'F')
            cp += *ptr - ('A' - 10);
        else
            return 0;

        if (cp > 0x10FFFF)
            return 0;
    }

    return cp;
}

//------------------------------------------------------------------------

String FW::vkeyToKey(U32 vkey)
{
    // Translate special keys.

    switch (vkey)
    {
    case VK_CANCEL:                 return FW_KEY_CANCEL;
    case VK_BACK:                   return FW_KEY_BACKSPACE;
    case VK_TAB:                    return FW_KEY_TAB;
    case VK_CLEAR:                  return FW_KEY_CLEAR;
    case VK_RETURN:                 return FW_KEY_ENTER;
    case VK_SHIFT:                  return FW_KEY_SHIFT;
    case VK_CONTROL:                return FW_KEY_CONTROL;
    case VK_MENU:                   return FW_KEY_ALT;
    case VK_PAUSE:                  return FW_KEY_PAUSE;
    case VK_CAPITAL:                return FW_KEY_CAPS_LOCK;
    case VK_KANA:                   return FW_KEY_KANA_MODE;
    case VK_JUNJA:                  return FW_KEY_JUNJA_MODE;
    case VK_FINAL:                  return FW_KEY_FINAL_MODE;
    case VK_HANJA:                  return FW_KEY_HANJA_MODE;
    case VK_ESCAPE:                 return FW_KEY_ESCAPE;
    case VK_CONVERT:                return FW_KEY_CONVERT;
    case VK_NONCONVERT:             return FW_KEY_NONCONVERT;
    case VK_ACCEPT:                 return FW_KEY_ACCEPT;
    case VK_MODECHANGE:             return FW_KEY_MODE_CHANGE;
    case VK_SPACE:                  return FW_KEY_SPACE;
    case VK_PRIOR:                  return FW_KEY_PAGE_UP;
    case VK_NEXT:                   return FW_KEY_PAGE_DOWN;
    case VK_END:                    return FW_KEY_END;
    case VK_HOME:                   return FW_KEY_HOME;
    case VK_LEFT:                   return FW_KEY_LEFT;
    case VK_UP:                     return FW_KEY_UP;
    case VK_RIGHT:                  return FW_KEY_RIGHT;
    case VK_DOWN:                   return FW_KEY_DOWN;
    case VK_SELECT:                 return FW_KEY_SELECT;
    case VK_PRINT:                  return FW_KEY_PRINT_SCREEN;
    case VK_EXECUTE:                return FW_KEY_EXECUTE;
    case VK_SNAPSHOT:               return FW_KEY_PRINT_SCREEN;
    case VK_INSERT:                 return FW_KEY_INSERT;
    case VK_DELETE:                 return FW_KEY_DELETE;
    case VK_HELP:                   return FW_KEY_HELP;
    case VK_LWIN:                   return FW_KEY_WIN;
    case VK_RWIN:                   return FW_KEY_WIN;
    case VK_APPS:                   return FW_KEY_APPS;
    case VK_SLEEP:                  break;
    case VK_NUMPAD0:                return FW_KEY_0;
    case VK_NUMPAD1:                return FW_KEY_1;
    case VK_NUMPAD2:                return FW_KEY_2;
    case VK_NUMPAD3:                return FW_KEY_3;
    case VK_NUMPAD4:                return FW_KEY_4;
    case VK_NUMPAD5:                return FW_KEY_5;
    case VK_NUMPAD6:                return FW_KEY_6;
    case VK_NUMPAD7:                return FW_KEY_7;
    case VK_NUMPAD8:                return FW_KEY_8;
    case VK_NUMPAD9:                return FW_KEY_9;
    case VK_MULTIPLY:               return FW_KEY_ASTERISK;
    case VK_ADD:                    return FW_KEY_PLUS;
    case VK_SEPARATOR:              return FW_KEY_COMMA;
    case VK_SUBTRACT:               return FW_KEY_MINUS;
    case VK_DECIMAL:                return FW_KEY_PERIOD;
    case VK_DIVIDE:                 return FW_KEY_SLASH;
    case VK_F1:                     return FW_KEY_F1;
    case VK_F2:                     return FW_KEY_F2;
    case VK_F3:                     return FW_KEY_F3;
    case VK_F4:                     return FW_KEY_F4;
    case VK_F5:                     return FW_KEY_F5;
    case VK_F6:                     return FW_KEY_F6;
    case VK_F7:                     return FW_KEY_F7;
    case VK_F8:                     return FW_KEY_F8;
    case VK_F9:                     return FW_KEY_F9;
    case VK_F10:                    return FW_KEY_F10;
    case VK_F11:                    return FW_KEY_F11;
    case VK_F12:                    return FW_KEY_F12;
    case VK_F13:                    return FW_KEY_F13;
    case VK_F14:                    return FW_KEY_F14;
    case VK_F15:                    return FW_KEY_F15;
    case VK_F16:                    return FW_KEY_F16;
    case VK_F17:                    return FW_KEY_F17;
    case VK_F18:                    return FW_KEY_F18;
    case VK_F19:                    return FW_KEY_F19;
    case VK_F20:                    return FW_KEY_F20;
    case VK_F21:                    return FW_KEY_F21;
    case VK_F22:                    return FW_KEY_F22;
    case VK_F23:                    return FW_KEY_F23;
    case VK_F24:                    return FW_KEY_F24;
    case VK_NUMLOCK:                return FW_KEY_NUM_LOCK;
    case VK_SCROLL:                 return FW_KEY_SCROLL;
    case VK_OEM_FJ_JISHO:           break;
    case VK_OEM_FJ_MASSHOU:         break;
    case VK_OEM_FJ_TOUROKU:         break;
    case VK_OEM_FJ_LOYA:            break;
    case VK_OEM_FJ_ROYA:            break;
    case VK_BROWSER_BACK:           return FW_KEY_BROWSER_BACK;
    case VK_BROWSER_FORWARD:        return FW_KEY_BROWSER_FORWARD;
    case VK_BROWSER_REFRESH:        return FW_KEY_BROWSER_REFRESH;
    case VK_BROWSER_STOP:           return FW_KEY_BROWSER_STOP;
    case VK_BROWSER_SEARCH:         return FW_KEY_BROWSER_SEARCH;
    case VK_BROWSER_FAVORITES:      return FW_KEY_BROWSER_FAVORITES;
    case VK_BROWSER_HOME:           return FW_KEY_BROWSER_HOME;
    case VK_VOLUME_MUTE:            return FW_KEY_VOLUME_MUTE;
    case VK_VOLUME_DOWN:            return FW_KEY_VOLUME_DOWN;
    case VK_VOLUME_UP:              return FW_KEY_VOLUME_UP;
    case VK_MEDIA_NEXT_TRACK:       return FW_KEY_MEDIA_NEXT_TRACK;
    case VK_MEDIA_PREV_TRACK:       return FW_KEY_MEDIA_PREVIOUS_TRACK;
    case VK_MEDIA_STOP:             return FW_KEY_MEDIA_STOP;
    case VK_MEDIA_PLAY_PAUSE:       return FW_KEY_MEDIA_PLAY_PAUSE;
    case VK_LAUNCH_MAIL:            return FW_KEY_LAUNCH_MAIL;
    case VK_LAUNCH_MEDIA_SELECT:    return FW_KEY_SELECT_MEDIA;
    case VK_LAUNCH_APP1:            return FW_KEY_LAUNCH_APPLICATION1;
    case VK_LAUNCH_APP2:            return FW_KEY_LAUNCH_APPLICATION2;
    case VK_OEM_1:                  break;
    case VK_OEM_PLUS:               break;
    case VK_OEM_COMMA:              break;
    case VK_OEM_MINUS:              break;
    case VK_OEM_PERIOD:             break;
    case VK_OEM_2:                  break;
    case VK_OEM_3:                  break;
    case VK_OEM_4:                  break;
    case VK_OEM_5:                  break;
    case VK_OEM_6:                  break;
    case VK_OEM_7:                  break;
    case VK_OEM_8:                  break;
    case VK_OEM_AX:                 break;
    case VK_OEM_102:                break;
    case VK_ICO_HELP:               break;
    case VK_ICO_00:                 break;
    case VK_PROCESSKEY:             return FW_KEY_PROCESS;
    case VK_ICO_CLEAR:              break;
    case VK_PACKET:                 break;
    case VK_OEM_RESET:              break;
    case VK_OEM_JUMP:               break;
    case VK_OEM_PA1:                break;
    case VK_OEM_PA2:                break;
    case VK_OEM_PA3:                break;
    case VK_OEM_WSCTRL:             break;
    case VK_OEM_CUSEL:              break;
    case VK_OEM_ATTN:               break;
    case VK_OEM_FINISH:             break;
    case VK_OEM_COPY:               break;
    case VK_OEM_AUTO:               break;
    case VK_OEM_ENLW:               break;
    case VK_OEM_BACKTAB:            break;
    case VK_ATTN:                   return FW_KEY_ATTN;
    case VK_CRSEL:                  return FW_KEY_CRSEL;
    case VK_EXSEL:                  return FW_KEY_EXSEL;
    case VK_EREOF:                  return FW_KEY_ERASE_EOF;
    case VK_PLAY:                   return FW_KEY_PLAY;
    case VK_ZOOM:                   return FW_KEY_ZOOM;
    case VK_NONAME:                 break;
    case VK_PA1:                    break;
    case VK_OEM_CLEAR:              return FW_KEY_CLEAR;
    default:                        break;
    }

    // Translate to Unicode.

    S32 cp = MapVirtualKeyW(vkey, 2);

    // Translate dead keys to combining marks.

    switch (cp)
    {
    case 0x80000060: cp = 0x0300; break; // FW_KEY_DEAD_GRAVE
    case 0x800000B4: cp = 0x0301; break; // FW_KEY_DEAD_ACUTE
    case 0x8000005E: cp = 0x0302; break; // FW_KEY_DEAD_CIRCUMFLEX
    case 0x8000007E: cp = 0x0303; break; // FW_KEY_DEAD_TILDE
    case 0x800000AF: cp = 0x0304; break; // FW_KEY_DEAD_MACRON
    case 0x800002D8: cp = 0x0306; break; // FW_KEY_DEAD_BREVE
    case 0x800002D9: cp = 0x0307; break; // FW_KEY_DEAD_ABOVE_DOT
    case 0x800000A8: cp = 0x0308; break; // FW_KEY_DEAD_DIAERESIS
    case 0x800002DA: cp = 0x030A; break; // FW_KEY_DEAD_ABOVE_RING
    case 0x800002DD: cp = 0x030B; break; // FW_KEY_DEAD_DOUBLE_ACUTE
    case 0x800002C7: cp = 0x030C; break; // FW_KEY_DEAD_CARON
    case 0x800000B8: cp = 0x0327; break; // FW_KEY_DEAD_CEDILLA
    case 0x800002DB: cp = 0x0328; break; // FW_KEY_DEAD_OGONEK
    case 0x8000037A: cp = 0x0345; break; // FW_KEY_DEAD_IOTA
    case 0x8000309B: cp = 0x3099; break; // FW_KEY_DEAD_VOICED
    case 0x8000309C: cp = 0x309A; break; // FW_KEY_DEAD_SEMIVOICED
    default: break;
    }

    // Convert to string.

    return (cp) ? unicodeToKey(cp) : "";
}

//------------------------------------------------------------------------
