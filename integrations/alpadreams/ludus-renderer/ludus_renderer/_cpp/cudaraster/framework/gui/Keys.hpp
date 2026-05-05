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
#include "base/Defs.hpp"

namespace FW
{
//------------------------------------------------------------------------
// Custom key identifier strings.
//------------------------------------------------------------------------

#define FW_KEY_NONE                 ""
#define FW_KEY_MOUSE_LEFT           "!MouseLeft"            // Left (primary) mouse button.
#define FW_KEY_MOUSE_RIGHT          "!MouseRight"           // Right (secondary) mouse button.
#define FW_KEY_MOUSE_MIDDLE         "!MouseMiddle"          // Middle (ternary) mouse button.
#define FW_KEY_WHEEL_UP             "!WheelUp"              // Mouse wheel up.
#define FW_KEY_WHEEL_DOWN           "!WheelDown"            // Mouse wheel down.
#define FW_KEY_LEFT_SOFTKEY         "LeftSoftKey"           // Left softkey.
#define FW_KEY_RIGHT_SOFTKEY        "RightSoftKey"          // Right softkey.

//------------------------------------------------------------------------
// DOM3 key identifier strings.
// http://www.w3.org/TR/DOM-Level-3-Events/keyset.html
//------------------------------------------------------------------------

#define FW_KEY_ACCEPT               "Accept"                // The Accept (Commit) key.
#define FW_KEY_AGAIN                "Again"                 // The Again key.
#define FW_KEY_ALL_CANDIDATES       "AllCandidates"         // The All Candidates key.
#define FW_KEY_ALPHANUMERIC         "Alphanumeric"          // The Alphanumeric key.
#define FW_KEY_ALT                  "Alt"                   // The Alt (Menu) key.
#define FW_KEY_ALT_GRAPH            "AltGraph"              // The Alt-Graph key.
#define FW_KEY_APPS                 "Apps"                  // The Application key.
#define FW_KEY_ATTN                 "Attn"                  // The ATTN key.
#define FW_KEY_BROWSER_BACK         "BrowserBack"           // The Browser Back key.
#define FW_KEY_BROWSER_FAVORITES    "BrowserFavorites"      // The Browser Favorites key.
#define FW_KEY_BROWSER_FORWARD      "BrowserForward"        // The Browser Forward key.
#define FW_KEY_BROWSER_HOME         "BrowserHome"           // The Browser Home key.
#define FW_KEY_BROWSER_REFRESH      "BrowserRefresh"        // The Browser Refresh key.
#define FW_KEY_BROWSER_SEARCH       "BrowserSearch"         // The Browser Search key.
#define FW_KEY_BROWSER_STOP         "BrowserStop"           // The Browser Stop key.
#define FW_KEY_CAPS_LOCK            "CapsLock"              // The Caps Lock (Capital) key.
#define FW_KEY_CLEAR                "Clear"                 // The Clear key.
#define FW_KEY_CODE_INPUT           "CodeInput"             // The Code Input key.
#define FW_KEY_COMPOSE              "Compose"               // The Compose key.
#define FW_KEY_CONTROL              "Control"               // The Control (Ctrl) key.
#define FW_KEY_CRSEL                "Crsel"                 // The Crsel key.
#define FW_KEY_CONVERT              "Convert"               // The Convert key.
#define FW_KEY_COPY                 "Copy"                  // The Copy key.
#define FW_KEY_CUT                  "Cut"                   // The Cut key.
#define FW_KEY_DOWN                 "Down"                  // The Down Arrow key.
#define FW_KEY_END                  "End"                   // The End key.
#define FW_KEY_ENTER                "Enter"                 // The Enter key. Note: This key identifier is also used for the Return (Macintosh numpad) key.
#define FW_KEY_ERASE_EOF            "EraseEof"              // The Erase EOF key.
#define FW_KEY_EXECUTE              "Execute"               // The Execute key.
#define FW_KEY_EXSEL                "Exsel"                 // The Exsel key.
#define FW_KEY_F1                   "F1"                    // The F1 key.
#define FW_KEY_F2                   "F2"                    // The F2 key.
#define FW_KEY_F3                   "F3"                    // The F3 key.
#define FW_KEY_F4                   "F4"                    // The F4 key.
#define FW_KEY_F5                   "F5"                    // The F5 key.
#define FW_KEY_F6                   "F6"                    // The F6 key.
#define FW_KEY_F7                   "F7"                    // The F7 key.
#define FW_KEY_F8                   "F8"                    // The F8 key.
#define FW_KEY_F9                   "F9"                    // The F9 key.
#define FW_KEY_F10                  "F10"                   // The F10 key.
#define FW_KEY_F11                  "F11"                   // The F11 key.
#define FW_KEY_F12                  "F12"                   // The F12 key.
#define FW_KEY_F13                  "F13"                   // The F13 key.
#define FW_KEY_F14                  "F14"                   // The F14 key.
#define FW_KEY_F15                  "F15"                   // The F15 key.
#define FW_KEY_F16                  "F16"                   // The F16 key.
#define FW_KEY_F17                  "F17"                   // The F17 key.
#define FW_KEY_F18                  "F18"                   // The F18 key.
#define FW_KEY_F19                  "F19"                   // The F19 key.
#define FW_KEY_F20                  "F20"                   // The F20 key.
#define FW_KEY_F21                  "F21"                   // The F21 key.
#define FW_KEY_F22                  "F22"                   // The F22 key.
#define FW_KEY_F23                  "F23"                   // The F23 key.
#define FW_KEY_F24                  "F24"                   // The F24 key.
#define FW_KEY_FINAL_MODE           "FinalMode"             // The Final Mode (Final) key used on some asian keyboards.
#define FW_KEY_FIND                 "Find"                  // The Find key.
#define FW_KEY_FULL_WIDTH           "FullWidth"             // The Full-Width Characters key.
#define FW_KEY_HALF_WIDTH           "HalfWidth"             // The Half-Width Characters key.
#define FW_KEY_HANGUL_MODE          "HangulMode"            // The Hangul (Korean characters) Mode key.
#define FW_KEY_HANJA_MODE           "HanjaMode"             // The Hanja (Korean characters) Mode key.
#define FW_KEY_HELP                 "Help"                  // The Help key.
#define FW_KEY_HIRAGANA             "Hiragana"              // The Hiragana (Japanese Kana characters) key.
#define FW_KEY_HOME                 "Home"                  // The Home key.
#define FW_KEY_INSERT               "Insert"                // The Insert (Ins) key.
#define FW_KEY_JAPANESE_HIRAGANA    "JapaneseHiragana"      // The Japanese-Hiragana key.
#define FW_KEY_JAPANESE_KATAKANA    "JapaneseKatakana"      // The Japanese-Katakana key.
#define FW_KEY_JAPANESE_ROMAJI      "JapaneseRomaji"        // The Japanese-Romaji key.
#define FW_KEY_JUNJA_MODE           "JunjaMode"             // The Junja Mode key.
#define FW_KEY_KANA_MODE            "KanaMode"              // The Kana Mode (Kana Lock) key.
#define FW_KEY_KANJI_MODE           "KanjiMode"             // The Kanji (Japanese name for ideographic characters of Chinese origin) Mode key.
#define FW_KEY_KATAKANA             "Katakana"              // The Katakana (Japanese Kana characters) key.
#define FW_KEY_LAUNCH_APPLICATION1  "LaunchApplication1"    // The Start Application One key.
#define FW_KEY_LAUNCH_APPLICATION2  "LaunchApplication2"    // The Start Application Two key.
#define FW_KEY_LAUNCH_MAIL          "LaunchMail"            // The Start Mail key.
#define FW_KEY_LEFT                 "Left"                  // The Left Arrow key.
#define FW_KEY_META                 "Meta"                  // The Meta key.
#define FW_KEY_MEDIA_NEXT_TRACK     "MediaNextTrack"        // The Media Next Track key.
#define FW_KEY_MEDIA_PLAY_PAUSE     "MediaPlayPause"        // The Media Play Pause key.
#define FW_KEY_MEDIA_PREVIOUS_TRACK "MediaPreviousTrack"    // The Media Previous Track key.
#define FW_KEY_MEDIA_STOP           "MediaStop"             // The Media Stok key.
#define FW_KEY_MODE_CHANGE          "ModeChange"            // The Mode Change key.
#define FW_KEY_NONCONVERT           "Nonconvert"            // The Nonconvert (Don't Convert) key.
#define FW_KEY_NUM_LOCK             "NumLock"               // The Num Lock key.
#define FW_KEY_PAGE_DOWN            "PageDown"              // The Page Down (Next) key.
#define FW_KEY_PAGE_UP              "PageUp"                // The Page Up key.
#define FW_KEY_PASTE                "Paste"                 // The Paste key.
#define FW_KEY_PAUSE                "Pause"                 // The Pause key.
#define FW_KEY_PLAY                 "Play"                  // The Play key.
#define FW_KEY_PREVIOUS_CANDIDATE   "PreviousCandidate"     // The Previous Candidate function key.
#define FW_KEY_PRINT_SCREEN         "PrintScreen"           // The Print Screen (PrintScrn, SnapShot) key.
#define FW_KEY_PROCESS              "Process"               // The Process key.
#define FW_KEY_PROPS                "Props"                 // The Props key.
#define FW_KEY_RIGHT                "Right"                 // The Right Arrow key.
#define FW_KEY_ROMAN_CHARACTERS     "RomanCharacters"       // The Roman Characters function key.
#define FW_KEY_SCROLL               "Scroll"                // The Scroll Lock key.
#define FW_KEY_SELECT               "Select"                // The Select key.
#define FW_KEY_SELECT_MEDIA         "SelectMedia"           // The Select Media key.
#define FW_KEY_SHIFT                "Shift"                 // The Shift key.
#define FW_KEY_STOP                 "Stop"                  // The Stop key.
#define FW_KEY_UP                   "Up"                    // The Up Arrow key.
#define FW_KEY_UNDO                 "Undo"                  // The Undo key.
#define FW_KEY_VOLUME_DOWN          "VolumeDown"            // The Volume Down key.
#define FW_KEY_VOLUME_MUTE          "VolumeMute"            // The Volume Mute key.
#define FW_KEY_VOLUME_UP            "VolumeUp"              // The Volume Up key.
#define FW_KEY_WIN                  "Win"                   // The Windows Logo key.
#define FW_KEY_ZOOM                 "Zoom"                  // The Zoom key.
#define FW_KEY_BACKSPACE            "U+0008"                // The Backspace (Back) key.
#define FW_KEY_TAB                  "U+0009"                // The Horizontal Tabulation (Tab) key.
#define FW_KEY_CANCEL               "U+0018"                // The Cancel key.
#define FW_KEY_ESCAPE               "U+001B"                // The Escape (Esc) key.
#define FW_KEY_SPACE                "U+0020"                // The Space (Spacebar) key.
#define FW_KEY_EXCLAMATION          "U+0021"                // The Exclamation Mark (Factorial, Bang) key (!).
#define FW_KEY_DOUBLE_QUOTE         "U+0022"                // The Quotation Mark (Quote Double) key (").
#define FW_KEY_HASH                 "U+0023"                // The Number Sign (Pound Sign, Hash, Crosshatch, Octothorpe) key (#).
#define FW_KEY_DOLLAR               "U+0024"                // The Dollar Sign (milreis, escudo) key ($).
#define FW_KEY_AMPERSAND            "U+0026"                // The Ampersand key (&).
#define FW_KEY_SINGLE_QUOTE         "U+0027"                // The Apostrophe (Apostrophe-Quote, APL Quote) key (').
#define FW_KEY_LEFT_PARENTHESIS     "U+0028"                // The Left Parenthesis (Opening Parenthesis) key (().
#define FW_KEY_RIGHT_PARENTHESIS    "U+0029"                // The Right Parenthesis (Closing Parenthesis) key ()).
#define FW_KEY_ASTERISK             "U+002A"                // The Asterisk (Star) key (*).
#define FW_KEY_PLUS                 "U+002B"                // The Plus Sign (Plus) key (+).
#define FW_KEY_COMMA                "U+002C"                // The Comma (decimal separator) sign key (,).
#define FW_KEY_MINUS                "U+002D"                // The Hyphen-minus (hyphen or minus sign) key (-).
#define FW_KEY_PERIOD               "U+002E"                // The Full Stop (period, dot, decimal point) key (.).
#define FW_KEY_SLASH                "U+002F"                // The Solidus (slash, virgule, shilling) key (/).
#define FW_KEY_0                    "U+0030"                // The Digit Zero key (0).
#define FW_KEY_1                    "U+0031"                // The Digit One key (1).
#define FW_KEY_2                    "U+0032"                // The Digit Two key (2).
#define FW_KEY_3                    "U+0033"                // The Digit Three key (3).
#define FW_KEY_4                    "U+0034"                // The Digit Four key (4).
#define FW_KEY_5                    "U+0035"                // The Digit Five key (5).
#define FW_KEY_6                    "U+0036"                // The Digit Six key (6).
#define FW_KEY_7                    "U+0037"                // The Digit Seven key (7).
#define FW_KEY_8                    "U+0038"                // The Digit Eight key (8).
#define FW_KEY_9                    "U+0039"                // The Digit Nine key (9).
#define FW_KEY_COLON                "U+003A"                // The Colon key (:).
#define FW_KEY_SEMICOLON            "U+003B"                // The Semicolon key (;).
#define FW_KEY_LESS_THAN            "U+003C"                // The Less-Than Sign key (<).
#define FW_KEY_EQUALS               "U+003D"                // The Equals Sign key (=).
#define FW_KEY_GREATER_THAN         "U+003E"                // The Greater-Than Sign key (>).
#define FW_KEY_QUESTION             "U+003F"                // The Question Mark key (?).
#define FW_KEY_AT                   "U+0040"                // The Commercial At (@) key.
#define FW_KEY_A                    "U+0041"                // The Latin Capital Letter A key (A).
#define FW_KEY_B                    "U+0042"                // The Latin Capital Letter B key (B).
#define FW_KEY_C                    "U+0043"                // The Latin Capital Letter C key (C).
#define FW_KEY_D                    "U+0044"                // The Latin Capital Letter D key (D).
#define FW_KEY_E                    "U+0045"                // The Latin Capital Letter E key (E).
#define FW_KEY_F                    "U+0046"                // The Latin Capital Letter F key (F).
#define FW_KEY_G                    "U+0047"                // The Latin Capital Letter G key (G).
#define FW_KEY_H                    "U+0048"                // The Latin Capital Letter H key (H).
#define FW_KEY_I                    "U+0049"                // The Latin Capital Letter I key (I).
#define FW_KEY_J                    "U+004A"                // The Latin Capital Letter J key (J).
#define FW_KEY_K                    "U+004B"                // The Latin Capital Letter K key (K).
#define FW_KEY_L                    "U+004C"                // The Latin Capital Letter L key (L).
#define FW_KEY_M                    "U+004D"                // The Latin Capital Letter M key (M).
#define FW_KEY_N                    "U+004E"                // The Latin Capital Letter N key (N).
#define FW_KEY_O                    "U+004F"                // The Latin Capital Letter O key (O).
#define FW_KEY_P                    "U+0050"                // The Latin Capital Letter P key (P).
#define FW_KEY_Q                    "U+0051"                // The Latin Capital Letter Q key (Q).
#define FW_KEY_R                    "U+0052"                // The Latin Capital Letter R key (R).
#define FW_KEY_S                    "U+0053"                // The Latin Capital Letter S key (S).
#define FW_KEY_T                    "U+0054"                // The Latin Capital Letter T key (T).
#define FW_KEY_U                    "U+0055"                // The Latin Capital Letter U key (U).
#define FW_KEY_V                    "U+0056"                // The Latin Capital Letter V key (V).
#define FW_KEY_W                    "U+0057"                // The Latin Capital Letter W key (W).
#define FW_KEY_X                    "U+0058"                // The Latin Capital Letter X key (X).
#define FW_KEY_Y                    "U+0059"                // The Latin Capital Letter Y key (Y).
#define FW_KEY_Z                    "U+005A"                // The Latin Capital Letter Z key (Z).
#define FW_KEY_LEFT_SQUARE_BRACKET  "U+005B"                // The Left Square Bracket (Opening Square Bracket) key ([).
#define FW_KEY_BACKSLASH            "U+005C"                // The Reverse Solidus (Backslash) key (\).
#define FW_KEY_RIGHT_SQUARE_BRACKET "U+005D"                // The Right Square Bracket (Closing Square Bracket) key (]).
#define FW_KEY_CIRCUMFLEX           "U+005E"                // The Circumflex Accent key (^).
#define FW_KEY_UNDERSCORE           "U+005F"                // The Low Sign (Spacing Underscore, Underscore) key (_).
#define FW_KEY_BACK_QUOTE           "U+0060"                // The Grave Accent (Back Quote) key (`).
#define FW_KEY_LEFT_CURLY_BRACKET   "U+007B"                // The Left Curly Bracket (Opening Curly Bracket, Opening Brace, Brace Left) key ({).
#define FW_KEY_VERTICAL_BAR         "U+007C"                // The Vertical Line (Vertical Bar, Pipe) key (|).
#define FW_KEY_RIGHT_CURLY_BRACKET  "U+007D"                // The Right Curly Bracket (Closing Curly Bracket, Closing Brace, Brace Right) key (}).
#define FW_KEY_DELETE               "U+007F"                // The Delete (Del) Key.
#define FW_KEY_INVERTED_EXCLAMATION "U+00A1"                // The Inverted Exclamation Mark key (¡).
#define FW_KEY_DEAD_GRAVE           "U+0300"                // The Combining Grave Accent (Greek Varia, Dead Grave) key.
#define FW_KEY_DEAD_ACUTE           "U+0301"                // The Combining Acute Accent (Stress Mark, Greek Oxia, Tonos, Dead Eacute) key.
#define FW_KEY_DEAD_CIRCUMFLEX      "U+0302"                // The Combining Circumflex Accent (Hat, Dead Circumflex) key.
#define FW_KEY_DEAD_TILDE           "U+0303"                // The Combining Tilde (Dead Tilde) key.
#define FW_KEY_DEAD_MACRON          "U+0304"                // The Combining Macron (Long, Dead Macron) key.
#define FW_KEY_DEAD_BREVE           "U+0306"                // The Combining Breve (Short, Dead Breve) key.
#define FW_KEY_DEAD_ABOVE_DOT       "U+0307"                // The Combining Dot Above (Derivative, Dead Above Dot) key.
#define FW_KEY_DEAD_DIAERESIS       "U+0308"                // The Combining Diaeresis (Double Dot Abode, Umlaut, Greek Dialytika, Double Derivative, Dead Diaeresis) key.
#define FW_KEY_DEAD_ABOVE_RING      "U+030A"                // The Combining Ring Above (Dead Above Ring) key.
#define FW_KEY_DEAD_DOUBLE_ACUTE    "U+030B"                // The Combining Double Acute Accent (Dead Doubleacute) key.
#define FW_KEY_DEAD_CARON           "U+030C"                // The Combining Caron (Hacek, V Above, Dead Caron) key.
#define FW_KEY_DEAD_CEDILLA         "U+0327"                // The Combining Cedilla (Dead Cedilla) key.
#define FW_KEY_DEAD_OGONEK          "U+0328"                // The Combining Ogonek (Nasal Hook, Dead Ogonek) key.
#define FW_KEY_DEAD_IOTA            "U+0345"                // The Combining Greek Ypogegrammeni (Greek Non-Spacing Iota Below, Iota Subscript, Dead Iota) key.
#define FW_KEY_EURO                 "U+20AC"                // The Euro Currency Sign key (€).
#define FW_KEY_DEAD_VOICED          "U+3099"                // The Combining Katakana-Hiragana Voiced Sound Mark (Dead Voiced Sound) key.
#define FW_KEY_DEAD_SEMIVOICED      "U+309A"                // The Combining Katakana-Hiragana Semi-Voiced Sound Mark (Dead Semivoiced Sound) key.

//------------------------------------------------------------------------

String  unicodeToKey    (S32 cp);
S32     keyToUnicode    (const String& key);
String  vkeyToKey       (U32 vkey);

//------------------------------------------------------------------------
}
