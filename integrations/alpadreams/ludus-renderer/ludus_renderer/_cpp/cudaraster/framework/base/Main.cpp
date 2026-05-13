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

#include "base/Main.hpp"
#include "base/DLLImports.hpp"
#include "base/Thread.hpp"
#include "gui/Window.hpp"
#include "gpu/GLContext.hpp"
#include "gpu/CudaModule.hpp"
#include "gpu/CudaCompiler.hpp"

#include <crtdbg.h>
#include <conio.h>

using namespace FW;

//------------------------------------------------------------------------

int         FW::argc            = 0;
char**      FW::argv            = NULL;
int         FW::exitCode        = 0;

static bool	s_enableLeakCheck   = true;

//------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    // Store arguments.

    FW::argc = argc;
    FW::argv = argv;

    // Force the main thread to run on a single core.

    SetThreadAffinityMask(GetCurrentThread(), 1);

    // Initialize CRTDBG.

#if FW_DEBUG
    int flag = 0;
    flag |= _CRTDBG_ALLOC_MEM_DF;       // use allocation guards
//  flag |= _CRTDBG_CHECK_ALWAYS_DF;    // check whole memory on each alloc
//  flag |= _CRTDBG_DELAY_FREE_MEM_DF;  // keep freed memory and check that it isn't written
    _CrtSetDbgFlag(flag);

//  _CrtSetBreakAlloc(64);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif

    // Initialize the application.

    Thread::getCurrent();
    FW::init();
    failIfError();

    // Message loop.

    while (Window::getNumOpen())
    {
        // Wait for a message.

        MSG msg;
        if (!PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            Window::realizeAll();
            GetMessage(&msg, NULL, 0, 0);
        }

        // Process the message.

        TranslateMessage(&msg);
        DispatchMessage(&msg);

        // Nesting level was not restored => something fishy is going on.

        if (incNestingLevel(0) != 0)
        {
            fail(
                "Unhandled access violation detected!\n"
                "\n"
                "To get a stack trace, try the following:\n"
                "- Select \"Debug / Exceptions...\" in Visual Studio.\n"
                "- Expand the \"Win32 Exceptions\" category.\n"
                "- Check the \"Thrown\" box for \"Access violation\".\n"
                "- Re-run the application under debugger (F5).");
        }
    }

    // Clean up.

    failIfError();
    CudaCompiler::staticDeinit();
    CudaModule::staticDeinit();
    GLContext::staticDeinit();
    Window::staticDeinit();
    deinitDLLImports();
    profileEnd(false);
    failIfError();

    while (hasLogFile())
        popLogFile();

    delete Thread::getCurrent();

    // Dump memory leaks.

#if FW_DEBUG
    if (s_enableLeakCheck && _CrtDumpMemoryLeaks())
    {
        printf("Press any key to continue . . . ");
        _getch();
        printf("\n");
    }
#endif

    return exitCode;
}

//------------------------------------------------------------------------

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
    FW_UNREF(hInstance);
    FW_UNREF(hPrevInstance);
    FW_UNREF(lpCmdLine); // TODO: Parse this to argc/argv.
    FW_UNREF(nShowCmd);
    return main(0, NULL);
}

//------------------------------------------------------------------------

void FW::disableLeakCheck(void)
{
    s_enableLeakCheck = false;
}

//------------------------------------------------------------------------
