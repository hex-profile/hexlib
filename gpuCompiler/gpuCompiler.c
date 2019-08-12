#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "cmdLine/cmdLine.h"
#include "parseTools/parseTools.h"
#include "shared/gpuModule.h"
#include "formatting/formatModifiers.h"
#include "numbers/float/floatType.h"
#include "storage/rememberCleanup.h"
#include "userOutput/errorLogEx.h"
#include "userOutput/msgLog.h"
#include "userOutput/printMsg.h"
#include "checkHeap.h"
#include "formattedOutput/sprintMsg.h"
#include "formattedOutput/textFiles.h"
#include "formattedOutput/userOutputThunks.h"
#include "formattedOutput/logToStlConsole.h"
#include "fileToolsImpl/fileToolsImpl.h"
#include "errorLog/foreignErrorBlock.h"
#include "formattedOutput/requireMsg.h"

using namespace std;

//================================================================
//
// CompilerKit
//
//================================================================

KIT_COMBINE4(CompilerKit, ErrorLogKit, MsgLogKit, ErrorLogExKit, FileToolsKit);

//================================================================
//
// splitPath
//
//================================================================

void splitPath(StlString path, StlString& pathDir, StlString& fileName, StlString& fileExt)
{
    using Size = StlString::size_type;
    Size nullPos = StlString::npos;

    Size pos1 = path.rfind('\\');
    Size pos2 = path.rfind('/');

    if (pos1 == nullPos) pos1 = 0;
    if (pos2 == nullPos) pos2 = 0;
    Size lastSlash = max(pos1, pos2);

    pathDir = path.substr(0, lastSlash);
    StlString file = path.substr(lastSlash ? lastSlash+1 : 0);

    Size dotPos = file.rfind('.');
    fileName = file.substr(0, dotPos);

    fileExt = dotPos == nullPos ? "" : file.substr(dotPos+1);
}

//================================================================
//
// stringBeginsWith
// stringEndsWith
//
//================================================================

bool stringBeginsWith(const StlString& str, const StlString& prefix)
{
    return str.substr(0, prefix.size()) == prefix;
}

bool stringEndsWith(const StlString& str, const StlString& suffix)
{
    return
        str.size() >= suffix.size() &&
        str.substr(str.size() - suffix.size()) == suffix;
}

//================================================================
//
// filenameToCString
//
// Converts slashes to double slashes.
//
//================================================================

StlString filenameToCString(const CharType* strBegin, const CharType* strEnd)
{
    StlString result;

    const CharType* ptr = strBegin;
    const CharType* end = strEnd;

    for (;;)
    {
        const CharType* normBegin = ptr;

        while (ptr != end && *ptr != '\\')
            ++ptr;

        result.append(normBegin, ptr);

        if (ptr == end)
            break;

        result.append(CT("\\\\"));
        ++ptr;
    }

    return result;
}

//----------------------------------------------------------------

inline StlString filenameToCString(const StlString& str)
{
    return filenameToCString(str.data(), str.data() + str.size());
}

//================================================================
//
// runProcess (Linux)
//
//================================================================

#if defined(__linux__)

stdbool runProcess(StlString cmdLine, stdPars(CompilerKit))
{
    int status = system(cmdLine.c_str());

    if_not (status == 0)
    {
        printMsg(kit.msgLog, STR("Cannot launch %0"), cmdLine, msgErr);
        return false;
    }

    returnTrue;
}

#endif

//================================================================
//
// runProcess (Windows)
//
//================================================================

#if defined(_WIN32)

stdbool runProcess(StlString cmdLine, stdPars(CompilerKit))
{
    STARTUPINFO si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    // printMsg(kit.msgLog, STR("Run: %0"), cmdLine, msgErr);

    bool createOk = CreateProcess(NULL, const_cast<char*>(cmdLine.c_str()), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi) != 0;

    if_not (createOk)
    {
        printMsg(kit.msgLog, STR("Cannot launch %0"), cmdLine, msgErr);
        returnFalse;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exitCode = 0;
    REQUIRE(GetExitCodeProcess(pi.hProcess, &exitCode) != 0);
    require(exitCode == 0); // success?

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    returnTrue;
}

#endif

//================================================================
//
// runProcess
//
//================================================================

stdbool runProcess(const vector<StlString>& args, stdPars(CompilerKit))
{
    StlString cmdline;

    for (size_t i = 0; i < args.size(); ++i)
    {
        cmdLine::convertToArg(args[i], cmdline);
        cmdline += CT(" ");
    }

    ////

    vector<StlString> testArg;
    cmdLine::parseCmdLine(cmdline, testArg);
    REQUIRE(testArg == args);

    ////

    require(runProcess(cmdline, stdPass));

    returnTrue;
}

//================================================================
//
// parseClArgs
//
//================================================================

stdbool parseClArgs
(
    const vector<StlString>& args,
    vector<StlString>& includes,
    vector<StlString>& defines,
    vector<StlString>& cppFiles,
    vector<StlString>& cudaFiles,
    StlString& outputDir,
    StlString& outputFile,
    vector<StlString>& otherArgs,
    stdPars(CompilerKit)
)
{
    int prevOption = 0; // 0, 'I', 'D'
    outputDir = CT("");
    outputFile = CT("");

    ////

    for (size_t i = 0; i < args.size(); ++i)
    {
        const StlString& s = args[i];

        ////

        if (s.substr(0, 1) == CT("/") || s.substr(0, 1) == CT("-"))
        {
            StlString option = s.substr(1);
            // printMsg(kit.msgLog, STR("Found option %0"), option);

            if (option.substr(0, 1) == CT("I"))
            {
                prevOption = 'I';

                StlString body = option.substr(1);

                if (body.size())
                {
                    includes.push_back(body);
                    prevOption = 0;
                }
            }
            else if (option.substr(0, 1) == CT("D"))
            {
                prevOption = 'D';

                StlString body = option.substr(1);

                if (body.size())
                {
                    // printMsg(kit.msgLog, STR("Found define %0"), body);
                    defines.push_back(body);
                    prevOption = 0;
                }
            }
            else if (option.substr(0, 2) == CT("Fo"))
            {
                StlString outputOption = option.substr(2);

                StlString tmpDir, tmpName, tmpExt;
                splitPath(outputOption, tmpDir, tmpName, tmpExt);

                if (tmpExt.size())
                    {outputDir = tmpDir; outputFile = outputOption;}
                else
                    {outputDir = outputOption; outputFile = CT("");}
            }
            else
            {
                prevOption = 0;
                otherArgs.push_back(s);
            }
        }
        else
        {
            if (prevOption == 'I')
                includes.push_back(s);
            else if (prevOption == 'D')
                defines.push_back(s);
            else if (stringEndsWith(s, CT(".cxx")))
                cudaFiles.push_back(s);
            else if (stringEndsWith(s, CT(".c")) || stringEndsWith(s, CT(".cpp")))
                cppFiles.push_back(s);
            else
                otherArgs.push_back(s);

            prevOption = 0;
        }
    }

    returnTrue;
}

//================================================================
//
// prepareForDeviceCompilation
//
// Appends #ifdef __CUDA_ARCH__
//
//================================================================

stdbool prepareForDeviceCompilation(const StlString& inputName, const StlString& outputName, stdPars(CompilerKit))
{
    InputTextFile<CharType> inputStream;
    require(inputStream.open(inputName, stdPass));

    OutputTextFile outputStream;
    require(outputStream.open(outputName, stdPass));

    ////

    printMsg(outputStream, STR("#ifdef __CUDA_ARCH__"));
    printMsg(outputStream, STR(""));

    printMsg(outputStream, STR("#line 1 \"%0\""), filenameToCString(inputName));

    StlString tmpStr;

    while (inputStream.getLine(tmpStr, stdPass))
        printMsg(outputStream, STR("%0"), tmpStr);

    require(inputStream.eof());

    printMsg(outputStream, STR(""));
    printMsg(outputStream, STR("#endif"));

    require(outputStream.flush(stdPass));

    returnTrue;
}

//================================================================
//
// lineIsSpace
//
//================================================================

bool lineIsSpace(const CharType* strBeg, const CharType* strEnd)
{
    for (const CharType* ptr = strBeg; ptr != strEnd; ++ptr)
        ensure(isAnySpace(*ptr));

    return true;
}

//================================================================
//
// lineIsPreprocessorLineDirective
//
//================================================================

bool lineIsPreprocessorLineDirective(const CharType* strBeg, const CharType* strEnd)
{
    const CharType* ptr = strBeg;
    const CharType* end = strEnd;

    skipSpaceTab(ptr, end);

    ensure(skipText(ptr, end, STR("#")));
    skipSpaceTab(ptr, end);

    ensure(skipText(ptr, end, STR("line")));

    return true;
}

//================================================================
//
// lineIsSignificant
//
//================================================================

bool lineIsSignificant(const CharType* strBeg, const CharType* strEnd)
{
    return
        !lineIsSpace(strBeg, strEnd) &&
        !lineIsPreprocessorLineDirective(strBeg, strEnd);
}

//================================================================
//
// sourcesAreIdentical
//
//================================================================

bool sourcesAreIdentical(const CharType* oldPtr, size_t oldSize, const CharType* newPtr, size_t newSize)
{
    using namespace std;

    //
    // Quick comparison
    //

    if (oldSize == newSize)
    {
        if (memcmp(oldPtr, newPtr, oldSize * sizeof(CharType)) == 0)
            return true;
    }

    //
    // Parse lines
    //

    const CharType* oldEnd = oldPtr + oldSize;
    const CharType* newEnd = newPtr + newSize;

    //
    //
    //

    for (;;)
    {

        const CharType* oldLineBeg = 0;
        const CharType* oldLineEnd = 0;

        bool oldOk = getNextLine(oldPtr, oldEnd, oldLineBeg, oldLineEnd);

        while (oldOk && !lineIsSignificant(oldLineBeg, oldLineEnd))
            oldOk = getNextLine(oldPtr, oldEnd, oldLineBeg, oldLineEnd);

        ////

        const CharType* newLineBeg = 0;
        const CharType* newLineEnd = 0;

        bool newOk = getNextLine(newPtr, newEnd, newLineBeg, newLineEnd);

        while (newOk && !lineIsSignificant(newLineBeg, newLineEnd))
            newOk = getNextLine(newPtr, newEnd, newLineBeg, newLineEnd);

        ////

        if (!oldOk || !newOk) // cant read
        {
            ensure(!oldOk && !newOk);
            break;
        }

        ////

        size_t oldLineSize = oldLineEnd - oldLineBeg;
        size_t newLineSize = newLineEnd - newLineBeg;

        ensure(oldLineSize == newLineSize);
        ensure(memcmp(oldLineBeg, newLineBeg, oldLineSize * sizeof(CharType)) == 0);
    }

    ////

    return true;
}

//================================================================
//
// extractKernelNames
//
// Finds all kernel names in the line
//
//================================================================

void extractKernelNames(const CharType* ptr, const CharType* end, vector<StlString>& kernelNames)
{
    skipSpaceTab(ptr, end);

    for (;;)
    {
        //
        // try to find __declspec starting from the current place (advance or exit!)
        //

        while (ptr != end && *ptr != '_')
            ++ptr;

        if (ptr == end)
            break;

        ++ptr;

        //
        // try to parse from the current point
        //

        breakBlock_
        {
            breakRequire(skipTextThenSpaceTab(ptr, end, STR("_declspec")));
            breakRequire(skipTextThenSpaceTab(ptr, end, STR("(")));
            breakRequire(skipTextThenSpaceTab(ptr, end, STR("__global__")));
            breakRequire(skipTextThenSpaceTab(ptr, end, STR(")")));
            breakRequire(skipTextThenSpaceTab(ptr, end, STR("void")));

            ////

            const CharType* launchBoundsPlace = ptr;

            bool launchBoundsDetected =
                skipTextThenSpaceTab(ptr, end, STR("__declspec")) &&
                skipTextThenSpaceTab(ptr, end, STR("(")) &&
                skipTextThenSpaceTab(ptr, end, STR("launch_bounds"));

            if_not (launchBoundsDetected)
                ptr = launchBoundsPlace;
            else
            {
                breakRequire(skipTextThenSpaceTab(ptr, end, STR("(")));

                int32 scopeLevel = 2;

                for (;;)
                {
                    while (ptr != end && *ptr != '(' && *ptr != ')')
                        ++ptr;

                    if (ptr == end)
                        break;

                    if (*ptr == '(') ++scopeLevel;
                    if (*ptr == ')') --scopeLevel;

                    ++ptr;
                    skipSpaceTab(ptr, end);

                    if (scopeLevel <= 0)
                        break;
                }

                breakRequire(scopeLevel == 0);
            }

            ////

            const CharType* identBegin = ptr;
            breakRequire(skipIdent(ptr, end));
            const CharType* identEnd = ptr;

            kernelNames.push_back(StlString(identBegin, identEnd));
        }
    }
}

//================================================================
//
// findNextWordOnLetter
//
//================================================================

inline bool findNextWordOnLetter(const CharType*& ptr, const CharType* end, CharType letter)
{
    for (;;)
    {
        skipSpaceTab(ptr, end);
        ensure(ptr != end);

        if (*ptr == letter)
            return true;

        skipNonSpaceCharacters(ptr, end);
    }

    return true;
}

//================================================================
//
// tryParseTextureDef
//
//================================================================

inline bool tryParseTextureDef(const CharType*& ptr, const CharType* end, vector<StlString>& samplerNames)
{
    const CharType* p = ptr;

    ensure(skipTextThenSpaceTab(p, end, STR("extern")));
    ensure(skipTextThenSpaceTab(p, end, STR("\"C\"")));
    ensure(skipTextThenSpaceTab(p, end, STR("{")));
    ensure(skipTextThenSpaceTab(p, end, STR("texture")));
    ensure(skipTextThenSpaceTab(p, end, STR("<")));

    //
    // Scan until all matching '>' are encountered
    //

    int32 braceLevel = 1;

    for (; p != end; ++p)
    {
        if (*p == '<') ++braceLevel;
        if (*p == '>') --braceLevel;
        if (braceLevel <= 0) break;
    }

    ensure(p != end && braceLevel == 0);
    ensure(skipTextThenSpaceTab(p, end, STR(">")));

    ////

    const CharType* identBegin = p;
    ensure(skipIdent(p, end));
    const CharType* identEnd = p;

    StlString samplerName(identBegin, identEnd);

    ////

    samplerNames.push_back(samplerName);
    ptr = p;

    return true;
}

//================================================================
//
// extractSamplerNames
//
//================================================================

inline void extractSamplerNames(const CharType* ptr, const CharType* end, vector<StlString>& samplerNames)
{
    for (;;)
    {

        //
        // Find next similar word: May NOT advance
        //

        if_not (findNextWordOnLetter(ptr, end, 'e'))
            break;

        //
        // Parse: Should always advance
        //

        if_not (tryParseTextureDef(ptr, end, samplerNames))
            ++ptr; // can do it because next word was found
    }
}

//================================================================
//
// extractKernelAndSamplerNames
//
//================================================================

void extractKernelAndSamplerNames(const CharType* filePtr, size_t fileSize, vector<StlString>& kernelNames, vector<StlString>& samplerNames)
{
    const CharType* fileEnd = filePtr + fileSize;

    for (;;)
    {
        const CharType* strBeg = 0;
        const CharType* strEnd = 0;

        if_not (getNextLine(filePtr, fileEnd, strBeg, strEnd))
            break;

        extractKernelNames(strBeg, strEnd, kernelNames);
        extractSamplerNames(strBeg, strEnd, samplerNames);
    }
}

//================================================================
//
// addTargetArch
//
//================================================================

void addTargetArch(vector<StlString>& nvccArgs, const StlString& platformArch)
{
    nvccArgs.push_back(CT("--gpu-architecture"));
    nvccArgs.push_back(platformArch);
}

//================================================================
//
// compileDevicePartToBin
//
//================================================================

stdbool compileDevicePartToBin
(
    const StlString& inputPath,
    const StlString& binPath,
    const StlString& asmPath,
    vector<StlString>& kernelNames,
    vector<StlString>& samplerNames,
    const vector<StlString>& includes,
    const vector<StlString>& defines,
    const StlString& platformArch,
    const StlString& platformBitness,
    stdPars(CompilerKit)
)
{
    StlString inputDir;
    StlString inputName;
    StlString inputExt;

    splitPath(inputPath, inputDir, inputName, inputExt);

    ////

    StlString binDir;
    StlString binName;
    StlString binExt;

    splitPath(binPath, binDir, binName, binExt);

    //----------------------------------------------------------------
    //
    // Prepare input file (surround with #ifdef __CUDA_ARCH__)
    //
    //----------------------------------------------------------------

    StlString cuInputPath = sprintMsg(STR("%0/%1.cu"), inputDir, inputName);
    REMEMBER_CLEANUP1(remove(cuInputPath.c_str()), const StlString&, cuInputPath);
    require(prepareForDeviceCompilation(inputPath, cuInputPath, stdPass));

    //----------------------------------------------------------------
    //
    // Preprocess with NVCC
    //
    //----------------------------------------------------------------

    StlString cupPath = sprintMsg(STR("%0/%1.cup"), binDir, binName);
    StlString cachedPath = sprintMsg(STR("%0/%1.src"), binDir, binName);

    {
        vector<StlString> nvccArgs;

        nvccArgs.push_back(CT("nvcc.exe"));

        nvccArgs.push_back(CT("-m") + platformBitness);

        nvccArgs.push_back(CT("-E"));

        for (size_t i = 0; i < includes.size(); ++i)
        {
            nvccArgs.push_back(CT("-I"));
            nvccArgs.push_back(includes[i]);
        }

        for (size_t i = 0; i < defines.size(); ++i)
        {
            nvccArgs.push_back(CT("-D"));
            nvccArgs.push_back(defines[i]);
        }

        nvccArgs.push_back(CT("-D"));
        nvccArgs.push_back(CT("HOSTCODE=0"));

        nvccArgs.push_back(CT("-D"));
        nvccArgs.push_back(CT("DEVCODE=1"));

        nvccArgs.push_back(CT("-o"));
        nvccArgs.push_back(cupPath);

        addTargetArch(nvccArgs, platformArch);

        nvccArgs.push_back(cuInputPath);

        if (0)
        {
            for (size_t i = 0; i < nvccArgs.size(); ++i)
                printMsg(kit.msgLog, STR("[NVCC %0] %1"), dec(i, 2), nvccArgs[i]);
        }

        require(runProcess(nvccArgs, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Read new preprocessed file
    //
    //----------------------------------------------------------------

    basic_ifstream<CharType> cupFile(cupPath.c_str(), ios_base::in | ios_base::binary | ios_base::ate );
    REQUIRE(!!cupFile); // for tellg
    size_t cupSize = (size_t) cupFile.tellg();
    cupFile.seekg(0, ios_base::beg);
    vector<CharType> cupData(clampMin<size_t>(cupSize, 1));
    cupFile.read(&cupData[0], cupSize);
    REQUIRE(!!cupFile);
    cupFile.close();

    //
    // Extract kernel and sampler names
    //

    extractKernelAndSamplerNames(&cupData[0], cupSize, kernelNames, samplerNames);

    //----------------------------------------------------------------
    //
    // Try to use cached version
    //
    //----------------------------------------------------------------

    if
    (
        kit.fileTools.fileExists(binPath.c_str()) &&
        kit.fileTools.fileExists(asmPath.c_str()) &&
        kit.fileTools.fileExists(cupPath.c_str()) &&
        kit.fileTools.fileExists(cachedPath.c_str())
    )
    {
        //
        // Read old preprocessed file
        //

        basic_ifstream<CharType> oldFile(cachedPath.c_str(), ios_base::in | ios_base::binary | ios_base::ate );
        REQUIRE(!!oldFile); // for tellg
        size_t oldSize = (size_t) oldFile.tellg();
        oldFile.seekg(0, ios_base::beg);
        vector<CharType> oldData(clampMin<size_t>(oldSize, 1));
        oldFile.read(&oldData[0], oldSize);
        REQUIRE(!!oldFile);
        oldFile.close();

        ////

        if (sourcesAreIdentical(&oldData[0], oldSize, &cupData[0], cupSize))
        {
            remove(cupPath.c_str());
            returnTrue;
        }
    }

    //----------------------------------------------------------------
    //
    // Compile preprocessed file with NVCC,
    // overwrite .bin
    //
    //----------------------------------------------------------------

    {
        remove(cachedPath.c_str());

        vector<StlString> nvccArgs;

        nvccArgs.push_back(CT("nvcc.exe"));

        nvccArgs.push_back(CT("-m" + platformBitness));

        nvccArgs.push_back(CT("--ptxas-options=-v"));

        nvccArgs.push_back(CT("-fatbin"));
        nvccArgs.push_back(CT("-o"));
        nvccArgs.push_back(binPath);

        nvccArgs.push_back(cupPath);

        addTargetArch(nvccArgs, platformArch);

        require(runProcess(nvccArgs, stdPass));
    }

    //
    // Dump ASM
    //
    // Cmd is used to redirect output.
    // After /C switch, the whole command line should be enclosed in another pair of quotes.
    //

    StlString dumpSass = sprintMsg(STR("\"cuobjdump\" --dump-sass \"%0\" >\"%1\""), // --dump-elf
        binPath, asmPath);

    require(runProcess(sprintMsg(STR("cmd /c \"%0\""), dumpSass), stdPass));

    //
    // Sucessful, rename .cup to .src
    //

    rename(cupPath.c_str(), cachedPath.c_str());

    returnTrue;
}

//================================================================
//
// makeCppBinAssembly
//
//================================================================

stdbool makeCppBinAssembly
(
    const StlString& cppPath,
    const StlString& binPath,
    const StlString& outPath,
    const vector<StlString>& kernelNames,
    const vector<StlString>& samplerNames,
    stdPars(CompilerKit)
)
{
    OutputTextFile outStream;
    require(outStream.open(outPath, stdPass));

    //
    // Kernel links forward declarations
    // Sampler link forward declarations
    //

    printMsg(outStream, STR("struct GpuKernelLink;"));
    printMsg(outStream, STR("struct GpuSamplerLink;"));
    printMsg(outStream, STR(""));

    printMsg(outStream, STR("namespace {"));
    printMsg(outStream, STR(""));

    for (size_t i = 0; i < kernelNames.size(); ++i)
        printMsg(outStream, STR("extern const GpuKernelLink %0;"), kernelNames[i]);

    if (kernelNames.size())
        printMsg(outStream, STR(""));

    for (size_t i = 0; i < samplerNames.size(); ++i)
        printMsg(outStream, STR("extern const GpuSamplerLink %0;"), samplerNames[i]);

    if (samplerNames.size())
        printMsg(outStream, STR(""));

    printMsg(outStream, STR("}"));
    printMsg(outStream, STR(""));

    //
    // Include base C++ file
    //

    printMsg(outStream, STR("#" "include \"%0\""), filenameToCString(cppPath));
    printMsg(outStream, STR(""));

    //
    //
    //

    basic_ifstream<uint8> binStream(binPath.c_str(), ios::in | ios::binary);
    REQUIRE_MSG1(!!binStream, STR("Cannot open %0"), binPath);

    //
    // Neccessary definitions
    //

    printMsg(outStream, STR("#ifndef GPU_DEFINE_MODULE_DESC"));
    printMsg(outStream, STR("%0"), charArrayFromPtr(PREP_STRINGIZE(GPU_DEFINE_MODULE_DESC)));
    printMsg(outStream, STR("#endif"));
    printMsg(outStream, STR(""));

    printMsg(outStream, STR("#ifndef GPU_DEFINE_KERNEL_LINK"));
    printMsg(outStream, STR("%0"), charArrayFromPtr(PREP_STRINGIZE(GPU_DEFINE_KERNEL_LINK)));
    printMsg(outStream, STR("#endif"));
    printMsg(outStream, STR(""));

    printMsg(outStream, STR("#ifndef GPU_DEFINE_SAMPLER_LINK"));
    printMsg(outStream, STR("%0"), charArrayFromPtr(PREP_STRINGIZE(GPU_DEFINE_SAMPLER_LINK)));
    printMsg(outStream, STR("#endif"));
    printMsg(outStream, STR(""));

    //
    // Module data
    //

    printMsg(outStream, STR("namespace {"));
    printMsg(outStream, STR("namespace gpuEmbeddedBinary {"));
    printMsg(outStream, STR(""));

    printMsg(outStream, STR("const unsigned char moduleData[] = "));
    printMsg(outStream, STR("{"));

    for (;;)
    {
        StlString row;

        for (int i = 0; i < 16; ++i)
        {
            uint8 value;
            binStream.read(&value, 1);

            if (!binStream)
                break;

            row += sprintMsg(STR("0x%0, "), hex(value, 2));
        }

        printMsg(outStream, STR("  %0"), row);

        if (!binStream)
            break;
    }

    REQUIRE(binStream.eof());

    printMsg(outStream, STR("};"));
    printMsg(outStream, STR(""));

    //
    // Kernel names
    //

    if (kernelNames.size())
    {
        printMsg(outStream, STR("const char* const moduleKernels[] = "));
        printMsg(outStream, STR("{"));

        for (size_t i = 0; i < kernelNames.size(); ++i)
            printMsg(outStream, STR("  \"%0\", "), kernelNames[i]);

        printMsg(outStream, STR("};"));
        printMsg(outStream, STR(""));
    }

    //
    // Samplers names
    //

    if (samplerNames.size())
    {
        printMsg(outStream, STR("const char* const moduleSamplers[] = "));
        printMsg(outStream, STR("{"));

        for (size_t i = 0; i < samplerNames.size(); ++i)
            printMsg(outStream, STR("  \"%0\", "), samplerNames[i]);

        printMsg(outStream, STR("};"));
        printMsg(outStream, STR(""));
    }

    //
    // Module desc
    //

    printMsg(outStream, STR("const GpuModuleDesc moduleDesc ="));
    printMsg(outStream, STR("{"));

    printMsg(outStream, STR("  %0, %1,"), STR("moduleData"), STR("sizeof(moduleData) / sizeof(moduleData[0])"));
    printMsg(outStream, STR("  %0, %1,"), kernelNames.size() ? STR("moduleKernels") : STR("0"), kernelNames.size());
    printMsg(outStream, STR("  %0, %1"), samplerNames.size() ? STR("moduleSamplers") : STR("0"), samplerNames.size());

    printMsg(outStream, STR("};"));
    printMsg(outStream, STR(""));

#if defined(_MSC_VER)
    printMsg(outStream, STR("#pragma section(\"gpu_section$m\", read)"));
    printMsg(outStream, STR("__declspec(allocate(\"gpu_section$m\")) const GpuModuleDesc* moduleRef = &moduleDesc;"));
#elif defined(__GNUC__)
    printMsg(outStream, STR("const GpuModuleDesc* moduleRef __attribute__((section(\"gpu_section\"))) = &moduleDesc;"));
#else
    #error
#endif

    printMsg(outStream, STR(""));

    printMsg(outStream, STR("}"));
    printMsg(outStream, STR(""));

    //
    // Kernel and sampler links
    //

    for (size_t i = 0; i < kernelNames.size(); ++i)
    {
        printMsg(outStream, STR("const GpuKernelLink %0 = {&gpuEmbeddedBinary::moduleRef, %1};"),
            kernelNames[i], i);
    }

    for (size_t i = 0; i < samplerNames.size(); ++i)
    {
        printMsg(outStream, STR("const GpuSamplerLink %0 = {&gpuEmbeddedBinary::moduleRef, %1};"),
            samplerNames[i], i);
    }

    ////

    printMsg(outStream, STR(""));
    printMsg(outStream, STR("}"));

    //
    // Flush
    //

    require(outStream.flush(stdPass));

    returnTrue;
}

//================================================================
//
// importUnicodeString
//
//================================================================

bool importUnicodeString(StlString::const_iterator ptr, StlString::const_iterator end, StlString& result)
{
    size_t len = (end - ptr) >> 1;
    result.resize(len);

    for (size_t i = 0; i < len; ++i)
    {
        CharType c0 = *ptr++;
        CharType c1 = *ptr++;
        ensure(c1 == 0 && c0 >= 0 && c0 <= 0x7F);
        result[i] = c0;
    }

    return true;
}

//================================================================
//
// mainFunc
//
//================================================================

stdbool mainFunc(int argCount, const CharType* argStr[], stdPars(CompilerKit))
{

    //----------------------------------------------------------------
    //
    // Read arguments
    //
    //----------------------------------------------------------------

    vector<StlString> cmdArgs;

    for (int i = 1; i < argCount; ++i)
        cmdArgs.push_back(argStr[i]);

    if (0)
    {
        for (size_t i = 0; i < cmdArgs.size(); ++i)
            printMsg(kit.msgLog, STR("%1"), i, cmdArgs[i]);
    }

    //----------------------------------------------------------------
    //
    // Read response files
    //
    //----------------------------------------------------------------

    vector<StlString> args;

    for (size_t i = 0; i < cmdArgs.size(); ++i)
    {
        if_not (cmdArgs[i].substr(0, 1) == CT("@"))
            args.push_back(cmdArgs[i]);
        else
        {
            StlString rspFileName = cmdArgs[i].substr(1);

            //StlString cmdline = "cmd /c copy " + rspFileName + " " + rspFileName + ".copy";
            //system(cmdline.c_str());

            InputTextFile<CharType> rspFile;
            require(rspFile.open(rspFileName, stdPass));

            StlString content;
            require(rspFile.readEntireFileToString(content, stdPass));

            ////

            if (content.size() >= 2 && uint8(content[0]) == 0xFF && uint8(content[1]) == 0xFE)
            {
                StlString tmp;

                if_not (importUnicodeString(content.begin() + 2, content.end(), tmp))
                {
                    printMsg(kit.msgLog, STR("Non-ASCII characters in response file %0"), rspFileName);
                    returnFalse;
                }

                tmp.swap(content);
            }

            basic_stringstream<CharType> strStream(content);

            for (;;)
            {
                StlString s;
                getline(strStream, s);

                bool ok = !!strStream;

                if_not (ok)
                {
                    REQUIRE_EX(strStream.eof(), printMsg(kit.msgLog, STR("Cannot parse file %0"), rspFileName));
                    break;
                }

                // printMsg(msgLog, STR("%0"), s);
                cmdLine::parseCmdLine(s, args);
            }

        }
    }

    //----------------------------------------------------------------
    //
    // Display args
    //
    //----------------------------------------------------------------

    if (0)
    {
        printMsg(kit.msgLog, STR("Arguments:"));

        for (size_t i = 0; i < args.size(); ++i)
            printMsg(kit.msgLog, STR("[%0] %1"), i, args[i]);
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    vector<StlString> includes;
    vector<StlString> defines;
    vector<StlString> cppFiles;
    vector<StlString> cudaFiles;
    StlString outputDir;
    StlString outputFile;
    vector<StlString> otherArgs;

    require(parseClArgs(args, includes, defines, cppFiles, cudaFiles, outputDir, outputFile, otherArgs, stdPass));

    bool gpuDetected = cudaFiles.size() > 0;

    //----------------------------------------------------------------
    //
    // Direct pass-thru (no .CXX files).
    //
    //----------------------------------------------------------------

    if_not (gpuDetected)
    {
        vector<StlString> clArgs;

        clArgs.push_back(CT("cl.exe"));
        clArgs.insert(clArgs.end(), cmdArgs.begin(), cmdArgs.end());

        require(runProcess(clArgs, stdPass));

        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Print defines
    //
    //----------------------------------------------------------------

    if (0)
    {
        printMsg(kit.msgLog, STR("Defines:"));

        for (size_t i = 0; i < defines.size(); ++i)
            printMsg(kit.msgLog, STR("[%0] %1"), i, defines[i]);
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    bool gpuHardwareTarget = false;

    StlString platformArch;
    StlString platformBitness;

    for (size_t i = 0; i < defines.size(); ++i)
    {
        if (stringBeginsWith(defines[i], CT("HEXLIB_PLATFORM=1")))
            gpuHardwareTarget = true;

        ////

        StlString platformArchStr = CT("HEXLIB_CUDA_ARCH=");

        if (stringBeginsWith(defines[i], platformArchStr))
            platformArch = defines[i].substr(platformArchStr.length());

        ////

        StlString platformBitnessStr = CT("HEXLIB_GPU_BITNESS=");

        if (stringBeginsWith(defines[i], platformBitnessStr))
            platformBitness = defines[i].substr(platformBitnessStr.length());
    }

    ////

    if (gpuHardwareTarget)
    {
        REQUIRE_MSG(platformArch.length() != 0, 
            STR("For CUDA hardware target, HEXLIB_CUDA_ARCH should be specified (sm_20, sm_30, ...)"));

        REQUIRE_MSG(platformBitness == CT("32") || platformBitness == CT("64"), 
            STR("For CUDA hardware target, HEXLIB_GPU_BITNESS should be specified (32 or 64)"));
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    REQUIRE_MSG(outputDir.size() != 0, STR("Cannot find output directory option"));

    //----------------------------------------------------------------
    //
    // Compile normal C/CPP files set
    //
    //----------------------------------------------------------------

    if (cppFiles.size())
    {
        vector<StlString> clArgs;

        clArgs.push_back(CT("cl.exe"));

        for (size_t i = 0; i < includes.size(); ++i)
        {
            clArgs.push_back(CT("-I"));
            clArgs.push_back(includes[i]);
        }

        for (size_t i = 0; i < defines.size(); ++i)
        {
            clArgs.push_back(CT("-D"));
            clArgs.push_back(defines[i]);
        }

        clArgs.insert(clArgs.end(), otherArgs.begin(), otherArgs.end());
        clArgs.push_back(sprintMsg(STR("/Fo%0"), outputFile.size() ? outputFile : outputDir));
        clArgs.insert(clArgs.end(), cppFiles.begin(), cppFiles.end());

        require(runProcess(clArgs, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Compile emulation file set
    //
    //----------------------------------------------------------------

    if (cudaFiles.size() && !gpuHardwareTarget)
    {
        vector<StlString> clArgs;

        clArgs.push_back(CT("cl.exe"));

        for (size_t i = 0; i < includes.size(); ++i)
        {
            clArgs.push_back(CT("-I"));
            clArgs.push_back(includes[i]);
        }

        for (size_t i = 0; i < defines.size(); ++i)
        {
            clArgs.push_back(CT("-D"));
            clArgs.push_back(defines[i]);
        }

        clArgs.push_back(CT("-D"));
        clArgs.push_back(CT("HOSTCODE=1"));

        clArgs.push_back(CT("-D"));
        clArgs.push_back(CT("DEVCODE=1"));

        clArgs.insert(clArgs.end(), otherArgs.begin(), otherArgs.end());
        clArgs.push_back(sprintMsg(STR("/Fo%0"), outputFile.size() ? outputFile : outputDir));
        clArgs.insert(clArgs.end(), cudaFiles.begin(), cudaFiles.end());

        require(runProcess(clArgs, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Compile CUDA files set
    //
    //----------------------------------------------------------------

    if (cudaFiles.size() && gpuHardwareTarget)
    {

        for (size_t i = 0; i < cudaFiles.size(); ++i)
        {

            StlString inputPath = cudaFiles[i];

            StlString inputDir;
            StlString inputName;
            StlString inputExt;

            splitPath(inputPath, inputDir, inputName, inputExt);

            //
            // Compile device part to CUBIN
            //

            vector<StlString> kernelNames;
            vector<StlString> samplerNames;

            StlString binPath = sprintMsg(STR("%0/%1.bin"), outputDir, inputName);
            StlString asmPath = sprintMsg(STR("%0/%1.asm"), outputDir, inputName);

            require(compileDevicePartToBin(inputPath, binPath, asmPath, kernelNames, samplerNames, includes, defines, platformArch, platformBitness, stdPass));

            if (0)
            {
                for (size_t i = 0; i < samplerNames.size(); ++i)
                    printMsg(kit.msgLog, STR("Sampler: %0"), samplerNames[i], msgErr);
            }

            //
            // Translate to C module
            //

            StlString cppAssemblyPath = sprintMsg(STR("%0/%1.assembly.cpp"), inputDir, inputName);
            REMEMBER_CLEANUP1(remove(cppAssemblyPath.c_str()), const StlString&, cppAssemblyPath);
            require(makeCppBinAssembly(inputPath, binPath, cppAssemblyPath, kernelNames, samplerNames, stdPass));

            //
            // Compile combined CPP
            //

            {
                vector<StlString> clArgs;

                clArgs.push_back(CT("cl.exe"));

                for (size_t i = 0; i < includes.size(); ++i)
                {
                    clArgs.push_back(CT("-I"));
                    clArgs.push_back(includes[i]);
                }

                clArgs.push_back(CT("-I"));
                clArgs.push_back(CT("."));

                for (size_t i = 0; i < defines.size(); ++i)
                {
                    clArgs.push_back(CT("-D"));
                    clArgs.push_back(defines[i]);
                }

                clArgs.push_back(CT("-D"));
                clArgs.push_back(CT("HOSTCODE=1"));

                clArgs.push_back(CT("-D"));
                clArgs.push_back(CT("DEVCODE=0"));

                clArgs.insert(clArgs.end(), otherArgs.begin(), otherArgs.end());
                clArgs.push_back(sprintMsg(STR("/Fo%0"), outputFile.size() ? outputFile : outputDir));
                clArgs.push_back(cppAssemblyPath);

                require(runProcess(clArgs, stdPass));
            }
        }
    }

    //
    //
    //

    returnTrue;
}

//================================================================
//
// main
//
//================================================================

int main(int argCount, const CharType* argStr[])
{
    //----------------------------------------------------------------
    //
    // Text logs and trace root
    //
    //----------------------------------------------------------------

#ifdef _DEBUG
    LogToStlConsole msgLog(true);
#else
    LogToStlConsole msgLog(false);
#endif

    ErrorLogByMsgLog errorLog(msgLog);
    ErrorLogExByMsgLog errorLogEx(msgLog);
    FileToolsImpl fileTools;

    CompilerKit kit = kitCombine
    (
        ErrorLogKit(errorLog),
        ErrorLogExKit(errorLogEx),
        MsgLogKit(msgLog),
        FileToolsKit(fileTools)
    );

    TRACE_ROOT(stdTraceName, TRACE_AUTO_LOCATION);

    //----------------------------------------------------------------
    //
    // Core
    //
    //----------------------------------------------------------------

    bool ok = foreignErrorBlock(mainFunc(argCount, argStr, stdPass));

    ////

    if_not (checkHeapIntegrity())
        printMsg(kit.msgLog, STR("Heap memory is damaged!"), msgErr);
    else if_not (checkHeapLeaks())
        printMsg(kit.msgLog, STR("Memory leaks are detected!"), msgErr);

    ////

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
