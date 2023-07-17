#include "pixelBufferDrawing.h"

#include "testbedGL/common/glDebugCheck.h"
#include "dataAlloc/arrayMemory.h"
#include "userOutput/printMsg.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// ReportCompileInfoKit
//
//================================================================

using ReportCompileInfoKit = KitCombine<DiagnosticKit, MallocKit>;

//================================================================
//
// reportCompileInfo
//
//================================================================

stdbool reportCompileInfo(GLuint shader, const CharArray& prefix, stdPars(ReportCompileInfoKit))
{
    REQUIRE_GL_FUNC(glGetShaderiv);
    GLint compileSuccess = 0;
    REQUIRE_GL(glGetShaderiv(shader, GL_COMPILE_STATUS, &compileSuccess));

    if (compileSuccess)
        returnTrue;

    ////

    REQUIRE_GL_FUNC(glGetShaderiv);
    GLint bufferSize = 0;
    REQUIRE_GL(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &bufferSize));

    REQUIRE(bufferSize >= 1);
    if (bufferSize == 1) returnTrue;

    ArrayMemory<char> tmp;
    require(tmp.realloc(bufferSize, cpuBaseByteAlignment, kit.malloc, stdPass));
    ARRAY_EXPOSE_UNSAFE(tmp);

    REQUIRE_GL_FUNC(glGetShaderInfoLog);
    REQUIRE_GL(glGetShaderInfoLog(shader, bufferSize, NULL, tmpPtr));

    ////

    printMsg(kit.msgLog, STR("%0"), prefix, msgErr);
    printMsg(kit.msgLog, STR("%0"), const_cast<const char*>(tmpPtr), msgErr);

    ////

    returnTrue;
}

//================================================================
//
// compileShaderProgram
//
//================================================================

stdbool compileShaderProgram(const char* vertexShaderSrc, const char* fragmentShaderSrc, GLuint& program, stdPars(ReportCompileInfoKit))
{
    //----------------------------------------------------------------
    //
    // Program
    //
    //----------------------------------------------------------------

    REQUIRE_GL_FUNC2(glCreateProgram, glDeleteProgram);
    program = glCreateProgram();
    REMEMBER_CLEANUP_EX(cleanProgram, {glDeleteProgram(program); program = 0;});

    //----------------------------------------------------------------
    //
    // Vertex shader
    //
    //----------------------------------------------------------------

    REQUIRE(vertexShaderSrc);
    REQUIRE_GL_FUNC2(glCreateShader, glDeleteShader);
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    REMEMBER_CLEANUP_EX(cleanVertexShader, {glDeleteShader(vertexShader); vertexShader = 0;});

    ////

    REQUIRE_GL_FUNC(glShaderSource);
    REQUIRE_GL(glShaderSource(vertexShader, 1, &vertexShaderSrc, NULL));

    REQUIRE_GL_FUNC(glCompileShader);
    REQUIRE_GL(glCompileShader(vertexShader));
    require(reportCompileInfo(vertexShader, STR("Vertex shader compilation:"), stdPass));

    ////

    GLint vertexCompiled = 0;
    REQUIRE_GL_FUNC(glGetShaderiv);
    REQUIRE_GL(glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vertexCompiled));
    require(vertexCompiled == GL_TRUE);

    ////

    REQUIRE_GL_FUNC(glAttachShader);
    REQUIRE_GL(glAttachShader(program, vertexShader));

    //----------------------------------------------------------------
    //
    // Fragment shader
    //
    //----------------------------------------------------------------

    REQUIRE(fragmentShaderSrc);
    REQUIRE_GL_FUNC2(glCreateShader, glDeleteShader);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    REMEMBER_CLEANUP_EX(cleanFragmentShader, {glDeleteShader(fragmentShader); fragmentShader = 0;});

    ////

    REQUIRE_GL_FUNC(glShaderSource);
    REQUIRE_GL(glShaderSource(fragmentShader, 1, &fragmentShaderSrc, NULL));

    REQUIRE_GL_FUNC(glCompileShader);
    REQUIRE_GL(glCompileShader(fragmentShader));
    require(reportCompileInfo(fragmentShader, STR("Fragment shader compilation:"), stdPass));

    ////

    GLint fragmentCompiled = 0;
    REQUIRE_GL_FUNC(glGetShaderiv);
    REQUIRE_GL(glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fragmentCompiled));
    require(fragmentCompiled == GL_TRUE);

    ////

    REQUIRE_GL_FUNC(glAttachShader);
    REQUIRE_GL(glAttachShader(program, fragmentShader));

    //----------------------------------------------------------------
    //
    // Link program
    //
    //----------------------------------------------------------------

    REQUIRE_GL_FUNC(glLinkProgram);
    REQUIRE_GL(glLinkProgram(program));

    //----------------------------------------------------------------
    //
    // Program errors
    //
    //----------------------------------------------------------------

    {
        GLint linkSuccess = 0;
        REQUIRE_GL_FUNC(glGetProgramiv);
        REQUIRE_GL(glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess));

        if_not (linkSuccess != 0)
        {
            GLint logSize = 0;
            REQUIRE_GL_FUNC(glGetProgramiv);
            REQUIRE_GL(glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize));
            REQUIRE(logSize >= 0);

            ////

            if_not (logSize == 1)
            {
                ArrayMemory<char> tmp;
                require(tmp.realloc(logSize, cpuBaseByteAlignment, kit.malloc, stdPass));
                ARRAY_EXPOSE_UNSAFE(tmp);

                REQUIRE_GL_FUNC(glGetProgramInfoLog);
                REQUIRE_GL(glGetProgramInfoLog(program, tmpSize, NULL, tmpPtr));

                printMsg(kit.msgLog, STR("Shader program compilation:"), msgErr);
                printMsg(kit.msgLog, STR("%0"), const_cast<const char*>(tmpPtr), msgErr);
            }
        }
    }

    ////

    GLint programLinked = 0;
    REQUIRE_GL_FUNC(glGetProgramiv);
    REQUIRE_GL(glGetProgramiv(program, GL_LINK_STATUS, &programLinked));
    require(programLinked == GL_TRUE);

    ////

    cleanProgram.cancel();
    cleanVertexShader.cancel();
    cleanFragmentShader.cancel();

    returnTrue;
}

//================================================================
//
// PixelBufferDrawingState
//
//================================================================

struct PixelBufferDrawingState
{
    GLuint shaderProgram = 0;
    GLint bufferAddressLocation = -1;
    GLint bufferPitchLocation = -1;
    GLint bufferSizeFloatLocation = -1;
    GLint bufferSizeUintLocation = -1;
};

//================================================================
//
// PixelBufferDrawingImpl
//
//================================================================

class PixelBufferDrawingImpl : public PixelBufferDrawing, private PixelBufferDrawingState
{

public:

    ~PixelBufferDrawingImpl()
    {
        deinit();
    }

    inline void deinit()
    {
        if (shaderProgram != 0)
            DEBUG_BREAK_CHECK_GL(glDeleteProgram(shaderProgram));

        PixelBufferDrawingState& state = *this;
        state = {};
    }

    stdbool reinit(stdPars(ReportCompileInfoKit));

    stdbool draw(const PixelBuffer& buffer, const Point<Space>& pos, stdPars(DiagnosticKit));

private:

    static const char* vertexShaderText;
    static const char* fragmentShaderText;

};

////

UniquePtr<PixelBufferDrawing> PixelBufferDrawing::create()
{
    return makeUnique<PixelBufferDrawingImpl>();
}

//================================================================
//
// PixelBufferDrawingImpl::vertexShaderText
// PixelBufferDrawingImpl::fragmentShaderText
//
//================================================================

const char* PixelBufferDrawingImpl::vertexShaderText =
    "varying out vec2 textureCoordinate;\n"
    "uniform vec2 bufferSizeFloat;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
    "    textureCoordinate = vec2(gl_MultiTexCoord0) * bufferSizeFloat;\n"
    "}\n";

const char* PixelBufferDrawingImpl::fragmentShaderText =
    "#extension GL_NV_shader_buffer_load: enable\n"
    "#extension GL_ARB_shading_language_packing : enable\n"
    "\n"
    "uniform uint* bufferAddress;\n"
    "uniform int bufferPitch;\n"
    "uniform uvec2 bufferSizeUint;\n"
    "in vec2 textureCoordinate;\n"
    "varying out vec4 pixelColor;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    ivec2 idx = ivec2(textureCoordinate);\n" /* space->grid and round */
    "    uint* bufferPtr = bufferAddress + idx.x + idx.y * bufferPitch;\n"
    "    uint pixelValue = 0U;\n"
    "\n"
    "    bool validAccess = uint(idx.x) < bufferSizeUint.x && uint(idx.y) < bufferSizeUint.y;\n"
    "    if (validAccess) pixelValue = *bufferPtr;\n"
    "\n"
    "    vec4 color = unpackUnorm4x8(pixelValue);\n"
    "    pixelColor = color.zyxw;\n"
    "}\n";

//================================================================
//
// PixelBufferDrawingImpl::reinit
//
//================================================================

stdbool PixelBufferDrawingImpl::reinit(stdPars(ReportCompileInfoKit))
{
    deinit();

    REMEMBER_CLEANUP_EX(totalCleanup, deinit());

    ////

    require(compileShaderProgram(vertexShaderText, fragmentShaderText, shaderProgram, stdPass));

    ////

    REQUIRE_GL_FUNC(glGetUniformLocation);

    bufferAddressLocation = glGetUniformLocation(shaderProgram, "bufferAddress");
    REQUIRE(bufferAddressLocation != -1);

    bufferPitchLocation = glGetUniformLocation(shaderProgram, "bufferPitch");
    REQUIRE(bufferPitchLocation != -1);

    bufferSizeFloatLocation = glGetUniformLocation(shaderProgram, "bufferSizeFloat");
    REQUIRE(bufferSizeFloatLocation != -1);

    bufferSizeUintLocation = glGetUniformLocation(shaderProgram, "bufferSizeUint");
    REQUIRE(bufferSizeUintLocation != -1);

    ////

    totalCleanup.cancel();

    returnTrue;
}

//================================================================
//
// PixelBufferDrawingImpl::draw
//
//================================================================

stdbool PixelBufferDrawingImpl::draw(const PixelBuffer& buffer, const Point<Space>& pos, stdPars(DiagnosticKit))
{
    REQUIRE(shaderProgram != 0);

    ////

    Point<Space> bufferSize = buffer.size();
    REQUIRE(bufferSize >= 0);

    ////

    GLuint64EXT bufferAddress = 0;
    Space bufferPitch = 0;
    require(buffer.getGraphicsBuffer(bufferAddress, bufferPitch, stdPass));
    REQUIRE(bufferPitch >= 0);

    ////

    REQUIRE_GL_FUNC(glProgramUniformui64NV);
    glProgramUniformui64NV(shaderProgram, bufferAddressLocation, bufferAddress);

    REQUIRE_GL_FUNC(glProgramUniform1i);
    glProgramUniform1i(shaderProgram, bufferPitchLocation, bufferPitch);

    REQUIRE_GL_FUNC(glProgramUniform2f);
    glProgramUniform2f(shaderProgram, bufferSizeFloatLocation, float32(bufferSize.X), float32(bufferSize.Y));

    REQUIRE_GL_FUNC(glProgramUniform2ui);
    glProgramUniform2ui(shaderProgram, bufferSizeUintLocation, bufferSize.X, bufferSize.Y);

    ////

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);

    ////

    glViewport(pos.X, pos.Y, bufferSize.X, bufferSize.Y);
    REMEMBER_CLEANUP(glViewport(0, 0, 0, 0));

    ////

    REQUIRE_GL_FUNC(glUseProgram);
    glUseProgram(shaderProgram);
    REMEMBER_CLEANUP(glUseProgram(0));

    ////

    {
        glBegin(GL_QUADS);

        glTexCoord2f(0, 1);
        glVertex3f(0, 0, 0);

        glTexCoord2f(0, 0);
        glVertex3f(0, 1, 0);

        glTexCoord2f(1, 0);
        glVertex3f(1, 1, 0);

        glTexCoord2f(1, 1);
        glVertex3f(1, 0, 0);

        glEnd();
    }

    ////

    constexpr bool drawCheck = true;
    REQUIRE_GL(drawCheck);

    ////

    returnTrue;
}
