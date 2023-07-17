#include "windowManagerGLFW.h"

#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "dataAlloc/arrayMemory.inl"
#include "errorLog/debugBreak.h"
#include "errorLog/errorLog.h"
#include "simpleString/simpleString.h"
#include "storage/constructDestruct.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsgTrace.h"
#include "storage/adapters/lambdaThunk.h"
#include "testbedGL/windowManager/timeoutHelper.h"
#include "errorLog/convertExceptions.h"

//================================================================
//
// glfwErrorCallback
//
// It's nailed to a single thread by GLFW design.
//
//================================================================

static SimpleStringChar glfwError;

//----------------------------------------------------------------

void glfwErrorCallback(int errorCode, const char* errorDesc)
{
    glfwError = errorDesc;
}

//----------------------------------------------------------------

inline bool glfwNoError()
{
    return glfwError.valid() && glfwError.size() == 0;
}

//================================================================
//
// REQUIRE_GLFW
//
//================================================================

template <typename Kit>
inline bool checkGLFWHelper(const CharType* statement, stdPars(Kit))
{
    if (glfwError.valid() && glfwError.size() == 0)
        return true;

    ////

    const char* msg = "Insufficient memory";

    if (glfwError.valid())
        msg = glfwError.cstr();

    printMsgTrace(kit.msgLogEx, STR("GLFW library: %0 returned error \"%1\"."), statement, msg, msgErr, stdPassThru);

    ////

    glfwError.clear();

    return false;
}

//----------------------------------------------------------------

#define REQUIRE_GLFW(statement) \
    require(((statement), checkGLFWHelper(PREP_STRINGIZE(statement), stdPass)))

//================================================================
//
// Event handlers.
//
//================================================================

using RefreshHandler = Callable<void ()>;
using KeyHandler = Callable<void (const KeyEvent& event)>;
using MouseMoveHandler = Callable<void (const MouseMoveEvent& event)>;
using MouseButtonHandler = Callable<void (const MouseButtonEvent& event)>;
using ScrollHandler = Callable<void (const ScrollEvent& event)>;
using ResizeHandler = Callable<void (const ResizeEvent& event)>;

//================================================================
//
// WindowGLFW
//
//================================================================

class WindowGLFW : public Window
{

public:

    //----------------------------------------------------------------
    //
    // Init/deinit.
    //
    //----------------------------------------------------------------

    ~WindowGLFW() {close();}

    stdbool open(const WindowCreationArgs& args, stdPars(WindowManager::Kit));

    void close()
    {
        timeoutHelper.reset();

        if (baseWindow)
        {
            glfwDestroyWindow(baseWindow);
            baseWindow = nullptr;
        }
    }

    //----------------------------------------------------------------
    //
    // API.
    //
    //----------------------------------------------------------------

    stdbool setThreadDrawingContext(stdPars(Kit))
    {
        REQUIRE(baseWindow);
        REQUIRE_GLFW(glfwMakeContextCurrent(baseWindow));
        returnTrue;
    }

    stdbool setVisible(bool visible, stdPars(Kit))
    {
        REQUIRE(baseWindow);

        if (visible)
            REQUIRE_GLFW(glfwShowWindow(baseWindow));
        else
            REQUIRE_GLFW(glfwHideWindow(baseWindow));

        returnTrue;
    }

    template <typename Lambda>
    stdbool callbackShell(const EventReceivers& receivers, const Lambda& lambda, stdPars(Kit));

    stdbool getEvents(bool waitEvents, const OptionalObject<uint32>& waitTimeoutMs, const EventReceivers& receivers, stdPars(Kit));

    stdbool shouldContinue(stdPars(Kit))
    {
        REQUIRE(baseWindow);
        int result = glfwWindowShouldClose(baseWindow);
        REQUIRE_GLFW("glfwWindowShouldClose");
        require(result == GL_FALSE);
        returnTrue;
    }

    stdbool swapBuffers(stdPars(Kit))
    {
        REQUIRE(baseWindow);
        REQUIRE_GLFW(glfwSwapBuffers(baseWindow));
        returnTrue;
    }

    stdbool getWindowLocation(WindowLocation& location, stdPars(Kit));

    stdbool setWindowLocation(const WindowLocation& location, stdPars(Kit));

    stdbool getImageSize(Point<Space>& size, stdPars(Kit))
    {
        REQUIRE(baseWindow);
        REQUIRE_GLFW(glfwGetFramebufferSize(baseWindow, &size.X, &size.Y));
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Timeout helper.
    //
    //----------------------------------------------------------------

public:

    UniquePtr<TimeoutHelper> timeoutHelper;

    //----------------------------------------------------------------
    //
    // Base window.
    //
    //----------------------------------------------------------------

private:

    GLFWwindow* baseWindow = nullptr;

    //----------------------------------------------------------------
    //
    // Callbacks.
    //
    //----------------------------------------------------------------

public:

    RefreshHandler* refreshHandler = nullptr;
    KeyHandler* keyHandler = nullptr;
    MouseMoveHandler* mouseMoveHandler = nullptr;
    MouseButtonHandler* mouseButtonHandler = nullptr;
    ScrollHandler* scrollHandler = nullptr;
    ResizeHandler* resizeHandler = nullptr;

};

//================================================================
//
// glfwRefreshCallback
//
//================================================================

void glfwRefreshCallback(GLFWwindow* window)
{
    void* userPtr = glfwGetWindowUserPointer(window);
    ensurev(DEBUG_BREAK_CHECK(userPtr));

    auto& the = * (WindowGLFW*) userPtr;

    if (DEBUG_BREAK_CHECK(the.refreshHandler))
        the.refreshHandler->call();
}

//================================================================
//
// getKeyModifiers
//
//================================================================

inline KeyModifiers getKeyModifiers(int mods)
{
    KeyModifiers result = 0;

    if (mods & GLFW_MOD_SHIFT)
        result |= KeyModifier::Shift;

    if (mods & GLFW_MOD_CONTROL)
        result |= KeyModifier::Control;

    if (mods & GLFW_MOD_ALT)
        result |= KeyModifier::Alt;

    if (mods & GLFW_MOD_SUPER)
        result |= KeyModifier::Super;

    return result;
}

//================================================================
//
// glfwKeyCallback
//
//================================================================

void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    void* userPtr = glfwGetWindowUserPointer(window);
    ensurev(DEBUG_BREAK_CHECK(userPtr));

    auto& the = * (WindowGLFW*) userPtr;

    //----------------------------------------------------------------
    //
    // Modifiers
    //
    //----------------------------------------------------------------

    auto keyModifiers = getKeyModifiers(mods);

    //----------------------------------------------------------------
    //
    // Key code
    //
    //----------------------------------------------------------------

    KeyCode keyCode = Key::None;

    switch (key)
    {
        #define TMP_MACRO(src, dst) \
            case src: keyCode = dst; break;

        ////

        TMP_MACRO(GLFW_KEY_SPACE, Key::Space)
        TMP_MACRO(GLFW_KEY_APOSTROPHE, Key::Apostrophe)
        TMP_MACRO(GLFW_KEY_COMMA, Key::Comma)
        TMP_MACRO(GLFW_KEY_MINUS, Key::Minus)
        TMP_MACRO(GLFW_KEY_PERIOD, Key::Period)
        TMP_MACRO(GLFW_KEY_SLASH, Key::Slash)
        TMP_MACRO(GLFW_KEY_0, Key::_0)
        TMP_MACRO(GLFW_KEY_1, Key::_1)
        TMP_MACRO(GLFW_KEY_2, Key::_2)
        TMP_MACRO(GLFW_KEY_3, Key::_3)
        TMP_MACRO(GLFW_KEY_4, Key::_4)
        TMP_MACRO(GLFW_KEY_5, Key::_5)
        TMP_MACRO(GLFW_KEY_6, Key::_6)
        TMP_MACRO(GLFW_KEY_7, Key::_7)
        TMP_MACRO(GLFW_KEY_8, Key::_8)
        TMP_MACRO(GLFW_KEY_9, Key::_9)
        TMP_MACRO(GLFW_KEY_SEMICOLON, Key::Semicolon)
        TMP_MACRO(GLFW_KEY_EQUAL, Key::Equal)
        TMP_MACRO(GLFW_KEY_A, Key::A)
        TMP_MACRO(GLFW_KEY_B, Key::B)
        TMP_MACRO(GLFW_KEY_C, Key::C)
        TMP_MACRO(GLFW_KEY_D, Key::D)
        TMP_MACRO(GLFW_KEY_E, Key::E)
        TMP_MACRO(GLFW_KEY_F, Key::F)
        TMP_MACRO(GLFW_KEY_G, Key::G)
        TMP_MACRO(GLFW_KEY_H, Key::H)
        TMP_MACRO(GLFW_KEY_I, Key::I)
        TMP_MACRO(GLFW_KEY_J, Key::J)
        TMP_MACRO(GLFW_KEY_K, Key::K)
        TMP_MACRO(GLFW_KEY_L, Key::L)
        TMP_MACRO(GLFW_KEY_M, Key::M)
        TMP_MACRO(GLFW_KEY_N, Key::N)
        TMP_MACRO(GLFW_KEY_O, Key::O)
        TMP_MACRO(GLFW_KEY_P, Key::P)
        TMP_MACRO(GLFW_KEY_Q, Key::Q)
        TMP_MACRO(GLFW_KEY_R, Key::R)
        TMP_MACRO(GLFW_KEY_S, Key::S)
        TMP_MACRO(GLFW_KEY_T, Key::T)
        TMP_MACRO(GLFW_KEY_U, Key::U)
        TMP_MACRO(GLFW_KEY_V, Key::V)
        TMP_MACRO(GLFW_KEY_W, Key::W)
        TMP_MACRO(GLFW_KEY_X, Key::X)
        TMP_MACRO(GLFW_KEY_Y, Key::Y)
        TMP_MACRO(GLFW_KEY_Z, Key::Z)
        TMP_MACRO(GLFW_KEY_LEFT_BRACKET, Key::LeftBracket)
        TMP_MACRO(GLFW_KEY_BACKSLASH, Key::Backslash)
        TMP_MACRO(GLFW_KEY_RIGHT_BRACKET, Key::RightBracket)
        TMP_MACRO(GLFW_KEY_GRAVE_ACCENT, Key::GraveAccent)
        TMP_MACRO(GLFW_KEY_WORLD_1, Key::World1)
        TMP_MACRO(GLFW_KEY_WORLD_2, Key::World2)

        TMP_MACRO(GLFW_KEY_ESCAPE, Key::Escape)
        TMP_MACRO(GLFW_KEY_ENTER, Key::Enter)
        TMP_MACRO(GLFW_KEY_TAB, Key::Tab)
        TMP_MACRO(GLFW_KEY_BACKSPACE, Key::Backspace)
        TMP_MACRO(GLFW_KEY_INSERT, Key::Insert)
        TMP_MACRO(GLFW_KEY_DELETE, Key::Delete)
        TMP_MACRO(GLFW_KEY_RIGHT, Key::Right)
        TMP_MACRO(GLFW_KEY_LEFT, Key::Left)
        TMP_MACRO(GLFW_KEY_DOWN, Key::Down)
        TMP_MACRO(GLFW_KEY_UP, Key::Up)
        TMP_MACRO(GLFW_KEY_PAGE_UP, Key::PageUp)
        TMP_MACRO(GLFW_KEY_PAGE_DOWN, Key::PageDown)
        TMP_MACRO(GLFW_KEY_HOME, Key::Home)
        TMP_MACRO(GLFW_KEY_END, Key::End)
        TMP_MACRO(GLFW_KEY_CAPS_LOCK, Key::CapsLock)
        TMP_MACRO(GLFW_KEY_SCROLL_LOCK, Key::ScrollLock)
        TMP_MACRO(GLFW_KEY_NUM_LOCK, Key::NumLock)
        TMP_MACRO(GLFW_KEY_PRINT_SCREEN, Key::PrintScreen)
        TMP_MACRO(GLFW_KEY_PAUSE, Key::Pause)
        TMP_MACRO(GLFW_KEY_F1, Key::F1)
        TMP_MACRO(GLFW_KEY_F2, Key::F2)
        TMP_MACRO(GLFW_KEY_F3, Key::F3)
        TMP_MACRO(GLFW_KEY_F4, Key::F4)
        TMP_MACRO(GLFW_KEY_F5, Key::F5)
        TMP_MACRO(GLFW_KEY_F6, Key::F6)
        TMP_MACRO(GLFW_KEY_F7, Key::F7)
        TMP_MACRO(GLFW_KEY_F8, Key::F8)
        TMP_MACRO(GLFW_KEY_F9, Key::F9)
        TMP_MACRO(GLFW_KEY_F10, Key::F10)
        TMP_MACRO(GLFW_KEY_F11, Key::F11)
        TMP_MACRO(GLFW_KEY_F12, Key::F12)
        TMP_MACRO(GLFW_KEY_F13, Key::F13)
        TMP_MACRO(GLFW_KEY_F14, Key::F14)
        TMP_MACRO(GLFW_KEY_F15, Key::F15)
        TMP_MACRO(GLFW_KEY_F16, Key::F16)
        TMP_MACRO(GLFW_KEY_F17, Key::F17)
        TMP_MACRO(GLFW_KEY_F18, Key::F18)
        TMP_MACRO(GLFW_KEY_F19, Key::F19)
        TMP_MACRO(GLFW_KEY_F20, Key::F20)
        TMP_MACRO(GLFW_KEY_F21, Key::F21)
        TMP_MACRO(GLFW_KEY_F22, Key::F22)
        TMP_MACRO(GLFW_KEY_F23, Key::F23)
        TMP_MACRO(GLFW_KEY_F24, Key::F24)
        TMP_MACRO(GLFW_KEY_F25, Key::F25)
        TMP_MACRO(GLFW_KEY_KP_0, Key::Kp0)
        TMP_MACRO(GLFW_KEY_KP_1, Key::Kp1)
        TMP_MACRO(GLFW_KEY_KP_2, Key::Kp2)
        TMP_MACRO(GLFW_KEY_KP_3, Key::Kp3)
        TMP_MACRO(GLFW_KEY_KP_4, Key::Kp4)
        TMP_MACRO(GLFW_KEY_KP_5, Key::Kp5)
        TMP_MACRO(GLFW_KEY_KP_6, Key::Kp6)
        TMP_MACRO(GLFW_KEY_KP_7, Key::Kp7)
        TMP_MACRO(GLFW_KEY_KP_8, Key::Kp8)
        TMP_MACRO(GLFW_KEY_KP_9, Key::Kp9)
        TMP_MACRO(GLFW_KEY_KP_DECIMAL, Key::KpDecimal)
        TMP_MACRO(GLFW_KEY_KP_DIVIDE, Key::KpDivide)
        TMP_MACRO(GLFW_KEY_KP_MULTIPLY, Key::KpMultiply)
        TMP_MACRO(GLFW_KEY_KP_SUBTRACT, Key::KpSubtract)
        TMP_MACRO(GLFW_KEY_KP_ADD, Key::KpAdd)
        TMP_MACRO(GLFW_KEY_KP_ENTER, Key::KpEnter)
        TMP_MACRO(GLFW_KEY_KP_EQUAL, Key::KpEqual)
        TMP_MACRO(GLFW_KEY_LEFT_SHIFT, Key::LeftShift)
        TMP_MACRO(GLFW_KEY_LEFT_CONTROL, Key::LeftControl)
        TMP_MACRO(GLFW_KEY_LEFT_ALT, Key::LeftAlt)
        TMP_MACRO(GLFW_KEY_LEFT_SUPER, Key::LeftSuper)
        TMP_MACRO(GLFW_KEY_RIGHT_SHIFT, Key::RightShift)
        TMP_MACRO(GLFW_KEY_RIGHT_CONTROL, Key::RightControl)
        TMP_MACRO(GLFW_KEY_RIGHT_ALT, Key::RightAlt)
        TMP_MACRO(GLFW_KEY_RIGHT_SUPER, Key::RightSuper)
        TMP_MACRO(GLFW_KEY_MENU, Key::Menu)

        ////

        #undef TMP_MACRO

    };

    //----------------------------------------------------------------
    //
    // Key action.
    //
    //----------------------------------------------------------------

    auto keyAction = KeyAction::Press;

    if (action == GLFW_PRESS)
        keyAction = KeyAction::Press;
    else if (action == GLFW_RELEASE)
        keyAction = KeyAction::Release;
    else if (action == GLFW_REPEAT)
        keyAction = KeyAction::Repeat;
    else
        ensurev(DEBUG_BREAK_CHECK(false));

    //----------------------------------------------------------------
    //
    // Pass key.
    //
    //----------------------------------------------------------------

    KeyEvent r;
    r.action = keyAction;
    r.code = keyCode;
    r.modifiers = keyModifiers;

    ////

    if (DEBUG_BREAK_CHECK(the.keyHandler))
        the.keyHandler->call(r);
}

//================================================================
//
// glfwCursorPosCallback
//
//================================================================

void glfwCursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    void* userPtr = glfwGetWindowUserPointer(window);
    ensurev(DEBUG_BREAK_CHECK(userPtr));

    auto& the = * (WindowGLFW*) userPtr;

    ////

    auto pos = point
    (
        float32(xpos) + 0.5f,
        float32(ypos) + 0.5f
    );

    if (DEBUG_BREAK_CHECK(the.mouseMoveHandler))
        the.mouseMoveHandler->call(pos);
}

//================================================================
//
// glfwMouseButtonCallback
//
//================================================================

void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    void* userPtr = glfwGetWindowUserPointer(window);
    ensurev(DEBUG_BREAK_CHECK(userPtr));

    auto& the = * (WindowGLFW*) userPtr;

    ////

    double xpos{}, ypos{};
    glfwGetCursorPos(window, &xpos, &ypos);
    ensurev(DEBUG_BREAK_CHECK(glfwNoError()));

    auto pos = point
    (
        float32(xpos) + 0.5f,
        float32(ypos) + 0.5f
    );

    ////

    if (DEBUG_BREAK_CHECK(!!the.mouseButtonHandler))
        (*the.mouseButtonHandler)({pos, button, action == GLFW_PRESS, getKeyModifiers(mods)});
}

//================================================================
//
// glfwScrollCallback
//
//================================================================

void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    void* userPtr = glfwGetWindowUserPointer(window);
    ensurev(DEBUG_BREAK_CHECK(userPtr));

    auto& the = * (WindowGLFW*) userPtr;

    ////

    auto ofs = point(float32(xoffset), float32(yoffset));

    if (DEBUG_BREAK_CHECK(the.scrollHandler))
        the.scrollHandler->call(ofs);
}

//================================================================
//
// glfwFramebufferSizeCallback
//
//================================================================

void glfwFramebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    void* userPtr = glfwGetWindowUserPointer(window);
    ensurev(DEBUG_BREAK_CHECK(userPtr));

    auto& the = * (WindowGLFW*) userPtr;

    ////

    if (the.resizeHandler) // no warnings -- can be called from setWindowLocation
        the.resizeHandler->call(point(width, height));
}

//================================================================
//
// WindowGLFW::open
//
//================================================================

stdbool WindowGLFW::open(const WindowCreationArgs& arg, stdPars(WindowManager::Kit))
{
    //----------------------------------------------------------------
    //
    // Close everything.
    //
    //----------------------------------------------------------------

    close();

    //----------------------------------------------------------------
    //
    // Timeout helper.
    //
    //----------------------------------------------------------------

    convertExceptions(timeoutHelper = TimeoutHelper::create());

    //----------------------------------------------------------------
    //
    // Create a window.
    //
    //----------------------------------------------------------------

    auto& location = arg.location;

    ////

    REQUIRE_GLFW(glfwDefaultWindowHints());

    REQUIRE_GLFW(glfwWindowHint(GLFW_RESIZABLE, arg.resizable ? GL_TRUE : GL_FALSE));
    REQUIRE_GLFW(glfwWindowHint(GLFW_DEPTH_BITS, 0));
    REQUIRE_GLFW(glfwWindowHint(GLFW_STENCIL_BITS, 0));

    REQUIRE_GLFW(glfwWindowHint(GLFW_VISIBLE, GL_FALSE));

    ////

    auto primaryMonitor = glfwGetPrimaryMonitor();
    REQUIRE_GLFW("glfwGetPrimaryMonitor");
    REQUIRE(primaryMonitor);

    auto primaryMode = glfwGetVideoMode(primaryMonitor);
    REQUIRE_GLFW("glfwGetVideoMode");
    REQUIRE(primaryMode);

    ////

    GLFWmonitor* creationMonitor = nullptr;

    REQUIRE(location.size >= 0);
    auto creationSize = location.size;

    ////

    if (location.mode == WindowMode::FullScreen)
    {
        creationMonitor = primaryMonitor;
        creationSize = point(primaryMode->width, primaryMode->height);

        REQUIRE_GLFW(glfwWindowHint(GLFW_RED_BITS, primaryMode->redBits));
        REQUIRE_GLFW(glfwWindowHint(GLFW_GREEN_BITS, primaryMode->greenBits));
        REQUIRE_GLFW(glfwWindowHint(GLFW_BLUE_BITS, primaryMode->blueBits));
        REQUIRE_GLFW(glfwWindowHint(GLFW_REFRESH_RATE, primaryMode->refreshRate));
    }

    ////

    REQUIRE_GLFW(glfwWindowHint(GLFW_DECORATED, location.decorated ? GL_TRUE : GL_FALSE));

    ////

    REQUIRE(arg.name);

    GLFWwindow* w = glfwCreateWindow(creationSize.X, creationSize.Y, arg.name, creationMonitor, 0);
    REQUIRE_GLFW("glfwCreateWindow");
    REQUIRE(w != 0);
    REMEMBER_CLEANUP_EX(cleanWindow, glfwDestroyWindow(w));

    //----------------------------------------------------------------
    //
    // Thread OpenGL context.
    //
    //----------------------------------------------------------------

    REQUIRE_GLFW(glfwMakeContextCurrent(w));

    //----------------------------------------------------------------
    //
    // GL extensions
    //
    // All specific extensions are checked directly in the application code,
    // by testing a function pointer.
    //
    //----------------------------------------------------------------

    REQUIRE(glewInit() == GLEW_OK);

    //----------------------------------------------------------------
    //
    // Vertical sync.
    //
    //----------------------------------------------------------------

    REQUIRE_GLFW(glfwSwapInterval(location.verticalSync ? 1 : 0));

    //----------------------------------------------------------------
    //
    // Window mode.
    //
    //----------------------------------------------------------------

    if (location.mode == WindowMode::Minimized)
        REQUIRE_GLFW(glfwIconifyWindow(w));

    if (location.mode == WindowMode::Normal)
        REQUIRE_GLFW(glfwSetWindowPos(w, location.pos.X, location.pos.Y));

    if (location.mode == WindowMode::Maximized)
        REQUIRE_GLFW(glfwMaximizeWindow(w));

    //----------------------------------------------------------------
    //
    // Callbacks.
    //
    //----------------------------------------------------------------

    REQUIRE_GLFW(glfwSetWindowUserPointer(w, this));

    REQUIRE_GLFW(glfwSetWindowRefreshCallback(w, glfwRefreshCallback));
    REQUIRE_GLFW(glfwSetKeyCallback(w, glfwKeyCallback));
    REQUIRE_GLFW(glfwSetCursorPosCallback(w, glfwCursorPosCallback));
    REQUIRE_GLFW(glfwSetMouseButtonCallback(w, glfwMouseButtonCallback));
    REQUIRE_GLFW(glfwSetScrollCallback(w, glfwScrollCallback));
    REQUIRE_GLFW(glfwSetFramebufferSizeCallback(w, glfwFramebufferSizeCallback));

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    cleanWindow.cancel();
    baseWindow = w;

    returnTrue;
}

//================================================================
//
// WindowGLFW::callbackShell
//
//================================================================

template <typename Lambda>
stdbool WindowGLFW::callbackShell(const EventReceivers& receivers, const Lambda& lambda, stdPars(Kit))
{
    //----------------------------------------------------------------
    //
    // Refresh callback.
    //
    //----------------------------------------------------------------

    auto refreshHandlerImpl = RefreshHandler::O | [&] ()
    {
        receivers.refreshReceiver(stdPassThru);
    };

    ////

    refreshHandler = &refreshHandlerImpl;
    REMEMBER_CLEANUP(refreshHandler = nullptr);

    //----------------------------------------------------------------
    //
    // Key callback.
    //
    //----------------------------------------------------------------

    auto keyHandlerImpl = KeyHandler::O | [&] (auto& event)
    {
        receivers.keyReceiver(event, stdPassThru);
    };

    ////

    keyHandler = &keyHandlerImpl;
    REMEMBER_CLEANUP(keyHandler = nullptr);

    //----------------------------------------------------------------
    //
    // Mouse move callback.
    //
    //----------------------------------------------------------------

    auto mouseMoveHandlerImpl = MouseMoveHandler::O | [&] (auto& event)
    {
        receivers.mouseMoveReceiver(event, stdPassThru);
    };

    ////

    mouseMoveHandler = &mouseMoveHandlerImpl;
    REMEMBER_CLEANUP(mouseMoveHandler = nullptr);

    //----------------------------------------------------------------
    //
    // Mouse button callback.
    //
    //----------------------------------------------------------------

    auto mouseButtonHandlerImpl = MouseButtonHandler::O | [&] (auto& event)
    {
        receivers.mouseButtonReceiver(event, stdPassThru);
    };

    ////

    mouseButtonHandler = &mouseButtonHandlerImpl;
    REMEMBER_CLEANUP(mouseButtonHandler = nullptr);

    //----------------------------------------------------------------
    //
    // Scroll callback.
    //
    //----------------------------------------------------------------

    auto scrollHandlerImpl = ScrollHandler::O | [&] (auto& event)
    {
        receivers.scrollReceiver(event, stdPassThru);
    };

    ////

    scrollHandler = &scrollHandlerImpl;
    REMEMBER_CLEANUP(scrollHandler = nullptr);

    //----------------------------------------------------------------
    //
    // Resize callback.
    //
    //----------------------------------------------------------------

    auto resizeHandlerImpl = ResizeHandler::O | [&] (auto& event)
    {
        receivers.resizeReceiver(event, stdPassThru);
    };

    ////

    resizeHandler = &resizeHandlerImpl;
    REMEMBER_CLEANUP(resizeHandler = nullptr);

    //----------------------------------------------------------------
    //
    // The action.
    //
    //----------------------------------------------------------------

    require(lambda(stdPass));

    ////

    returnTrue;
}

//================================================================
//
// WindowGLFW::getEvents
//
//================================================================

stdbool WindowGLFW::getEvents(bool waitEvents, const OptionalObject<uint32>& waitTimeoutMs, const EventReceivers& receivers, stdPars(Kit))
{
    auto action = [&] (stdPars(auto))
    {
        if_not (waitEvents)
        {
            REQUIRE_GLFW(glfwPollEvents());
            returnTrue;
        }

        ////

        auto notifier = timeoutHelper::Callback::O | [&] ()
        {
            glfwPostEmptyEvent();
        };

        if (waitTimeoutMs)
            timeoutHelper->setTask(*waitTimeoutMs, notifier);

        REMEMBER_CLEANUP(if (waitTimeoutMs) timeoutHelper->cancelTask());

        REQUIRE_GLFW(glfwWaitEvents());

        ////

        returnTrue;
    };

    ////

    require(callbackShell(receivers, action, stdPass));

    returnTrue;
}

//================================================================
//
// WindowGLFW::getWindowLocation
//
//================================================================

stdbool WindowGLFW::getWindowLocation(WindowLocation& location, stdPars(Kit))
{
    REQUIRE(baseWindow);

    ////

    location = {};

    ////

    location.mode = WindowMode::Normal;

    if (glfwGetWindowAttrib(baseWindow, GLFW_ICONIFIED))
        location.mode = WindowMode::Minimized;

    if (glfwGetWindowAttrib(baseWindow, GLFW_MAXIMIZED))
        location.mode = WindowMode::Maximized;

    if (glfwGetWindowMonitor(baseWindow))
        location.mode = WindowMode::FullScreen;

    ////

    location.decorated = glfwGetWindowAttrib(baseWindow, GLFW_DECORATED) == GL_TRUE;

    ////

    REQUIRE_GLFW(glfwGetWindowPos(baseWindow, &location.pos.X, &location.pos.Y));
    REQUIRE_GLFW(glfwGetWindowSize(baseWindow, &location.size.X, &location.size.Y));

    ////

    returnTrue;
}

//================================================================
//
// WindowGLFW::setWindowLocationBody
//
//================================================================

stdbool WindowGLFW::setWindowLocation(const WindowLocation& location, stdPars(Kit))
{
    REQUIRE(baseWindow);

    ////

    WindowLocation current;
    require(getWindowLocation(current, stdPass));

    ////

    if_not (location.decorated == current.decorated)
        REQUIRE_GLFW(glfwSetWindowAttrib(baseWindow, GLFW_DECORATED, location.decorated ? GL_TRUE : GL_FALSE));

    ////

    if_not (current.mode == location.mode)
    {
        if (location.mode != WindowMode::FullScreen)
            REQUIRE_GLFW(glfwSetWindowMonitor(baseWindow, nullptr, location.pos.X, location.pos.Y, location.size.X, location.size.Y, 0));
        else
        {
            auto monitor = glfwGetPrimaryMonitor();
            REQUIRE_GLFW("glfwGetPrimaryMonitor");
            REQUIRE(monitor);

            const GLFWvidmode* videoMode = glfwGetVideoMode(monitor);
            REQUIRE_GLFW("glfwGetVideoMode");
            REQUIRE(videoMode);

            REQUIRE_GLFW(glfwSetWindowMonitor(baseWindow, monitor, 0, 0, videoMode->width, videoMode->height, videoMode->refreshRate));
        }

        ////

        if (location.mode == WindowMode::Minimized)
            REQUIRE_GLFW(glfwIconifyWindow(baseWindow));

        if (location.mode == WindowMode::Normal)
            REQUIRE_GLFW(glfwRestoreWindow(baseWindow));

        if (location.mode == WindowMode::Maximized)
            REQUIRE_GLFW(glfwMaximizeWindow(baseWindow));
    }

    ////

    if (location.mode == WindowMode::Normal)
    {
        if_not (current.pos == location.pos)
            REQUIRE_GLFW(glfwSetWindowPos(baseWindow, location.pos.X, location.pos.Y));

        if_not (current.size == location.size)
            REQUIRE_GLFW(glfwSetWindowSize(baseWindow, location.size.X, location.size.Y));
    }

    ////

    REQUIRE_GLFW(glfwSwapInterval(location.verticalSync ? 1 : 0));

    ////

    returnTrue;
}

//================================================================
//
// WindowManagerGLFW::init
//
//================================================================

stdbool WindowManagerGLFW::init(stdPars(Kit))
{
    if_not (initialized)
    {
        glfwSetErrorCallback(glfwErrorCallback);
        REQUIRE_GLFW(glfwInit() == GL_TRUE);
        initialized = true;
    }

    returnTrue;
}

//================================================================
//
// WindowManagerGLFW::deinit
//
//================================================================

void WindowManagerGLFW::deinit()
{
    if (initialized)
    {
        glfwSetErrorCallback(nullptr);
        glfwTerminate();
        initialized = false;
    }
}

//================================================================
//
// WindowManagerGLFW::postEmptyEvent
//
//================================================================

void WindowManagerGLFW::postEmptyEvent()
{
    glfwPostEmptyEvent();
}

//================================================================
//
// WindowManagerGLFW::getCurrentDisplayResolution
//
//================================================================

stdbool WindowManagerGLFW::getCurrentDisplayResolution(Point<Space>& result, stdPars(Kit))
{
    auto primaryMonitor = glfwGetPrimaryMonitor();
    REQUIRE_GLFW("glfwGetPrimaryMonitor");
    REQUIRE(primaryMonitor);

    auto primaryMode = glfwGetVideoMode(primaryMonitor);
    REQUIRE_GLFW("glfwGetVideoMode");
    REQUIRE(primaryMode);

    result = point(primaryMode->width, primaryMode->height);
    REQUIRE(result >= 0);

    returnTrue;
}

//================================================================
//
// WindowManagerGLFW::createWindow
//
//================================================================

stdbool WindowManagerGLFW::createWindow(UniquePtr<Window>& window, const WindowCreationArgs& par, stdPars(Kit))
{
    REQUIRE(initialized);

    ////

    window.reset();

    auto newWindow = makeUnique<WindowGLFW>();
    REMEMBER_CLEANUP_EX(cleanup, newWindow.reset());

    ////

    require(newWindow->open(par, stdPass));

    ////

    window = move(newWindow);
    cleanup.cancel();

    ////

    returnTrue;
}

//================================================================
//
// ContextBinderGLFW
//
//================================================================

struct ContextBinderGLFW : public ContextBinder
{
    virtual stdbool bind(stdPars(Kit))
    {
        REQUIRE(window);
        REQUIRE(glfwGetCurrentContext() == nullptr);
        REQUIRE_GLFW(glfwMakeContextCurrent(window));
        returnTrue;
    }

    virtual stdbool unbind(stdPars(Kit))
    {
        REQUIRE_GLFW(glfwMakeContextCurrent(nullptr));
        returnTrue;
    }

    GLFWwindow* window = nullptr;
};

//================================================================
//
// WindowManagerGLFW::createOffscreenGLContext
//
//================================================================

stdbool WindowManagerGLFW::createOffscreenGLContext(UniquePtr<ContextBinder>& context, stdPars(Kit))
{
    REQUIRE_GLFW(glfwDefaultWindowHints());
    REQUIRE_GLFW(glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE));
    REQUIRE_GLFW(glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API));
    REQUIRE_GLFW(glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API));

    GLFWwindow* w = glfwCreateWindow(1, 1, "Offscreen Context", nullptr, nullptr);
    REQUIRE_GLFW("glfwCreateWindow");
    REQUIRE(w != 0);
    REMEMBER_CLEANUP_EX(windowCleanup, glfwDestroyWindow(w));

    ////

    auto p = makeUnique<ContextBinderGLFW>();
    p->window = w;
    context = move(p);

    ////

    windowCleanup.cancel();
    returnTrue;
}
