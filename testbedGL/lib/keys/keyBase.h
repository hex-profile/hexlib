#pragma once

#include "numbers/int/intBase.h"

//================================================================
//
// KeyCode codes
//
//================================================================

using KeyCode = int32;

//----------------------------------------------------------------

namespace Key
{
    const KeyCode None = 0;

    const KeyCode Space = 32;
    const KeyCode Apostrophe = 39; // '

    const KeyCode Comma = 44; // ,
    const KeyCode Minus = 45; // -
    const KeyCode Period = 46; // .
    const KeyCode Slash = 47; // /
    const KeyCode _0 = 48;
    const KeyCode _1 = 49;
    const KeyCode _2 = 50;
    const KeyCode _3 = 51;
    const KeyCode _4 = 52;
    const KeyCode _5 = 53;
    const KeyCode _6 = 54;
    const KeyCode _7 = 55;
    const KeyCode _8 = 56;
    const KeyCode _9 = 57;
    const KeyCode Semicolon = 59; // ;
    const KeyCode Equal = 61; // =
    const KeyCode A = 65;
    const KeyCode B = 66;
    const KeyCode C = 67;
    const KeyCode D = 68;
    const KeyCode E = 69;
    const KeyCode F = 70;
    const KeyCode G = 71;
    const KeyCode H = 72;
    const KeyCode I = 73;
    const KeyCode J = 74;
    const KeyCode K = 75;
    const KeyCode L = 76;
    const KeyCode M = 77;
    const KeyCode N = 78;
    const KeyCode O = 79;
    const KeyCode P = 80;
    const KeyCode Q = 81;
    const KeyCode R = 82;
    const KeyCode S = 83;
    const KeyCode T = 84;
    const KeyCode U = 85;
    const KeyCode V = 86;
    const KeyCode W = 87;
    const KeyCode X = 88;
    const KeyCode Y = 89;
    const KeyCode Z = 90;
    const KeyCode LeftBracket = 91; // [
    const KeyCode Backslash = 92; // '\'
    const KeyCode RightBracket = 93; // ]
    const KeyCode GraveAccent = 96; // `
    const KeyCode World1 = 161; // non-US #1
    const KeyCode World2 = 162; // non-US #2

    const KeyCode Escape = 256;
    const KeyCode Enter = 257;
    const KeyCode Tab = 258;
    const KeyCode Backspace = 259;
    const KeyCode Insert = 260;
    const KeyCode Delete = 261;
    const KeyCode Right = 262;
    const KeyCode Left = 263;
    const KeyCode Down = 264;
    const KeyCode Up = 265;
    const KeyCode PageUp = 266;
    const KeyCode PageDown = 267;
    const KeyCode Home = 268;
    const KeyCode End = 269;
    const KeyCode CapsLock = 280;
    const KeyCode ScrollLock = 281;
    const KeyCode NumLock = 282;
    const KeyCode PrintScreen = 283;
    const KeyCode Pause = 284;

    const KeyCode F1 = 290;
    const KeyCode F2 = 291;
    const KeyCode F3 = 292;
    const KeyCode F4 = 293;
    const KeyCode F5 = 294;
    const KeyCode F6 = 295;
    const KeyCode F7 = 296;
    const KeyCode F8 = 297;
    const KeyCode F9 = 298;
    const KeyCode F10 = 299;
    const KeyCode F11 = 300;
    const KeyCode F12 = 301;
    const KeyCode F13 = 302;
    const KeyCode F14 = 303;
    const KeyCode F15 = 304;
    const KeyCode F16 = 305;
    const KeyCode F17 = 306;
    const KeyCode F18 = 307;
    const KeyCode F19 = 308;
    const KeyCode F20 = 309;
    const KeyCode F21 = 310;
    const KeyCode F22 = 311;
    const KeyCode F23 = 312;
    const KeyCode F24 = 313;
    const KeyCode F25 = 314;

    const KeyCode Kp0 = 320;
    const KeyCode Kp1 = 321;
    const KeyCode Kp2 = 322;
    const KeyCode Kp3 = 323;
    const KeyCode Kp4 = 324;
    const KeyCode Kp5 = 325;
    const KeyCode Kp6 = 326;
    const KeyCode Kp7 = 327;
    const KeyCode Kp8 = 328;
    const KeyCode Kp9 = 329;

    const KeyCode KpDecimal = 330;
    const KeyCode KpDivide = 331;
    const KeyCode KpMultiply = 332;
    const KeyCode KpSubtract = 333;
    const KeyCode KpAdd = 334;
    const KeyCode KpEnter = 335;
    const KeyCode KpEqual = 336;

    const KeyCode LeftShift = 340;
    const KeyCode LeftControl = 341;
    const KeyCode LeftAlt = 342;
    const KeyCode LeftSuper = 343;
    const KeyCode RightShift = 344;
    const KeyCode RightControl = 345;
    const KeyCode RightAlt = 346;
    const KeyCode RightSuper = 347;
    const KeyCode Menu = 348;
}

//================================================================
//
// KeyCode modifiers
//
//================================================================

using KeyModifiers = int32;

////

namespace KeyModifier
{
    const KeyModifiers Shift = 1;
    const KeyModifiers Control = 2;
    const KeyModifiers Alt = 4;
    const KeyModifiers Super = 8;

    const KeyModifiers Max = 0xF;
}

//================================================================
//
// KeyRec
//
//================================================================

struct KeyRec
{
    KeyCode code = Key::None;
    KeyModifiers modifiers = 0;
};

//================================================================
//
// KeyAction
//
//================================================================

enum class KeyAction {Press, Release, Repeat};

//================================================================
//
// KeyEvent
//
//================================================================

struct KeyEvent : public KeyRec
{
    KeyAction action = KeyAction::Press;
};
