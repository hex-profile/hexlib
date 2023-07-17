#pragma once

#include "charType/charType.h"
#include "numbers/int/intBase.h"

//================================================================
//
// ActionId
//
// The ID of user-defined action (signal).
//
// When an action is added, the client specifies its ID;
// later when the user sends the action signal, its ID is passed
// back to the client process function.
//
//================================================================

using ActionId = uint32;

//================================================================
//
// ActionKey
//
//----------------------------------------------------------------
//
// The name of the action hotkey, for example, "Ctrl+W".
//
// The format is:
//
// * Several "Ctrl+", "Shift+" or "Alt+" prefixes.
// * Key name.
//
// Key name can be:
//
// * Letter from "A" to "Z".
// * Digit from "0" to "9".
// * Symbol: ' + , - . / ; = [ \ ] `
// * Functional key name from "F1" to "F24".
// * One of special key names.
//
// Special key names are:
//
// "Alt"           "Home"          "Num 5"         "Right"
// "Application"   "Ins"           "Num 6"         "Right Alt"
// "Backspace"     "Insert"        "Num 7"         "Right Ctrl"
// "BkSp"          "Left"          "Num 8"         "Right Shift"
// "Break"         "Left Windows"  "Num 9"         "Right Windows"
// "Caps Lock"     "Num *"         "Num Del"       "Scroll Lock"
// "Ctrl"          "Num +"         "Num Enter"     "Shift"
// "Del"           "Num -"         "Num Lock"      "Space"
// "Delete"        "Num /"         "Page Down"     "Sys Req"
// "Down"          "Num 0"         "Page Up"       "Tab"
// "End"           "Num 1"         "Pause"         "Up"
// "Enter"         "Num 2"         "PgDn"
// "Esc"           "Num 3"         "PgUp"
// "Help"          "Num 4"         "Prnt Scrn"
//
//================================================================

using ActionKey = const CharType*;
