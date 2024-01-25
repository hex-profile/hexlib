#include "parseKey.h"

#include "prepTools/prepFor.h"
#include "data/space.h"

//================================================================
//
// skipText
//
//================================================================

template <bool lastText, typename TextArray>
inline bool skipText(const CharType*& bufferPtr, size_t& bufferCount, const TextArray& textArray)
{
    static const size_t textSize = COMPILE_ARRAY_SIZE(textArray) - 1;
    COMPILE_ASSERT(textSize >= 1);
    const CharType* textPtr = textArray;

    ////

    if (lastText)
        ensure(textSize == bufferCount);
    else
        ensure(textSize <= bufferCount);

    ////

    #define MAX_COUNT 16
    COMPILE_ASSERT(textSize <= MAX_COUNT);

    #define TMP_MACRO(i, _) \
        if (i < textSize) \
            if (bufferPtr[i] != textPtr[i]) \
                return false;

    PREP_FOR(MAX_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    bufferPtr += textSize;
    bufferCount -= textSize;
    return true;
}

//================================================================
//
// parseFunctionalKey
//
//================================================================

inline bool parseFunctionalKey(const CharType* ptr, size_t count, KeyCode& code)
{
    ensure(count >= 2 && count <= 3);
    ensure(ptr[0] == 'F');

    ensure(ptr[1] >= '1' && ptr[1] <= '9');
    Space index = (ptr[1] - '0');

    if (count == 3)
    {
        ensure(ptr[2] >= '0' && ptr[2] <= '9');
        index = 10 * index + (ptr[2] - '0');
    }

    KeyCode result = Key::F1 + (index - 1);
    ensure(result >= Key::F1 && result <= Key::F25);

    code = result;
    return true;
}

//================================================================
//
// parseKey
//
//================================================================

bool parseKey(const CharArray& buffer, KeyRec& result)
{
    const CharType* ptr = buffer.ptr;
    size_t count = buffer.size;

    //----------------------------------------------------------------
    //
    // Empty.
    //
    //----------------------------------------------------------------

    if (count == 0)
    {
        result = {};
        return true;
    }

    //----------------------------------------------------------------
    //
    // Modifiers.
    //
    //----------------------------------------------------------------

    KeyModifiers keyModifiers = 0;

    ////

    for (; ;)
    {
        if (skipText<false>(ptr, count, CT("Shift+")))
            {keyModifiers |= KeyModifier::Shift; continue;}

        if (skipText<false>(ptr, count, CT("Ctrl+")))
            {keyModifiers |= KeyModifier::Control; continue;}

        if (skipText<false>(ptr, count, CT("Alt+")))
            {keyModifiers |= KeyModifier::Alt; continue;}

        if (skipText<false>(ptr, count, CT("Super+")))
            {keyModifiers |= KeyModifier::Super; continue;}

        break;
    }

    //----------------------------------------------------------------
    //
    // Key name: Single char.
    //
    //----------------------------------------------------------------

    auto keyCode = Key::None;

    ////

    ensure(count >= 1);

    ////

    if (count == 1)
    {
        CharType c = *ptr;

        if (c >= 'A' && c <= 'Z')
            keyCode = Key::A + (c - 'A');

        else if (c >= '0' && c <= '9')
            keyCode = Key::_0 + (c - '0');

        else
        {
            switch (c)
            {
                #define TMP_MACRO(key, code) \
                    case (key): keyCode = (code); break;

                TMP_MACRO('\'', Key::Apostrophe)
                TMP_MACRO(',', Key::Comma)
                TMP_MACRO('-', Key::Minus)
                TMP_MACRO('.', Key::Period)
                TMP_MACRO('/', Key::Slash)
                TMP_MACRO(';', Key::Semicolon)
                TMP_MACRO('=', Key::Equal)
                TMP_MACRO('[', Key::LeftBracket)
                TMP_MACRO('\\', Key::Backslash)
                TMP_MACRO(']', Key::RightBracket)
                TMP_MACRO('`', Key::GraveAccent)

                #undef TMP_MACRO

                default: return false;
            }


        }

    }

    //----------------------------------------------------------------
    //
    // Functional keys.
    //
    //----------------------------------------------------------------

    else if (parseFunctionalKey(ptr, count, keyCode))
    {
    }

    //----------------------------------------------------------------
    //
    // Key name: Multi char.
    //
    // Names are compatible with AT format (Borland)
    //
    //----------------------------------------------------------------

    else
    {

        #define TMP_MACRO(name, code) \
            else if (skipText<true>(ptr, count, CT(name))) keyCode = (code);

        ////

        if (0) ;

        TMP_MACRO("Space", Key::Space)

        TMP_MACRO("Esc", Key::Escape)

        TMP_MACRO("Enter", Key::Enter)
        TMP_MACRO("Tab", Key::Tab)

        TMP_MACRO("Backspace", Key::Backspace)
        TMP_MACRO("BkSp", Key::Backspace)

        TMP_MACRO("Insert", Key::Insert)
        TMP_MACRO("Ins", Key::Insert)

        TMP_MACRO("Delete", Key::Delete)
        TMP_MACRO("Del", Key::Delete)

        TMP_MACRO("Right", Key::Right)
        TMP_MACRO("Left", Key::Left)
        TMP_MACRO("Down", Key::Down)
        TMP_MACRO("Up", Key::Up)

        TMP_MACRO("PgUp", Key::PageUp)
        TMP_MACRO("PgDn", Key::PageDown)
        TMP_MACRO("Page Up", Key::PageUp)
        TMP_MACRO("Page Down", Key::PageDown)

        TMP_MACRO("Home", Key::Home)
        TMP_MACRO("End", Key::End)

        TMP_MACRO("Caps Lock", Key::CapsLock)
        TMP_MACRO("Scroll Lock", Key::ScrollLock)
        TMP_MACRO("Num Lock", Key::NumLock)

        TMP_MACRO("Prnt Scrn", Key::PrintScreen)
        TMP_MACRO("Sys Req", Key::PrintScreen)

        TMP_MACRO("Pause", Key::Pause)
        TMP_MACRO("Break", Key::Pause)

        TMP_MACRO("Num 0", Key::Kp0)
        TMP_MACRO("Num 1", Key::Kp1)
        TMP_MACRO("Num 2", Key::Kp2)
        TMP_MACRO("Num 3", Key::Kp3)
        TMP_MACRO("Num 4", Key::Kp4)
        TMP_MACRO("Num 5", Key::Kp5)
        TMP_MACRO("Num 6", Key::Kp6)
        TMP_MACRO("Num 7", Key::Kp7)
        TMP_MACRO("Num 8", Key::Kp8)
        TMP_MACRO("Num 9", Key::Kp9)

        TMP_MACRO("Num /", Key::KpDivide)
        TMP_MACRO("Num *", Key::KpMultiply)
        TMP_MACRO("Num -", Key::KpSubtract)
        TMP_MACRO("Num +", Key::KpAdd)

        TMP_MACRO("Num Del", Key::KpDecimal)
        TMP_MACRO("Num Enter", Key::KpEnter)

        TMP_MACRO("Shift", Key::LeftShift)
        TMP_MACRO("Ctrl", Key::LeftControl)
        TMP_MACRO("Alt", Key::LeftAlt)
        TMP_MACRO("Super", Key::LeftSuper)

        TMP_MACRO("Right Shift", Key::RightShift)
        TMP_MACRO("Right Ctrl", Key::RightControl)
        TMP_MACRO("Right Alt", Key::RightAlt)
        TMP_MACRO("Right Super", Key::RightSuper)

        TMP_MACRO("Application", Key::Menu)

        else return false;

        #undef TMP_MACRO

    }

    //----------------------------------------------------------------
    //
    // Success
    //
    //----------------------------------------------------------------

    result.code = keyCode;
    result.modifiers = keyModifiers;

    return true;
}
