#pragma once

#include "storage/smartPtr.h"
#include "simpleString/simpleString.h"

//================================================================
//
// EditRequest
//
//================================================================

class EditRequest
{

public:

    static UniquePtr<EditRequest> create()
        {return makeUnique<EditRequest>();}

    virtual ~EditRequest() {}

public:

    void clearMemory()
    {
        editor.clear();
        helpMessage.clear();
    }

    bool hasUpdates() const
    {
        return editor.size() != 0;
    }

    void reset()
    {
        editor.clear();
        helpMessage.clear();
    }

    bool absorb(EditRequest& that)
    {
        if (that.hasUpdates())
            moveFrom(that);

        return true;
    }

    void moveFrom(EditRequest& that)
    {
        exchange(editor, that.editor);
        exchange(helpMessage, that.helpMessage);

        that.reset();
    }

    bool addRequest(const CharArray& editor, const CharArray& helpMessage)
    {
        this->editor = editor;
        this->helpMessage = helpMessage;

        return def(this->editor, this->helpMessage);
    }

    auto getEditor() const
        {return editor.str();}

    auto getHelpMessage() const
        {return helpMessage.str();}

private:

    SimpleString editor;
    SimpleString helpMessage;

};
