#include "debugBridgeTest.h"

#include <vector>
#include <deque>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "storage/rememberCleanup.h"

namespace debugBridgeTest {

using namespace std;

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Utils.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// appendMove
//
//================================================================

template <typename Src, typename Dst>
void appendMove(Dst& dst, Src&& src)
{
    if (src.size())
    {
        if (dst.empty())
            dst = move(src);
        else
        {
            dst.insert(dst.end(), make_move_iterator(src.begin()), make_move_iterator(src.end()));
            src.clear();
        }
    }
}

//================================================================
//
// String
//
//================================================================

using String = basic_string<Char>;

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Config.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ConfigSupportTest
//
//================================================================

class ConfigSupportTest : public ConfigSupport
{

public:

    ConfigSupportTest()
    {
        const char* tempDir = getenv("TMPDIR");

        if (!tempDir)
            tempDir = getenv("TEMP");

        if (!tempDir)
            tempDir = ".";

        filename = string(tempDir) + "/test.cfg";
    }

public:

    virtual void saveConfig(const Array<const Char>& config);

    virtual void loadConfig(ConfigReceiver& configReceiver);

    virtual void editConfig(const Array<const Char>& config, ConfigReceiver& configReceiver);

private:

    void error() {throw runtime_error("Config support error");}

private:

    string filename;

};

//================================================================
//
// ConfigSupportTest::saveConfig
//
//================================================================

void ConfigSupportTest::saveConfig(const Array<const Char>& config)
{
    cout << "BRIDGE: Saving config to " << filename << endl;

    ofstream stream(filename.c_str());
    stream.write(config.ptr, config.size);
    stream.close();

    if (!stream)
        {cerr << "BRIDGE: Cannot write config to " << filename << endl; error();}
}

//================================================================
//
// ConfigSupportTest::loadConfig
//
//================================================================

void ConfigSupportTest::loadConfig(ConfigReceiver& configReceiver)
{
    cout << "BRIDGE: Loading config from " << filename << endl;

    ////

    ifstream stream(filename.c_str());
    stringstream buffer;
    buffer << stream.rdbuf();

    if (!stream)
        {cerr << "BRIDGE: Cannot read config from " << filename << endl; error();}

    auto str = buffer.str();

    ////

    configReceiver.receive({str.data(), str.size()});
}

//================================================================
//
// ConfigSupportTest::editConfig
//
//================================================================

void ConfigSupportTest::editConfig(const Array<const Char>& config, ConfigReceiver& configReceiver)
{
    cout << "BRIDGE: Editing config: " << endl;

    ////

    saveConfig(config);

    ////

    cout << "BRIDGE: The config file is saved to " << filename << endl;
    cout << "BRIDGE: Please edit it and then press Enter" << endl;

    cin.get();

    ////

    loadConfig(configReceiver);
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Text console.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// MessageRec
//
//================================================================

struct MessageRec
{
    String text;
    MessageKind kind;

    MessageRec(const Char* text, MessageKind kind)
        : text{text}, kind{kind} {}
};

//================================================================
//
// MessageConsoleTest
//
//================================================================

class MessageConsoleTest : public MessageConsole
{

public:

    void clear()
    {
        clearHappened = true;
        buffer.clear();
    }

    void add(const Char* text, MessageKind kind)
    {
        buffer.emplace_back(text, kind);
    }

    void update();

private:

    bool clearHappened = false;
    deque<MessageRec> buffer;

    deque<MessageRec> actualBuffer;

};

//================================================================
//
// MessageConsoleTest::update
//
//================================================================

void MessageConsoleTest::update()
{
    for (auto& message: buffer)
        (message.kind == MessageKind::Info ? cout: cerr) << "BRIDGE: " << message.text << endl;

    if (clearHappened)
        actualBuffer.clear();

    appendMove(actualBuffer, buffer);
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Action setup.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ActionRec
//
//================================================================

struct ActionRec
{
    ActionId id;
    String key;
    String name;
    String description;
};

//================================================================
//
// ActionSetupTest
//
//================================================================

class ActionSetupTest : public ActionSetup
{

public:

    void clear()
    {
        clearHappened = true;
        buffer.clear();
    }

    void add(const ActionParams& action)
    {
        buffer.emplace_back(ActionRec{action.id, action.key, action.name, action.description});
    }

    void update();

private:

    bool clearHappened = false;
    deque<ActionRec> buffer;

private:

    deque<ActionRec> actualBuffer;

};

//================================================================
//
// ActionSetupTest::update
//
//================================================================

void ActionSetupTest::update()
{
    for (auto& rec: buffer)
        cout << "BRIDGE: Action setup " << 
        "ID=" << rec.id << " " <<
        "name='" << rec.name << "' " <<
        "key='" << rec.key << "' " <<
        "desc='" << rec.description << "'" << 
        endl; 

    if (clearHappened)
        actualBuffer.clear();

    appendMove(actualBuffer, buffer);
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Action receiving.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ActionReceivingTest
//
//================================================================

struct ActionReceivingTest : public ActionReceiving
{

public:

    virtual void getActions(ActionReceiver& receiver) 
    {
        ActionId actions[] = 
        {
            0x3111BAE9, 
            actionId::MouseLeftDown,
            actionId::EditConfig,
            actionId::ResetupActions
        };

        receiver.process({actions, sizeof(actions) / sizeof(ActionId)});
    }

private:

};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Video overlay.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// VideoOverlayTest
//
//================================================================

class VideoOverlayTest : public VideoOverlay
{

public:

    virtual void set(const ImagePoint& size, ImageProvider& imageProvider, StringPtr description);

    virtual void clear();

    virtual void update();

    virtual void getUserPoint(bool& valid, ImagePoint& pos)
        {valid = true; pos = {1, 2};}

private:

    bool bufferSet = false;
    ImagePoint bufferSize = {0, 0};
    vector<uint8_t> buffer;

};

//================================================================
//
// VideoOverlayTest::set
//
//================================================================

void VideoOverlayTest::set(const ImagePoint& size, ImageProvider& imageProvider, StringPtr description)
{
    cout << "BRIDGE: Video overlay: Setting image " << size.X << " x " << size.Y << ", description '" << description << "'" << endl;

    bufferSet = false;

    auto monoSizeX = size.X * 3;
    ImageSpace monoPitch = monoSizeX + 5;
    buffer.resize(monoPitch * size.Y);
    imageProvider.saveBgr24({&buffer[0], monoPitch, ImagePoint{monoSizeX, size.Y}});

    bufferSet = true;
    bufferSize = size;
}

//================================================================
//
// VideoOverlayTest::clear
//
//================================================================

void VideoOverlayTest::clear()
{
    cout << "BRIDGE: Video overlay: Clearing the image" << endl;

    bufferSet = false;
    bufferSize = {0, 0};
}

//================================================================
//
// VideoOverlayTest::update
//
//================================================================

void VideoOverlayTest::update()
{
    if (bufferSet)
        cout << "BRIDGE: Video overlay: Updating image " << bufferSize.X << " x " << bufferSize.Y << endl;

    bufferSet = false;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Full bridge.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// DebugBridgeTest
//
//================================================================

class DebugBridgeTest : public DebugBridge
{

public:

    virtual bool active() {return true;}

    virtual ConfigSupport* configSupport() {return &configSupportTest;}
    virtual ActionSetup* actionSetup() {return &actionSetupTest;}
    virtual ActionReceiving* actionReceiving() {return &actionReceivingTest;}
    virtual MessageConsole* globalConsole() {return &globalConsoleTest;}
    virtual MessageConsole* localConsole() {return &localConsoleTest;}
    virtual VideoOverlay* videoOverlay() {return &videoOverlayTest;}

private:

    ConfigSupportTest configSupportTest;
    ActionSetupTest actionSetupTest;
    ActionReceivingTest actionReceivingTest;
    MessageConsoleTest globalConsoleTest;
    MessageConsoleTest localConsoleTest;
    VideoOverlayTest videoOverlayTest;

};

//----------------------------------------------------------------

UniquePtr<DebugBridge> create()
{
    return makeUnique<DebugBridgeTest>();
}

//----------------------------------------------------------------

}
