#pragma once

#include <memory>
#include <stdint.h>

namespace debugBridge {

//================================================================
//
// Errors are returned via exceptions (!)
//
// Usage: A client uses a debug bridge provider in its create/destroy/process functions.
//
//================================================================

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Base types.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// VirtualDestructor
//
// Our guys like to destroy instances by a pointer to base interface.
//
//================================================================

struct VirtualDestructor
{
    virtual ~VirtualDestructor() =default;
};

//================================================================
//
// Strings.
//
//================================================================

using Char = char;
using StringPtr = const Char*;

//================================================================
//
// ImageSpace
//
// A signed type for the image address space, basically the same as ptrdiff_t.
//
// The type can hold an image size in bytes, and signed difference between
// pointers to any two elements of the same image.
//
// Currently, an image is limited to 2GB.
//
//================================================================

using ImageSpace = int32_t;

//================================================================
//
// ImagePoint
//
//================================================================

struct ImagePoint
{
    ImageSpace X;
    ImageSpace Y;
};

//================================================================
//
// PixelRgb32
//
// 32-bit RGB pixel type.
//
// Bit allocation:
//
// [00..07] Blue value
// [08..15] Green value
// [16..23] Red value
// [24..31] Zero
//
//================================================================

using PixelRgb32 = uint32_t;

//================================================================
//
// PixelMono
//
//================================================================

using PixelMono = uint8_t;

//================================================================
//
// ArrayRef<Type>
//
// Array memory layout description: ptr and size.
//
// ptr:
// Points to 0th array element. Can be NULL if the array is empty.
//
// size:
// The array size. Always >= 0.
// If size is zero, the array is empty.
//
//================================================================

template <typename Type>
struct ArrayRef
{
    Type* ptr;
    size_t size;

    inline Type* begin() const
        {return ptr;}

    inline Type* end() const 
        {return ptr + size;}

    inline Type& operator[](size_t i) const
        {return ptr[i];}
};

//================================================================
//
// ImageRef<Type>
//
// Image memory layout: base pointer, pitch and dimensions.
//
//----------------------------------------------------------------
//
// memPtr:
// Points to (0, 0) element. Can be undefined if the matrix is empty.
//
// memPitch:
// The difference of pointers to (X, Y+1) and (X, Y) elements.
// The difference is expressed in elements (not bytes). Can be negative.
//
// sizeX, sizeY:
// The width and height of the matrix. Both are >= 0.
// If either of them is zero, the matrix is empty.
//
//================================================================

template <typename Pixel>
struct ImageRef
{
    Pixel* ptr;
    ImageSpace pitch;
    ImagePoint size;
};

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
// ConfigReceiver
//
// Here the receive function is assumed to be called once.
// The next call will replace the config.
//
//================================================================

struct ConfigReceiver : public VirtualDestructor
{
    virtual void receive(ArrayRef<const Char> config) =0;
};

//================================================================
//
// configReceiverByLambda
//
//================================================================

template <typename Lambda>
class ConfigReceiverByLambda : public ConfigReceiver
{

public:

    ConfigReceiverByLambda(const Lambda& lambda)
        : lambda{lambda} {}

    virtual void receive(ArrayRef<const Char> config) 
        {lambda(config);}

private:

    Lambda lambda;

};

//----------------------------------------------------------------

template <typename Lambda>
inline auto configReceiverByLambda(const Lambda& lambda)
{
    return ConfigReceiverByLambda<Lambda>{lambda};
}

//================================================================
//
// ConfigSupport
//
//================================================================

struct ConfigSupport : public VirtualDestructor
{
    //
    // When an algo module receives a signal to save config,
    // it exports the current state to a string and calls the function.
    //

    virtual void saveConfig(ArrayRef<const Char> config) =0;

    //
    // When an algo module receives a signal to load config,
    // it calls the function to get a new config
    // and imports it into the current state.
    //
    // The algo module implementation supports partial loading:
    // if some internal vars are missing, they remain unchanged.
    //

    virtual void loadConfig(ConfigReceiver& configReceiver) =0;

    //
    // When an algo module receives a signal to edit config,
    // it exports its current state to a string and calls the function
    // (equivalent to save config).
    //
    // The function performs long editing inside, then calls the config receiver
    // with the new config (equivalent to load config).
    //
    // An algo module is blocked during the call.
    //

    virtual void editConfig(ArrayRef<const Char> config, ConfigReceiver& configReceiver) =0;
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Message logs.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// MessageKind
//
//================================================================

enum class MessageKind {Info, Warning, Error};

//================================================================
//
// MessageRef
//
//================================================================

struct MessageRef
{
    const Char* text;
    MessageKind kind;
};

//================================================================
//
// MessageConsole
//
//================================================================

struct MessageConsole : public VirtualDestructor
{
    virtual void add(ArrayRef<const MessageRef> messages) =0;

    inline void add(const Char* text, MessageKind kind)
    {
        MessageRef msg{text, kind};
        add({&msg, 1});
    }
};

//================================================================
//
// StatusConsole
//
//================================================================

struct StatusConsole : public MessageConsole
{
    virtual void clear() =0;
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Actions.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

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
// There are special predefined IDs for mouse events
// and for config events.
//
//================================================================

using ActionId = uint32_t;

//----------------------------------------------------------------

namespace actionId {
constexpr ActionId MouseLeftDown = 0xFFFFFFFEu;
constexpr ActionId MouseLeftUp = 0xFFFFFFFDu;

constexpr ActionId MouseRightDown = 0xFFFFFFFCu;
constexpr ActionId MouseRightUp = 0xFFFFFFFBu;

constexpr ActionId WheelDown = 0xFFFFFFFAu;
constexpr ActionId WheelUp = 0xFFFFFFF9u;

constexpr ActionId SaveConfig = 0xFFFFFFF8u;
constexpr ActionId LoadConfig = 0xFFFFFFF7u;
constexpr ActionId EditConfig = 0xFFFFFFF6u;

constexpr ActionId ResetupActions = 0xFFFFFFF5u;

constexpr ActionId MouseMove = 0xFFFFFFF4u;
}  // namespace actionId

//================================================================
//
// ActionKey
//
//----------------------------------------------------------------
//
// The with the name of the action hotkey, for example, "Ctrl+W".
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

using ActionKey = StringPtr;

//================================================================
//
// ActionName
//
//----------------------------------------------------------------
//
// The string with the name of the action signal.
//
// The string can contain slashes for hierarchy, for example:
// "MyAlgorithm/MySection/MySignal" is displayed in menu as:
// MyAlgoritm -> MySection -> MySignal.
//
//================================================================

using ActionName = StringPtr;

//================================================================
//
// ActionParamsRef
//
//================================================================

struct ActionParamsRef
{
    // Action ID. There are special predefined IDs.
    // See ActionId definition for details.
    ActionId id;

    // Hotkey string. Can be empty.
    // See ActionKey type definition for details.
    ActionKey key;

    // Action name string. Can be empty.
    // Can contain separating slashes, see ActionName definition for details.
    // My Algo/My Sub Module/My Action
    ActionName name;

    // Description string. Can be empty.
    StringPtr description;
};

//================================================================
//
// ActionSetup
//
//================================================================

struct ActionSetup : public VirtualDestructor
{
    virtual void clear() =0;

    virtual void add(ArrayRef<const ActionParamsRef> actions) =0;
};

//================================================================
//
// MousePos
//
//================================================================

class MousePos
{

public:

    MousePos() = default;

    inline MousePos(ImagePoint pos) 
        : posX(pos.X), posY(pos.Y) {}

    inline bool valid() const 
        {return posX >= 0 && posY >= 0;}

    inline ImagePoint pos()
        {return ImagePoint{posX, posY};}

private:

    int16_t posX = -1;
    int16_t posY = -1;

};

//================================================================
//
// ActionRec
//
//================================================================

struct ActionRec
{
    // ID
    ActionId id;

    // Mouse position in the coordinate system of the main image,
    // expressed in pixels of the main image.
    MousePos mousePos;
};

//================================================================
//
// ActionReceiving
//
// Gets actions that happened from the previous action receiving.
//
// Actions are transferred to the client, so they are passed
// to the client only once.
//
//================================================================

struct ActionReceiver : public VirtualDestructor
{
    virtual void process(ArrayRef<const ActionRec> actions) =0;
};

//----------------------------------------------------------------

struct ActionReceiving : public VirtualDestructor
{
    virtual void getActions(ActionReceiver& receiver) =0;
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
// ImageProvider
//
//================================================================

struct ImageProvider : public VirtualDestructor
{
    // Saves to BGR32.
    virtual void saveBgr32(ImageRef<PixelRgb32> dst) =0;

    // Saves to BGR24. The destination is an uint8 image
    // with width 3 times more than the color image width.
    // The pitch is expressed in its own uint8 elements, as usual.
    virtual void saveBgr24(ImageRef<PixelMono> dst) =0;
};

//================================================================
//
// VideoOverlay
//
//================================================================

struct VideoOverlay : public VirtualDestructor
{
    //
    // Clears the video image.
    //

    virtual void clear() =0;

    //
    // Replaces the video image.
    //

    virtual void set(const ImagePoint& size, ImageProvider& imageProvider, StringPtr description) =0;
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Debug bridge.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// DebugBridge
//
//================================================================

struct DebugBridge : public VirtualDestructor
{
    virtual bool active() =0;

    // Flush all changes.
    virtual void commit() =0;

    virtual ConfigSupport* configSupport() =0;
    virtual ActionSetup* actionSetup() =0;
    virtual ActionReceiving* actionReceiving() =0;
    virtual MessageConsole* messageConsole() =0;
    virtual StatusConsole* statusConsole() =0;
    virtual VideoOverlay* videoOverlay() =0;
};

//================================================================
//
// DebugBridgeProvider
//
//================================================================

struct DebugBridgeProvider : public VirtualDestructor
{
    // If there are no active users, returns nullptr.
    virtual DebugBridge* debugBridge() =0;
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Debug bridge null.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ConfigSupportNull
//
//================================================================

class ConfigSupportNull : public ConfigSupport
{
    void saveConfig(ArrayRef<const Char> config) override
        {(void) config;}
    
    void loadConfig(ConfigReceiver&) override
        {}

    void editConfig(ArrayRef<const Char> config, ConfigReceiver&) override
        {(void) config;}
};

//================================================================
//
// ActionSetupNull
//
//================================================================

class ActionSetupNull : public ActionSetup
{
    void clear() override 
        {}

    void add(ArrayRef<const ActionParamsRef> actions) override 
        {(void) actions;}
};

//================================================================
//
// ActionReceivingNull
//
//================================================================

struct ActionReceivingNull : public ActionReceiving
{
    void getActions(ActionReceiver&) override
        {}
};

//================================================================
//
// StatusConsoleNull
//
//================================================================

class StatusConsoleNull : public StatusConsole
{
    void clear() override
        {}

    void add(ArrayRef<const MessageRef> messages) override
        {(void) messages;}
};

//================================================================
//
// VideoOverlayNull
//
//================================================================

class VideoOverlayNull : public VideoOverlay
{
    void set(const ImagePoint& size, ImageProvider&, StringPtr description) override
        {(void) size; (void) description;}

    void clear() override
        {}
};

//================================================================
//
// DebugBridgeNull
//
//================================================================

class DebugBridgeNull : public DebugBridge
{

public:

    bool active() override
        {return false;}

    void commit() override
        {}

    ConfigSupport* configSupport() override {return &configSupportNull;}
    ActionSetup* actionSetup() override {return &actionSetupNull;}
    ActionReceiving* actionReceiving() override {return &actionReceivingNull;}
    MessageConsole* messageConsole() override {return &messageConsoleNull;}
    StatusConsole* statusConsole() override {return &statusConsoleNull;}
    VideoOverlay* videoOverlay() override {return &videoOverlayNull;}

private:

    ConfigSupportNull configSupportNull;
    ActionSetupNull actionSetupNull;
    ActionReceivingNull actionReceivingNull;
    StatusConsoleNull messageConsoleNull;
    StatusConsoleNull statusConsoleNull;
    VideoOverlayNull videoOverlayNull;

};

//================================================================
//
// DebugBridgeProviderNull
//
//================================================================

class DebugBridgeProviderNull : public DebugBridgeProvider
{

public:

    DebugBridge* debugBridge() override {return &debugBridgeImpl;}

private:

    DebugBridgeNull debugBridgeImpl;

};

//----------------------------------------------------------------

inline auto debugBridgeNullTest() {return DebugBridgeProviderNull{};}

//----------------------------------------------------------------

} // namespace debugBridge
