EN [RU](README.ru.md)

External Shell
==============

This directory contains the external shell, which includes the entry point (main).

From the entry point, the GUI thread starts, which runs:

- The external shell, which isolates everything that depends on the windowing
  system, OpenGL, and GLFW.
- GuiClass, which implements a pure UI, independent of OpenGL or the windowing
  system.

Contents:

* [PixelBuffer](#pixelbuffer):
  Image buffer shared between OpenGL and CUDA.
* [PixelBufferDrawing](#pixelbufferdrawing):
  Drawing `PixelBuffer` on screen using OpenGL.
* [WindowManager](#windowmanager):
  Class that isolates all GLFW usage within itself.
* [Initialization order](#initialization-order) of the external shell.

PixelBuffer
===========

`PixelBuffer` represents an image buffer shared between GRAPHICS (OpenGL) and
COMPUTE (e.g., CUDA).

States:

* `NONE`: Initial state, buffer is empty.
* `ALLOCATED`: Shared memory is allocated for the buffer, accessible for GRAPHICS.
* `COMPUTE-LOCKED`: Buffer is locked for COMPUTE, accessible for COMPUTE.

Usage:

* Calling `getGraphicsBuffer` is possible only in the ALLOCATED state.
* Calling `getComputeBuffer` is possible only in the COMPUTE-LOCKED state.
* To use the buffer in COMPUTE mode, it must be temporarily locked using
  the `lock` / `unlock` functions.

Extensions and platforms:

* Uses the OpenGL extension `glMakeNamedBufferResidentNV`.
* Supports CPU emulation mode (HEXLIB_PLATFORM == 0).

PixelBufferDrawing
==================

Drawing an instance of `PixelBuffer` on the screen using OpenGL.
Uses a special OpenGL shader to read pixels directly from GPU memory.
The shader is specific to NVIDIA.

Compilation and linking of shaders are done during initialization.

* Initialization and deinitialization: `reinit` and `deinit`.
* Drawing: `draw`.

WindowManager
=============

A class that isolates all GLFW dependencies within itself.

The window manager can:

- Create a window.

- Put an empty event in the event queue: `postEmptyEvent()`.
  This is needed to asynchronously wake up the GUI thread from another thread.

## Window functions

- Setting the OpenGL context: `setThreadDrawingContext`.
  Makes the OpenGL context of this window current for the calling thread.
  Apparently, it sets some global OpenGL TLS variable
  "current context".

- Buffer swapping: `swapBuffers`. This is the OpenGL double buffer switch,
  when the client has finished drawing and needs to switch the buffer on the screen.

- Receiving events: `getEvents`. Input: options "wait for events"
  and "timeout in milliseconds". Output: a set of pointers to callbacks
  `EventReceivers`, which it will call when receiving events
  inside the `getEvents` call.

Other window functions:
- Setting visibility: `setVisible`.
- Checking for closure: `shouldContinue`.
- Getting and setting location: `getWindowLocation` and
  `setWindowLocation`.
- Getting the image buffer size: `getImageSize`.

## Timeouts

When creating a window, the implementation creates an auxiliary thread
`TimeoutHelper`, which is necessary due to the lack of support for event
waiting with timeouts in GLFW. The purpose of this thread is to receive and
execute commands like "wake up the GUI thread after the specified number of
milliseconds". In the case of GLFW, to wake up, it puts a special empty event
in the event queue using the `glfwPostEmptyEvent` function.

## GLFW Callbacks

All GLFW usage, including its callbacks, happens in the same GUI thread.
There's no asynchrony here, except for the operating system adding events to
the GUI thread's event queue.

Unfortunately, GLFW uses global variables for its callbacks, and it's not
clearly defined in the documentation when these callbacks can be called,
which extends the required lifetime of event receivers indefinitely.

The window manager straightens this out by implementing an API where pointers
to event receivers are only passed during the execution of the get events
function.

A pointer to the actual event receiver is stored in the window fields. It's set
when entering get events and reset when exiting it.

GLFW callbacks are set in advance during initialization.
GLFW provides the called callback with a pointer to the window. The callback
code checks the window instance for a pointer to the actual receiver of this
event type, and if it's not null, calls it. If the pointer is null, the
callback does nothing but generates a debugger interrupt to track unwanted
behavior.

Another questionable GLFW callback is the error handler. In case of an error,
GLFW sometimes returns the error in the function value but often doesn't return
it anywhere and just calls a global callback.

The window manager's implementation of this callback stores the error message
in a global string. After calling a GLFW function, the calling code always
checks this string for errors.

Initialization Order
====================

There are the following dependencies:

* The GUI board owns a GPU memory image buffer, which is impossible without
  a GPU context. When creating the board, the buffer is empty, so no GPU
  context is required. However, GUI board destruction should occur before
  GPU context destruction.

* The GUI thread notifier can be created before creating the window, but only
  after initializing the window manager (GLFW), as it wakes the GUI thread
  using a GLFW function that puts an empty event in the event queue.

* OpenGL only becomes available after creating the window. Therefore, creating
  a shared CUDA/OpenGL buffer is only possible after creating the window.

Current initialization order:

* Creates all subsystems: GuiClass, Worker, ConfigKeeper, and
  LogKeeper. Their threads are not yet running.

* Initializes ConfigKeeper. Loads the config and updates the variables
  for all subsystems.

* Creates all boards: guiService, workerService, configService, and
  logService.

* Initializes and starts the LogKeeper thread. Starts the ConfigKeeper thread.

* Creates a GPU context and GPU streams for the GUI and WORKER threads.

* Creates the window manager (GLFW).

* Creates the GUI thread notifier and sets it for the GUI board.

* Initializes and starts the WORKER thread.

* Next, the window recreation loop runs. When creating a window,
  the OpenGL context is created. The window can be recreated without
  recreating the GPU context, despite the CUDA/OpenGL interop.

* After the window is created, OpenGL is available. Creates the window
  drawer PixelBufferDrawing, including compiling its shaders.

* Creates the window image buffer, shared between CUDA and OpenGL. The size of this
  buffer can change during the program's operation.
