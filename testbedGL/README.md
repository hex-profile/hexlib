EN [RU](README.ru.md)

[channels]: channels/README.md
[main-ui]: gui/README.md
[ext-shell]: testbedGL/README.md
[worker]: worker/README.md
[config-keeper]: configKeeper/README.md
[log-keeper]: logKeeper/README.md

Graphical Shell
===============

testbedGL is a graphical shell that implements a UI for a human user
and provides a programmatic API for an
[algorithm test module](docs/hexlib-test-module.en.md).

The graphical shell is implemented using OpenGL and works on Windows and Linux.
An NVIDIA GPU is required for operation.

* [User interface](docs/ui.en.md).
* [How to create a binary using the graphical shell](docs/linking.en.md).
* [How to run the UI inside Docker with hardware OpenGL
and access via VNC](docs/docker-opengl-vnc/how-to.en.md).

Software Documentation
======================

The graphical shell is an OpenGL application. It uses the portable
GLFW library to create a window and OpenGL context, as well as for handling
keyboard and mouse input-output.

The [external UI shell][ext-shell], tied to OpenGL and GLFW, is isolated in
a separate module, while the [main UI part][main-ui] is independent of the
windowing system and represents a class that takes a stream of keyboard
and mouse events as input, and outputs a complete picture of the window
contents, using the same tools as any image processing in hexlib.

[Error handling principle in the shell](docs/error-handling.en.md).

The graphical shell consists of several threads,
[interacting by a unified principle][channels]:

* GUI — the main thread, which starts from `main`. It contains
  the event handling and window rendering loop. This thread's cycle is short
  to maintain responsiveness, and it does not perform lengthy operations.

  In this thread, the [external shell][ext-shell] (`/testbedGL`) and
  [main UI part][main-ui] (`/gui`) operate.

* [WORKER][worker] is needed to offload lengthy operations from the GUI thread.
  It owns an instance of the algorithm module and calls it.
  WORKER processing cycles are initiated at its own discretion.
  Directory: `/worker`.

* [ConfigKeeper][config-keeper] is responsible for asynchronous operations
  with the config file, such as updating the config file and editing it.
  Directory: `/configKeeper`.

* [LogKeeper][log-keeper] is responsible for asynchronous log file updating
  and editing. Directory: `/logKeeper`.
