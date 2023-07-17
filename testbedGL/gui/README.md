EN [RU](README.ru.md)

[channels]: ../channels/README.md
[cfg-tree]: ../../shellCommon/cfgVars/cfgTree/README.md
[log-buffer]: ../channels/buffers/logBuffer/README.md
[overlay-buffer]: ../channels/buffers/overlayBuffer/README.md

Content
=======

The `/gui` directory contains the implementation of the UI, independent of the
window output system.

* [External API of the independent UI](#external-api-of-the-independent-ui)
  - [GUI Thread Service](#gui-thread-service)
  - [GuiClass](#guiclass)
  - [EventSource](#eventsource)
  - [DrawReceiver](#drawreceiver)
* [UI Implementation](#ui-implementation)
  - [GuiModule Interface](#guimodule-interface)
  - [GuiClass Implementation](#guiclass-implementation)
  - [GuiModule Implementation](#guimodule-implementation)

External API of the independent UI
==================================

GUI Thread Service
------------------

The GUI service operates [the same way as other services][channels].

Its supported input update types:
* Termination request.
* Algorithm module signal set update.
* [Update][log-buffer] of the global log.
* [Update][log-buffer] of the local log.
* [Update][overlay-buffer] of the main debug image.
* [Update][cfg-tree] of configuration variables.

GuiClass
--------

This class implements the outermost API of the independent UI.

Its main function is `processEvents`, which:

* Receives events from the event source [EventSource](#eventsource) and
* Outputs the generated window content images to the
  [DrawReceiver](#drawreceiver).

This function also receives other APIs:

* APIs for [thread exchange](../channels/README.md):
  server-side for GUI to execute tasks, and client-side for
  WORKER, ConfigKeeper, and LogKeeper to use.
* Global log update buffer, which is jointly used by
  GuiClass and the [external shell](../testbedGL/README.md), running
  in the same thread.
* Common serialization API for GuiClass and [external shell](../testbedGL/README.md).
  GuiClass handles this to move more code to the window-independent part.
* "Request shutdown" API.

EventSource
-----------

`EventSource` is a single "get events" function with the option
to "wait for events" and a timeout. This function outputs events
to a set of event receivers `EventReceivers`.

`EventReceivers` is a set of specific receiver APIs for
mouse, keyboard events, and so on, each representing
a single "process event of this type" function.

One of the event receivers is `RefreshReceiver`, which handles
window redraw requests. It can be called multiple times within one
"get events" call, for example, GLFW sends it many times when
dragging the window size with the mouse.

DrawReceiver
------------

`DrawReceiver` consists of the "receive image" function, which receives
an image in the form of an image provider `Drawer`.

`Drawer` contains a "draw" function, which writes the image
to the specified GPU image.

UI Implementation
=================

The actual drawing of the window content is handled by the `GuiModule` class, which
is a hexlib GPU module and uses extended GPU tools.

`GuiClass` contains an instance of `GuiModule`, as well as an instance of
`MinimalShell` and a memory pool for `GuiModule`, which are necessary to
provide `GuiModule` with GPU module tools, such as
fast allocators and others.

GuiModule Interface
-------------------

`GuiModule` is a typical GPU module with configuration functions
`serialize`, allocation functions `reallocValid` / `realloc`, and the main
processing function.

The main processing function is `draw`, which:
* Takes the current buffers (main debug image, global and local log) as input
* And draws the window content into the specified GPU image.

GuiModule receives certain events, for which corresponding receiver functions
are provided. Events can change its internal configuration, for example, the
user can change the size of the local console with the mouse or scroll the main
debug image.

The UI can be in a state of active animation. For example, messages have been
displayed in the global console that haven't disappeared yet.

From the GuiModule's perspective, this looks like:

* There is a function `getWakeMoment`, which returns the moment in time when
  the next UI change will occur and a redraw is needed. If there is no active
  animation, the function returns NONE.
* There is also a function `checkWake`, which updates the internal moment of
  the desired awakening.

Currently, `checkWake` works conservatively: if there might be visible messages
in the global log, and the wake moment is not set, the function sets it to the
current moment in time. More precise updating of the wake moment occurs during
the drawing process, using complete information from the displayed buffers.

GuiClass Implementation
-----------------------

Its implementation is relatively straightforward, but there are some
specifics:

* At the event waiting point, it decides whether to wait / not wait and with
  what timeout based on the wake moment from GuiModule, as described above.
  If the animation is not complete, wait with timeout is used.

* During each window redraw, it first gets and processes all incoming updates.
  With GLFW, a lot can happen within a single call to "get events," for example,
  when the user drags the window size with the mouse, GLFW does not release the
  "get events" call until the user releases the window, while calling the window
  redraw callback multiple times. I decided that it should process updates from
  the WORKER and other threads and update the window content, especially since
  checking for updates is fast.

* For the main image buffer on the GPU, a non-standard update retrieval is used
  to reduce memory consumption and hold only one buffer in memory instead of two,
  for updating and for the current state. It takes the update directly into the
  current buffer but slightly modifies its API so that the current buffer does
  not change when the update is empty.

* When receiving global console updates, new messages are assigned the current
  moment in time instead of their print time, as the message display timeout
  should be counted from the moment they appear on the screen.

GuiModule Implementation
------------------------

For text rendering, the `GpuConsoleDrawer` class is used, which slightly goes
beyond the scope of a GPU module in that it allocates GPU events during
reallocation.

Everything else is standard.
