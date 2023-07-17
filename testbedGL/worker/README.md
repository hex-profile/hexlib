EN [RU](README.ru.md)

[channels]: ../channels/README.md
[cfg-tree]: ../../shellCommon/cfgVars/cfgTree/README.md
[log-buffer]: ../channels/buffers/logBuffer/README.md
[overlay-buffer]: ../channels/buffers/overlayBuffer/README.md

WORKER
======

The WORKER thread operates independently from the GUI thread and executes
algorithm module processing cycles, which can be lengthy.

WORKER Service
--------------

The WORKER service operates [similarly to other services][channels].

It accepts the following types of updates:

* Termination request.

* Receiving algorithm module signals of type `ActionBuffer`,
  represented as an array of IDs for occurred signals.

* Display settings update `DisplaySettingsBuffer`.
  Currently, it only contains the desired debug image size,
  which best matches the window size.

* Mouse pointer update `MousePointerBuffer`. It contains the current position
  of the mouse inside the debug image, as well as flags for pressing and
  releasing the mouse buttons.

* [Update][cfg-tree] of configuration variables,
  which may occur when a person edits the configuration file.

Using Other Services
---------------------

WORKER can send [updates][cfg-tree] of its configuration variables to the
ConfigKeeper service.

Each processing cycle, WORKER sends updates to the GUI service:

* Algorithm module signal set update of type `ActionSetBuffer`.
* [Update][log-buffer] of the global log.
* [Update][log-buffer] of the local log.
* [Update][overlay-buffer] of the main debug image.

Currently, WORKER updates the signal set once during initialization,
but may change it during processing cycles in the future.

WORKER Implementation
---------------------

WORKER invokes the test algorithm module, which requires full support of
GPU-module tools. WORKER implements this support using `MinimalShell`.
Therefore, WORKER owns an instance of the algorithm module, an instance
of `MinimalShell`, and a memory pool for the algorithm module.

`Worker` class:

* The `init` function is executed before the thread starts. In addition to
  creating `MinimalShell`, it updates the algorithm module's signal set.

* The `run` function is the thread's body. It receives APIs:
  the server API for its service to receive updates, and
  client APIs for the GUI and ConfigUpdater services.
  The thread also receives an already created GPU context and stream.
