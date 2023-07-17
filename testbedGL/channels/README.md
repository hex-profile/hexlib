EN [RU](README.ru.md)

Table of Contents
=================

* [Threads Interaction](#threads-interaction)
* [Updates and the Delta Buffer Concept](#updates-and-the-delta-buffer-concept)
* [Delta Buffer API](#delta-buffer-api)
* [Board](#board)
* [Avoiding Reallocation During Exchange](#avoiding-reallocation-during-exchange)
* [Threads and GPU Usage](#threads-and-gpu-usage)

Threads Interaction
===================

All threads interact following the same scheme.

Each thread works as a service that continuously waits, receives, and processes
updates.

The scheme consists of multiple update producers and only one update consumer –
the thread itself.

All threads are equal and can serve each other.

For example, when a user requests to edit a config, the GUI thread sends an edit
task to the ConfigKeeper thread. ConfigKeeper starts the editor and waits for it
to complete, then sends a config update to the GUI thread, which updates its
variables and forwards the config update to the WORKER thread. Both ConfigKeeper
and WORKER can print text messages that are updates for the GUI thread.

Updates and the Delta Buffer Concept
====================================

Each thread has several types of received updates. For example, the GUI thread
accepts updates for the global text console, local text console, images, and so
on. A request to terminate the thread is also an update type.

For each update type, there is a class called a delta buffer or a changes buffer.
A delta buffer contains changes that can be applied to another similar buffer.

For instance, with a "text console" buffer, a client may want to print several
messages. It sends a delta buffer containing these messages to the server, which
receives them and adds them to its delta buffer, storing the console's current
state.

An update does not always mean addition. For example, a client may want to clear
the console and then print messages. It sends a delta buffer containing a clear
request and messages. When the server receives and adds this buffer to its
current buffer, the current buffer is cleared before adding messages, as if the
client were directly updating the current buffer.

Delta buffer classes are located in the `/channels/buffers` directory.

Delta Buffer API
================

* `hasUpdates`: Checks for updates. For example, if the text console was
  cleared by the client, the buffer is NOT empty, as it contains the "clear
  console" update.
* `reset`: Clears the buffer. Does not free memory.
* `absorb`: Adds changes from another buffer while clearing the other buffer.
  Does not free memory of the other buffer.
* `moveFrom`: Replaces the contents with the contents of another buffer and
  clears the other buffer. Does not free memory of the buffers.
* `clearMemory`: Clears the buffer and frees memory.

Board
=====

Exchange with each of the services is carried out through a so-called board.

The board contains a mutex and shared memory protected by this mutex: one buffer
for each type of buffer of a given service.

The board provides two APIs, one for the client and one for the server.

For the client, the API always looks like "add buffer." The client's buffer is
added to the shared buffer. Often, the shared buffer is empty because the server
has already processed everything and is waiting — in this case, the client's
buffer quickly replaces the shared buffer.

For the server, the API always looks like "take buffer." The server takes the
entire buffer and empties the shared buffer.

The client can add one type of buffer or all buffers at once. The server usually
takes all buffers at once.

The server's "take buffer" function waits for a non-empty shared buffer to
appear. In some services, a wait flag and a timeout in milliseconds can be
specified.

When the client adds a buffer, the board wakes up the server. For regular threads,
this is done through a condition_variable, and for the GUI thread, a callback
notifier is passed. In the case of GLFW, it puts a special empty event in the
event queue.

Boards are located in the `/channels` directory.

Avoiding Reallocation During Exchange
=====================================

To avoid constant buffer reallocations, a buffer exchange scheme is used. All
buffers involved in constant exchange exist permanently and are not removed,
including the client and server's own buffers, as well as shared buffers.

When a client adds its own buffer, it passes the buffer by reference. After
adding to the shared buffer, the client's buffer is emptied, but its memory is
not released.

This works the same way when the server takes the shared buffer into its own
buffer. The shared buffer is emptied, but its memory is not released. This is
similar to move semantics but with guaranteed emptying of the source buffer.

Mainly, buffer transfers are fast exchange and emptying operations without memory
release. The exception is when the server does not have time to process tasks,
then the client's buffer is indeed added to a non-empty shared buffer. But even
in this case, the buffer elements themselves are moved.

Threads and GPU Usage
=====================

In the graphical shell, only two threads use the GPU: GUI and WORKER.

Both threads use the same shared GPU context, but each thread has its own
GPU stream (command queue).

The outer shell creates the GPU context and both GPU streams during
initialization.

## GPU Module Tools Support

Both the WORKER thread and the GUI thread provide full support for the hexlib
GPU module using `MinimalShell`:

* The WORKER thread calls the target test module, thus requiring a full set
  of GPU module tools.

* The GUI thread performs GPU-based drawing of the graphical interface, which
  could have been done with simpler means, but I decided to provide full GPU
  module support, as good visualization and profiling tools may be needed for
  the graphical interface development.

## Delta Buffer for GPU Image Exchange

The delta buffer is constantly passed between threads, so the regular allocators
of hexlib are not suitable, and direct memory allocation with basic GPU
functions is used instead.

These functions are slow and cause full synchronization of everything on the
GPU. Therefore, the buffer operates on a greedy principle: when the image size
changes, memory is reallocated only if the previously allocated amount is
insufficient.

Access to image data works as follows: the WORKER thread writes image data using
a GPU kernel, then the buffer ownership is transferred to the GUI thread, which
reads the image one or more times during redraw operations, also using a GPU
kernel.

The GPU works asynchronously with the CPU, so synchronization is an issue. For
this purpose, each buffer contains two GPU events: one indicates the completion
of writing to the buffer, and the other — the completion of reading from the
buffer.

Each of the threads, before accessing the buffer memory, inserts a command to
wait for the event of the completion of the other's operation in its GPU stream,
and after working with the buffer memory, records the event of the completion
of its own operation.

This way, a smooth connection of the two GPU streams is achieved, without
inefficient CPU synchronizations. The only exception is increasing the image
size, which causes GPU memory reallocation: in this case, allocation functions
completely synchronize the entire GPU context.

NVIDIA suggests thinking of an event as an object containing a set of work
recorded at a certain moment in a specific GPU stream. The event wait operation
waits until all this work is done. After creating an event, this set is empty,
so waiting for such an event ends immediately. If you record in the same event
again, the new set of work replaces the old one. For example, the GUI thread
can read image data multiple times during redrawing, repeatedly overwriting the
reading completion event.
