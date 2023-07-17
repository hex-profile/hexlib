General Principle
=================

It's annoying when a program crashes not during initialization, but when it's
already running in a loop, such as being unable to save a document, settings,
and the like.

In this graphical interface, there is an event processing loop and an algorithm
work loop. Such loops are a natural place for error recovery.

However, the user should be able to see the presence of errors.

Rendering Errors
================

They are logged in the global log (the user can read it, for example, in a
file). If it fails to render, it fills the buffer with a special pattern
("DRAW ERROR").

Errors and Logs
===============

Errors are output to either the global or local log.

The log implementation writes its own errors to an internal error state.
The error status causes a fixed error message to be added at the end of the log
when reading lines from the log.

The clearLog operation also clears the error status.

When adding any message, the log first tries to transfer its error state as a
regular message to the end of the log, and only in case of success, the log
adds the user message.

When absorbing another log, the error state is also absorbed.
