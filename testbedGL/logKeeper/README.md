EN [RU](README.ru.md)

[channels]: ../channels/README.md

LogKeeper
=========

LogKeeper is designed to support asynchronous operations with the log file.

Upon initialization, LogKeeper stores the full path of the log file
and clears the log file if specified in the "Clear Log On Start" parameter.
Initialization occurs before starting the thread, that is, without any
asynchrony.

After starting the thread, LogKeeper operates
[the same way as other services][channels].

Its supported input update types:
* Termination request.
* Log update: buffer in the form of a "character array."
* Log editing request.

Upon receiving a log update, LogKeeper stores it in memory, and after
some time (or upon closure), appends it to the log file.
The delay time is determined by the "Commit Delay In Seconds" parameter.

Upon receiving a log editing request, LogKeeper flushes the buffer
to the log file, launches the specified text editor in the request,
and waits for its completion.
