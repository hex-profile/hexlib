EN [RU](README.ru.md)

LogBuffer
=========

Error Handling
--------------

The log implementation records its errors in an internal error state.

The error state causes the addition of a fixed error message at the end
of the log when reading lines from the log.

The `clearLog` operation also clears the error state.

When appending to the log, the log first tries to transfer its error state
as a regular message to the end of the log, and only in case of success, adds the data.

When absorbing another log, the error state is also absorbed.

After the absorption operation, the absorbed delta buffer is always empty and error-free.
If an exception occurs, the error is recorded in the main buffer's state.
The absorbed buffer is still cleared.

Last Modification Timestamp
---------------------------

The log stores the last modification timestamp (initially undefined).

This timestamp is updated when adding lines and when absorbing another log.
In an empty log, the timestamp can be undefined.

The timestamp can be requested from the external API and is also used as
an arbitrary error timestamp when displaying the internal error state.

Log Memory
----------

The log uses deque<>.

The log implementation is able to store only the last L messages (e.g., 2,000).
This is necessary because the program can spam the global console indefinitely.
The L limit can be disabled.

### Base Buffer

The log uses deque<>.

Managed to get by with these operations, in addition to the obvious ones:

* `cutToLimit`.
If not (size <= limit), removes excess elements from the head (no-throw).

* `removeFromEnd`.
Removes N elements from the end (no-throw).

* `removeFromBeginningAndAppendAtEnd`.
Removes K elements from the beginning and adds N elements at the end.
This is a simultaneous operation, which allows for better memory utilization.
The operation may throw an exception, but in case of an error, the buffer remains unchanged.
