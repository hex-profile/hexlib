EN [RU](README.ru.md)

[channels]: ../channels/README.md
[cfg-tree]: ../../shellCommon/cfgVars/cfgTree/README.md

ConfigKeeper
============

ConfigKeeper is designed to support asynchronous operations with the
configuration file.

It owns an in-memory storage of the textual representation of the
configuration variables, which reflects the current representation
of the variables of all subsystems. The storage is a buffer of type
[CfgTree][cfg-tree].

During initialization, ConfigKeeper reads the initial configuration
file into its storage and updates the actual variables of all
systems using the provided common serialization API. Initialization
occurs before the thread starts, i.e., without any asynchrony.

After starting its thread, ConfigKeeper operates
[the same way as other services][channels].

The supported types of incoming updates it handles are:
* Termination request.
* Configuration variables update: a buffer of type [CfgTree][cfg-tree].
* Configuration edit request.

Upon receiving a configuration variables update, ConfigKeeper updates
its in-memory storage, and after some time (or upon closing),
writes the storage to the configuration file.
The delay time is determined by the "Commit Delay In Seconds" parameter.

Upon receiving a configuration edit request, ConfigKeeper saves the
storage to the configuration file, launches the specified text editor
in the request, waits for it to complete, and then reads the
configuration file back into its storage.

Then, ConfigKeeper generates a [CfgTree][cfg-tree] buffer containing
only the human-modified configuration variables. To do this, before
launching the editor, it clears the "modified" flag of all storage
variables, and when reading the storage from the file, it checks for
data matches and sets this flag.

After generating the buffer with the configuration variables update,
ConfigKeeper applies it to its own variables and then sends it to the
GUI service.

General scheme of working with configuration variables
======================================================

All systems use the ConfigKeeper service following the same principle.

At the end of the current cycle, the system serializes itself, checking
for changes in its variables, and if there are any, it generates a buffer
of type [CfgTree][cfg-tree] containing the textual representation of the
modified variables. After that, it sends the buffer to the ConfigKeeper
service, which updates the common storage, and after some time (or upon
closing), writes the common storage to the configuration file.

In case of config editing, ConfigKeeper sends a buffer with the
variables update to the GUI thread, which applies it to its variables
and sends it to the other threads.
