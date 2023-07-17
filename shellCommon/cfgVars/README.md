EN [RU](README.ru.md)

Configuration Variables Support
================================

* /configFile: Old API for working with the configuration file.
  Maintained for compatibility purposes.

* [/cfgTree][cfg-tree]: Config tree: an in-memory data storage,
  containing text values of configuration variables.

* /cfgSerializeImpl: Writing variables to the config tree
  and reading variables from the config tree.

* /cfgOperations: High-level operations:
  - Writing the config tree to a file and reading it from a file.
  - Writing the config tree to a string and reading it from a string.
  - Writing variables to the config tree and reading them from the config tree.

[cfg-tree]: cfgTree/README.md
