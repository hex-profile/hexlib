EN [RU](README.ru.md)

CfgTree
=======

The config tree is a data structure for storing textual values of
configuration variables in memory.

Node Structure
--------------

A tree node contains the following elements:

* **Node name** - stored as a string with a hash. The name hash is used to
  speed up node search by name and is a 32-bit unsigned integer.
* **Node data** - stored as strings, containing the variable value, comment,
  and block comment.

When searching for a node by name, the hash is compared first, and if it matches,
strings are compared.

Memory Allocation for Strings
-----------------------------

All strings are allocated using a greedy method:

- If the string size increases, memory is allocated.
- If the string size decreases, memory is not released.

Storing Node Children
----------------------

Child nodes are stored as a linked list. To minimize reallocations, two lists
are maintained:

* **List of child nodes** - contains active child nodes of the tree.
* **List of shadow nodes** - contains deleted nodes that have not been released
  and can be reused.

When allocating a new child node, the shadow nodes list is first checked for a
suitable node. If a node is found, it is moved to the main list.

Singly Linked Circular List
---------------------------

Child nodes are stored in a simple circular list. Each node points to the next,
and the last node points back to the first. If there are no elements, a null
pointer is stored. When there is at least one element, the list is always
circular. The list is stored as a pointer to its end, allowing for fast element
addition and easy access to the head by taking the last node's reference to the
next node.

Tree Operations
----------------

* Find a child node by name.
* Find or create a child node by name - locates an existing node or
  creates a new one if the name is not found.
* Process all child nodes with a specific handler.
* Standard delta buffer API operations.

Absorbing Update
----------------

All nodes of the absorbed tree are moved into the main tree.
If the corresponding node of the main tree already exists, its data is updated
with the data of the absorbed tree node.
This operation does not allocate memory and does not raise exceptions.

Generating Update for Modified Nodes
------------------------------------

In the data of each node, there is a flag indicating the presence of changes.

The usual data-setting operation does not update this flag, as it
requires additional computational effort.

However, there is a special data-setting function that checks
whether the data has actually changed and sets this flag.

The update generation operation builds a new tree containing only the changed
nodes of the given tree.
