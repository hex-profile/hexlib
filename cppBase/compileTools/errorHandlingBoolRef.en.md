EN [RU](errorHandlingBoolRef.ru.md)

# Error Handling Mode Using a Boolean Flag Reference

This error handling method works by passing a reference to a boolean flag
between functions.

It contains controversial decisions, but leads to cleaner and shorter
application code, similar to that achieved with the use of exceptions.

## Difference from Traditional Hexbase Approach

This method of error handling does not require enclosing each function call
in the error checking macro `require`, which was particularly inconvenient
for large calls:

```
require
(
    kit.gpuImageConsole.addMatrixEx
    (
        rawImage,
        float32(typeMin<MonoPixel>()),
        float32(typeMax<MonoPixel>()),
        point(1.f), INTERP_NONE, rawImageSize, BORDER_ZERO,
        paramMsg(STR("View %, Raw Input"), args.displayedView),
        stdPass
    )
);
```

## Working with the Boolean Flag

How it works: each function, in addition to the standard set of parameters (kit,
callstack, profiler), receives an additional reference to a boolean flag within
`stdPars`.

The flag is initially created with the value `true` (when creating root standard
parameters) and remains `true` during normal program execution.

The flag is passed between functions by reference. If an error occurs in a
function, it sets the flag to `false` and returns. The calling function checks
the flag and, if it is reset, also returns.

## `stdPass` Macro and its Perks

In this error handling mode, the standard parameter passing macro `stdPass*`
becomes a "very bad macro (tm)". It:

* Closes the function call bracket;

* Performs a success flag check, and, if the flag is reset, returns from the
function;

* Ends with something like `do {...} while (0`, so the user code can close the
function call bracket and `stdPass` just looks like the last function parameter.

## Potential Problems and Workarounds

This approach eliminates boilerplate return checks but has side effects.
In particular, the macro `stdPass*` is no longer a single entity and divides
the function call into two statements.

As a result, if a function call is placed under the `if` or `while` operator,
only the first part of the macro might be executed, and the check
will be performed after the operator.

At first glance, it seems that this rules out the given method. However,
everything is not so bad: there are no such loops in the code base, and there
is a Python script to identify potential problematic areas.

In the case of the `if+else` construction, the compiler requires the use
of curly braces, issuing an error already at the compilation stage.

In the case of a single `if`, such a place is indeed skipped by the compiler,
while the flag check is always performed, regardless of the condition, which
does not violate the logic of operation, but can lead to the execution
of two or three extra assembly instructions.
