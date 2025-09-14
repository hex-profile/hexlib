EN [RU](errorHandlingBoolRef.ru.md)

# Error Handling Mode Using a Reference to a Boolean Flag

This error handling method works by passing a reference to a boolean flag
among functions.

It contains controversial decisions, but leads to cleaner and shorter
application code, similar to what is achieved with the use of exceptions.

In fact, this mode is a fallback compilation option if exceptions cannot be used
for some reason.

## Difference from the Traditional hexlib Approach

This error handling method does not require enclosing each function call
in a `require` error check macro, which was especially cumbersome for large
calls:

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

## How It Works

Each function, in addition to the standard set of parameters (kit, callstack,
profiler), receives an additional reference to a boolean flag inside `stdPars`.

The flag is initially created with the value `true` (when creating root
"standard parameters") and remains `true` during the normal course of program
execution.

The flag is passed between functions by reference. If an error occurs
in a function, it sets the flag to `false` and does a `return`. The calling
function checks the flag and, if it is reset, also does a `return`.

## The `stdPass` Macro and Its Features

The `stdPass*` standard parameter passing macro in this error handling mode
becomes a "very bad macro (tm)". It:

* Closes the function call bracket;

* Performs a success flag check, and, if the flag is reset, makes a `return`
from the function;

* Ends with something like `do {...} while (0`, so the user code can close
the function call bracket, and `stdPass` looks just like the last function
parameter.

## Potential Problems and Workarounds

This approach eliminates boilerplate return checks but has side effects.
In particular, the `stdPass*` macro is no longer a single entity and splits
the function call into two statements.

As a result, if the function call is under a conditional or loop statement,
only the first part of the macro may be executed, and the check
will be performed after the statement.

At first glance, this seems to doom this method. However, it's not so bad.

In the case of an `if+else` construction, the compiler requires the use of curly
brackets, giving an error during the compilation stage.

In the case of a single `if`, such a place is indeed missed by the compiler,
while the flag check is always performed, regardless of the condition,
which does not disrupt the logic of work, but may lead to the execution of two
or three extra assembly commands.

To identify problematic places with loops, there is a Python script,
the function `check_issues_in_bool_ref_error_mode`.
