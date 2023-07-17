How to link testbedGL
=====================

The graphical shell is a static library,
so to obtain a binary, you need to link it to your executable target:

```
hexlibProjectTemplate(myTestGL EXECUTABLE . "myEngine;testbedGL" FALSE "")

if (WIN32)
    set_target_properties(myTestGL PROPERTIES WIN32_EXECUTABLE true)
endif()
```

You also need to create an entry point in some file:

```
#include "testbedGLEntryPoint.h"
```

Here, you need to create a factory class inheriting from `TestModuleFactory`:

```
class MyTestFactory : public TestModuleFactory
{
    const CharType* configName() const
        {return CT("mytest");}

    const CharType* displayName() const
        {return CT("My Test");}

    UniquePtr<TestModule> create() const
        {return makeUnique<MyModule>();}
};
```

`MyModule` is the test module itself — a class that implements the `TestModule` API.

All that's left is to create an entry point using the `TESTBED_GL_ENTRY_POINT` macro:

```
TESTBED_GL_ENTRY_POINT(MyTestFactory{})
```

This macro will generate `main` or `WinMain` depending on the OS.
