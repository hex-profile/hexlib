Как подключить testbedGL
========================

Графическая оболочка представляет собой статическую библиотеку,
поэтому, чтобы получить бинарь, нужно её прилинковать к своему исполняемому таргету:

```
hexlibProjectTemplate(myTestGL EXECUTABLE . "myEngine;testbedGL" FALSE "")

if (WIN32)
    set_target_properties(myTestGL PROPERTIES WIN32_EXECUTABLE true)
endif()
```

Также нужно в каком-то файле создать точку входа:

```
#include "testbedGLEntryPoint.h"
```

Здесь нужно сделать класс фабрики, наследующий от `TestModuleFactory`:

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

`MyModule` это сам тестовый модуль — класс, который реализует API `TestModule`.

Остаётся создать entry point с помощью макроса `TESTBED_GL_ENTRY_POINT`:

```
TESTBED_GL_ENTRY_POINT(MyTestFactory{})
```

Этот макрос породит `main` или `WinMain` в зависимости от ОС.
