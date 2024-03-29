﻿[EN](README.md) RU

[channels]: ../channels/README.ru.md
[cfg-tree]: ../../shellCommon/cfgVars/cfgTree/README.ru.md
[log-buffer]: ../channels/buffers/logBuffer/README.ru.md
[overlay-buffer]: ../channels/buffers/overlayBuffer/README.ru.md

Содержание
==========

Каталог `/gui` содержит реализацию UI, независимую от системы оконного вывода.

* [Внешний API независимого UI](#внешний-api-независимого-ui)
  - [Сервис треда GUI](#сервис-треда-gui)
  - [GuiClass](#guiclass)
  - [EventSource](#eventsource)
  - [DrawReceiver](#drawreceiver)
* [Реализация UI](#реализация-ui)
  - [Интерфейс GuiModule](#интерфейс-guimodule)
  - [Реализация GuiClass](#реализация-guiclass)
  - [Реализация GuiModule](#реализация-guimodule)

Внешний API независимого UI
===========================

Сервис треда GUI
----------------

Сервис GUI работает [так же, как и остальные сервисы][channels].

Его поддерживаемые типы входных обновлений:
* Запрос завершения.
* Обновление набора сигналов модуля алгоритма.
* [Обновление][log-buffer] глобального лога.
* [Обновление][log-buffer] локального лога.
* [Обновление][overlay-buffer] главной отладочной картинки.
* [Обновление][cfg-tree] конфигурационных переменных.

GuiClass
--------

Этот класс реализует самый внешний API независимого UI.

Основная его функция это `processEvents`, которая

* Получает события из источника событий [EventSource](#eventsource) и
* Выдаёт сгенерированные картинки содержимого окна
в получатель [DrawReceiver](#drawreceiver).

Также эта функция получает другие API:

* API-шки для [обмена тредов](../channels/README.ru.md):
серверную для GUI, чтобы выполнять задания, и клиентские
для WORKER, ConfigKeeper и LogKeeper, чтобы ими пользоваться.
* Буфер обновления глобального лога, который совместно используется
GuiClass и [внешней оболочкой](../testbedGL/README.ru.md), работающими
в одном треде.
* API общей сериализации GuiClass и [внешней оболочки](../testbedGL/README.ru.md).
Этим занимается GuiClass, чтобы перенести больше кода в оконно-независимую часть.
* API "попросить завершение работы".

EventSource
-----------

`EventSource` это одна функция "получить события" с опцией
"подождать событий" и таймаутом. Эта функция выводит события
в набор получателей событий `EventReceivers`.

`EventReceivers` — набор специфических API получателей для
событий мыши, клавиатуры и так далее, каждый из которых представляет
собой одну функцию "обработать событие данного типа".

Один из получателей событий это `RefreshReceiver`, который обрабатывает
запрос перерисовки окна. Он может вызываться многократно изнутри одного
вызова "получить события", например, GLFW отправляет его много раз при
перетаскивании размера окна мышкой.

DrawReceiver
------------

`DrawReceiver` состоит из функции "получить изображение", которая получает
изображение в виде поставщика изображения `Drawer`.

`Drawer` содержит функцию "нарисовать", которая записывает изображение
в указанную картинку GPU.

Реализация UI
=============

Фактическим рисованием содержимого окна занимается класс `GuiModule`, который
является GPU-модулем hexlib и пользуется расширенными инструментами GPU.

`GuiClass` содержит внутри себя экземпляр `GuiModule`, а также экземпляр
`MinimalShell` и пул памяти `GuiModule`, необходимые для того, чтобы
предоставить `GuiModule` инструменты GPU-модуля, такие как
быстрые аллокаторы и другие.

Интерфейс GuiModule
-------------------

По интерфейсу `GuiModule` представляет собой обычный GPU-модуль с функциями
конфигурирования `serialize`, аллокации `reallocValid` / `realloc`
и главной функцией обработки.

Главная функция обработки это `draw`, которая:
* Получает на вход текущие буферы (главной отладочной картинки,
глобального и локального лога)
* И рисует содержимое окна в указанную картинку GPU.

GuiModule получает некоторые события, для чего выведены соответствующие
функции получателей. События могут изменять его внутреннюю конфигурацию,
например, пользователь может изменять мышкой размер локальной консоли
или скроллить главную отладочную картинку.

UI может находиться в состоянии активной анимации. Например,
в глобальную консоль были выведены сообщения, которые ещё не исчезли.

Со стороны GuiModule это выглядит так:

* Есть функция `getWakeMoment`, которая возвращает момент времени,
когда произойдёт следующее изменение UI и нужна перерисовка.
Если активной анимации нет, функция возвращает NONE.
* Также есть функция `checkWake`, которая обновляет внутренний
момент желаемого пробуждения.

Сейчас `checkWake` работает консервативно: если в глобальном логе могут быть
видимые сообщения, а момент пробуждения не установлен, функция устанавливает
его на текущий момент времени. Более точное обновление момента пробуждения
происходит в процессе отрисовки, с использованием полной информации
отображаемых буферов.

Реализация GuiClass
-------------------

Его реализация более-менее простая, но есть некоторые особенности:

* В точке ожидания событий он решает ждать / не ждать и с каким таймаутом на
  основе момента пробуждения от GuiModule, как описано выше. Если анимация не
  завершена, используется ожидание с таймаутом.

* При каждой перерисовке окна сначала он получает и обрабатывает все входные
  обновления. В случае с GLFW, внутри одного вызова "получить события" может
  происходить много всего, например, когда пользователь тащит размер окна
  мышкой, GLFW не выпускает из вызова "получить события", пока пользователь не
  отпустит окно, при этом вызывая много раз колбек перерисовки окна. Я решил,
  что пусть он тогда уж обрабатывает обновления от WORKER и других потоков и
  обновляет содержимое окна, тем более, что проверка обновлений быстрая.

* Для буфера главной картинки на GPU используется нестандартное получение
  обновления, чтобы уменьшить потребление памяти и держать в памяти только один
  буфер вместо двух, для обновления и для текущего состояния. Он принимает
  обновление сразу в текущий буфер, но немного изменяет его API,
  чтобы текущий буфер не менялся, когда обновление пустое.

* При приёме обновлений глобальной консоли, новым сообщениям присваивается
  текущий момент времени вместо времени их печати, поскольку таймаут отображения
  сообщений должен отсчитываться от момента их появления на экране.

Реализация GuiModule
--------------------

Для отрисовки текста используется класс `GpuConsoleDrawer`, который немного
выходит за рамки GPU-модуля тем, что выделяет евенты GPU при реаллокации.

В остальном всё стандартно.
