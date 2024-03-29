﻿Общий принцип
=============

Бесят падения программы не на этапе иницализации, а когда она уже работает в
цикле, например, невозможность сохранить документ, настройки и тому подобное.

В этой графической оболочке есть цикл обработки событий и цикл работы алгоритма.
Такие циклы — естественное место для восстановления от ошибок.

Однако пользователь должен иметь возможность увидеть наличие ошибок.

Ошибки отрисовки
================

Они записываются в глобальный лог (пользователь может почитать, например, в
файле). Если он не смог отрисовать, то закрашивает буфер специальным шаблоном
("DRAW ERROR").

Ошибки и логи
=============

Ошибки выводятся в глобальный или локальный лог.

Реализация логов записывает свои собственные ошибки во внутреннее состояние
ошибки. Статус ошибки вызывает добавление фиксированного сообщения об ошибке
в конце лога при чтении строк из лога.

Операция clearLog очищает также и статус ошибки.

При добавлении любого сообщения, лог сначала пытается перекачать своё
состояние ошибки как обычное сообщение в конец лога и лишь в случае успеха
добавляет пользовательское сообщение.

При поглощении другого лога состояние ошибки также поглощается.
