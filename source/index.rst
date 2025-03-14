Marine Robotics Computer Vision
==================================

Открытая библиотека компьютерного зрения для морских робототехнических систем.

.. contents:: Оглавление
   :depth: 2
   :local:

Структура каталогов
--------------------

.. code-block:: bash

    .
    ├── cmake  # Утилиты для сборки
    │   ├── mrcv-config.cmake.in
    │   ├── silent_copy.cmake
    │   └── utils.cmake
    │
    ├── examples  # Примеры использования
    │   ├── mrcv-example
    │   │   ├── main.cpp
    │   │   └── CMakeLists.txt
    │   └── CMakeLists.txt
    │
    ├── include  # Публичные заголовки
    │   └── mrcv
    │       ├── export.h
    │       ├── mrcv-common.h
    │       └── mrcv.h
    │
    ├── python  # Версия библиотеки на Python
    │   ├── examples
    │   └── src
    │
    ├── src  # Исходники функций библиотеки
    │   ├── mrcv-augmentation.cpp
    │   ├── mrcv-calibration.cpp
    │   ├── ...
    │   └── mrcv.cpp
    │
    ├── tests  # Тесты
    │   ├── add_test.cpp
    │   └── CMakeLists.txt
    │
    ├── CMakeLists.txt
    ├── CMakePresets.json
    └── README.md

Датасет изображений для работы с библиотекой доступен по ссылке:

`code-ai-400393-image-dataset.7z <https://disk.yandex.ru/d/TxReQ9J6PAo9Nw>`_


Page 2 
---------

.. include:: Page2.rst
