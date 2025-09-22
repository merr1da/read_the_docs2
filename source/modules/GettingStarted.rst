Подготовка к работе
===================

Установка и настройка зависимостей
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Для работы с библиотекой используется версия *OpenCV 4.10*, которая должна быть предварительно собрана и установлена на исполняемой платформе с операционными системами Windows или Linux. Для поддержки CUDA сборка OpenCV должна проводиться с установленными *CUDA 12.4* и *cuDNN 9.3*. Для сборки должна быть использована версия *CMake* не ниже 3.14. Также для работы с библиотекой требуется *libtorch 2.4.0*. При этом сборка с CUDA или без определяется установочным пакетом от разработчиков *libtorch*.

Установка и первоначальная настройка библиотеки
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Для операционной системы Windows без поддержки CUDA
---------------------------------------------------

Необходимо выполнить следующий ряд действий:

- Установить необходимые компоненты:
  
  1. CMake GUI 3.30.0-rc4 или новее с официального сайта (доступен по ссылке https://cmake.org/download/)
  2. Git Bash (доступен по ссылке https://gitforwindows.org/)
  3. Visual Studio 2022 Community Edition с компонентом **Desktop development with C++** (доступен по ссылке https://visualstudio.microsoft.com/)  
  4. yaml-cpp при помощи консольных инструкций:
  
     .. code-block:: console
     
        cd C:\
        git clone https://github.com/jbeder/yaml-cpp.git
        cd yaml-cpp
        mkdir build
        cd build
        cmake .. -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_INSTALL=ON
        cmake --build . --config Release
        cmake --install . --prefix "C:\yaml-cpp"
  
  5. LibTorch 2.4.0 без поддержки CUDA  
     (https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.0%2Bcpu.zip). После скачивания архива распакуйте его содержимое в папку ``C:\libtorch-12.4``, переименуйте папку ``C:\libtorch-12.4\libtorch`` в ``C:\libtorch-12.4\Release``.

- скачайте готовую сборку OpenCV 4.10.0 с официального сайта (https://opencv.org/releases/). Распакуйте содержимое папки build архива и скопируйте в ``C:\opencv-4.10.0-build\install``

Для установки библиотеки MRCV необходимо выполнить следующие действия:

- Клонировать актуальную версию проекта, используя команды:

  .. code-block:: console
  
     cd ~
     git clone --branch main https://github.com/valabsoft/code-ai-400393.git

- Установить зависимости: libtorch, OpenCV, yaml-cpp (отображены рекомендованные пути для ОС Windows):

  .. code-block:: text
  
     C:\
     ├───libtorch-12.4
     ├───opencv-4.10.0-build
     └───yaml-cpp

- Для корректной работы библиотеки под управлением операционной системы Windows необходимо прописать системные пути в переменные окружения:

  .. code-block:: text
  
      C:\opencv-4.10.0-build\install\x64\vc17\bin\
      C:\libtorch-12.4\Release\bin\

- Настроить сборку, отключив поддержку CUDA. Для этого отредактировать CMakeLists.txt проекта

  .. code-block:: text

      option(USE_CUDA "Use CUDA Build" OFF)

- Запустить проект от имени администратора и открыть mrcv как локальную папку (File → Open → Folder)
- Настройте CMakeLists.txt и CMakeLists.txt в каждом примере в папке examples, убедившись, что пути к libtorch, opencv и yaml-cpp указаны корректно (см. пример ниже)

  .. code-block:: text

      if(USE_CUDA)
         # Добавляем макрос MRCV_CUDA_ENABLED
         target_compile_definitions(mrcv PRIVATE MRCV_CUDA_ENABLED)
         if(WIN32)
            set(CMAKE_PREFIX_PATH "C:/libtorch-12.4-cuda/Release;C:/yaml-cpp")
            set(OpenCV_DIR "C:/opencv-4.10.0-build-cuda/install")
            set(Torch_DIR "C:/libtorch-12.4-cuda/Release/share/cmake/Torch")
         
            set(OpenCV_INCLUDE_DIRS "C:/opencv-4.10.0-build-cuda/install/include")
            set(OpenCV_LIB_DIR "C:/opencv-4.10.0-build-cuda/install/x64/vc17/lib")
            
            set(Torch_INCLUDE_DIRS "C:/libtorch-12.4-cuda/Release/include")
            set(Torch_LIB_DIR "C:/libtorch-12.4-cuda/Release/lib")
         else()
            set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
         
            set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
            set(OpenCV_LIB_DIR "/usr/local/lib")
            
            set(Torch_INCLUDE_DIRS "/usr/local/include/torch")
            set(Torch_LIB_DIR "/usr/local/lib")
            
         endif()
      else() 
         if(WIN32)
            set(CMAKE_PREFIX_PATH "C:/libtorch-12.4/Release;C:/yaml-cpp")
            set(OpenCV_DIR "C:/opencv-4.10.0-build/install")
            set(Torch_DIR "C:/libtorch-12.4/Release/share/cmake/Torch")

            set(OpenCV_INCLUDE_DIRS "C:/opencv-4.10.0-build/install/include")
            set(OpenCV_LIB_DIR "C:/opencv-4.10.0-build/install/x64/vc17/lib")

            set(Torch_INCLUDE_DIRS "C:/libtorch-12.4/Release/include")
            set(Torch_LIB_DIR "C:/libtorch-12.4/Release/lib")
         else()
            set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
            
            set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
            set(OpenCV_LIB_DIR "/usr/local/lib") 
            
            set(Torch_INCLUDE_DIRS "/usr/local/include/torch")
            set(Torch_LIB_DIR "/usr/local/lib")
         endif()
      endif()

-	Выбрать в Visual Studio конфигурацию сборки dev-win;
-	В разделе «Сборка» выбрать «Собрать проект»;
-	После успешной сборки в разделе «Сборка» выбрать «Установить mrcv».

Для операционной системы Windows с поддержкой CUDA
--------------------------------------------------

Необходимо выполнить следующий ряд действий:

- Установить необходимые компоненты:
  
  1. CMake GUI 3.30.0-rc4 или новее с официального сайта (доступен по ссылке https://cmake.org/download/)
  2. Git Bash (доступен по ссылке https://gitforwindows.org/)
  3. Visual Studio 2022 Community Edition с компонентом **Desktop development with C++** (доступен по ссылке https://visualstudio.microsoft.com/)  
  4. yaml-cpp при помощи консольных инструкций:
  
     .. code-block:: console
     
        cd C:\
        git clone https://github.com/jbeder/yaml-cpp.git
        cd yaml-cpp
        mkdir build
        cd build
        cmake .. -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_INSTALL=ON
        cmake --build . --config Release
        cmake --install . --prefix "C:\yaml-cpp"
  
  5. LibTorch 2.4.0 с поддержкой CUDA 12.4  
     (доступна по ссылке:  
     https://download.pytorch.org/libtorch/cu124/libtorch-win-shared-with-deps-2.4.0%2Bcu124.zip). После скачивания архива распакуйте его содержимое в папку ``C:\libtorch-12.4-cuda``, переименуйте папку ``C:\libtorch-12.4\libtorch`` в ``C:\libtorch-12.4\Release``.
  
  6. CUDA Toolkit 12.4  
     (доступ по ссылке https://developer.nvidia.com/cuda-12-4-0-download-archive). Установите CUDA Toolkit в папку по умолчанию: ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4``.
  
  7. cuDNN 9.3.0  
     (доступ по ссылке https://developer.nvidia.com/cudnn-9-3-0-download-archive?target_os=Windows&target_arch=x86_64)  
     Необходимо скачать архив с тремя папками: bin, include, lib. Их нужно скопировать в папку установки CUDA Toolkit 12.4.
     Путь по умолчанию:  
     ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4``.

- Клонировать репозитории с версией OpenCV 4.10:

  .. code-block:: console
  
     git clone https://github.com/opencv/opencv.git -b "4.10.0"
     git clone https://github.com/opencv/opencv_contrib.git -b "4.10.0"

- Создайте папку ``C:\opencv-4.10.0-build-cuda\install`` и перенесите в нее скачанные opencv и opencv_contrib.
- Создать пустые директории для сборки библиотеки ``C:\opencv-4.10.0-build-cuda\install\opencv\build`` и ``C:\opencv-4.10.0-build-cuda\install\bin``.
- Запустить CMake-GUI.
- Выбрать компилятор Visual Studio 17 2022.

  .. image:: /_static/compiler_selection.jpg
     :alt: Окно выбора компилятора

- В полях **Where is the source code** и **Where is the build binaries** указать пути к папке с исходниками OpenCV и созданной папке build.  
  Например, папка ``install`` содержит собранные материалы библиотеки OpenCV и экстра модулей.

  .. image:: /_static/directory_selection.jpg
     :alt: Окно выбора каталога

- Нажать **Configure**.
- После успешного конфигурирования найти и выставить параметры:

  - ``CMAKE_INSTALL_PREFIX`` -> ``C:/opencv-4.10.0-build-cuda/install``
  - ``EXECUTABLE_OUTPUT_PATH`` -> ``C:/opencv-4.10.0-build-cuda/install/bin``
  - ``OPENCV_EXTRA_MODULES_PATH`` -> ``C:/opencv-4.10.0-build-cuda/install/opencv_contrib/modules``
  - Отметить галочкой ``WITH_CUDA``
  
  **Примечание:** Если переменные отсутствуют в перечне, нужно поставить галочку в пункте Advanced.

- Нажать **Configure** и выставить дополнительные параметры:

  - Отметить ``CUDA_FAST_MATH``, ``OPENCV_DNN_CUDA``, ``ENABLE_FAST_MATH``, ``WITH_OPENGL``
  - Снять галочки с ``WITH_NVCUVENC``, ``WITH_NVCUVID``, ``WITH_VTK``
  - Указать архитектуру видеокарты в ``CUDA_ARCH_BIN`` (например, 7.5 для NVIDIA RTX 20xx)
  - Если cuDNN установлен в нестандартном месте, указать пути:
  
    - ``CUDNN_LIBRARY`` -> путь к файлу ``cudnn.lib``
    - ``CUDNN_INCLUDE_DIR`` -> путь к папке ``include`` cuDNN

- Нажать **Generate**.
- После генерации нажать **Open Project** для запуска проекта Visual Studio.
- В обозревателе решений Visual Studio в папке CMakeTargets нажать правой кнопкой на **ALL_BUILD** и выбрать **Build**.

  .. image:: /_static/solution_explorer.jpg
     :alt: Окно обозревателя решений

- После успешной сборки выполнить сборку конфигурации **INSTALL**.

Для установки библиотеки MRCV необходимо выполнить следующие действия:

- Клонировать актуальную версию проекта, используя команды:

  .. code-block:: console
  
     cd ~
     git clone --branch main https://github.com/valabsoft/code-ai-400393.git

- Установить зависимости (CUDA Toolkit 12.4 и cuDNN 9.3) библиотеки, указанные ранее: libtorch, OpenCV, yaml-cpp (отображены рекомендованные пути для ОС Windows):

  .. code-block:: text
  
     C:\
     ├───libtorch-12.4-cuda
     ├───opencv-4.10.0-build-cuda
     └───yaml-cpp

- Для корректной работы библиотеки под управлением операционной системы Windows необходимо прописать системные пути в переменные окружения:

  .. code-block:: text
  
      C:\opencv-4.10.0-build-cuda\install\x64\vc17\bin
      C:\libtorch-12.4-cuda\Release\bin
      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin

-	Настроить сборку, установив ключ поддержки CUDA. Для этого отредактировать CMakeLists.txt проекта

  .. code-block:: text

      option(USE_CUDA "Use CUDA Build" ON)

- Запустить проект от имени администратора и открыть mrcv как локальную папку (File → Open → Folder);
- Настройте CMakeLists.txt и CMakeLists.txt в каждом примере в папке examples, убедившись, что пути к libtorch, opencv и yaml-cpp указаны корректно (см. пример ниже)

  .. code-block:: text

      if(USE_CUDA)
         # Добавляем макрос MRCV_CUDA_ENABLED
         target_compile_definitions(mrcv PRIVATE MRCV_CUDA_ENABLED)
         if(WIN32)
            set(CMAKE_PREFIX_PATH "C:/libtorch-12.4-cuda/Release;C:/yaml-cpp")
            set(OpenCV_DIR "C:/opencv-4.10.0-build-cuda/install")
            set(Torch_DIR "C:/libtorch-12.4-cuda/Release/share/cmake/Torch")
         
            set(OpenCV_INCLUDE_DIRS "C:/opencv-4.10.0-build-cuda/install/include")
            set(OpenCV_LIB_DIR "C:/opencv-4.10.0-build-cuda/install/x64/vc17/lib")
            
            set(Torch_INCLUDE_DIRS "C:/libtorch-12.4-cuda/Release/include")
            set(Torch_LIB_DIR "C:/libtorch-12.4-cuda/Release/lib")
         else()
            set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
         
            set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
            set(OpenCV_LIB_DIR "/usr/local/lib")
            
            set(Torch_INCLUDE_DIRS "/usr/local/include/torch")
            set(Torch_LIB_DIR "/usr/local/lib")
            
         endif()
      else() 
         if(WIN32)
            set(CMAKE_PREFIX_PATH "C:/libtorch-12.4/Release;C:/yaml-cpp")
            set(OpenCV_DIR "C:/opencv-4.10.0-build/install")
            set(Torch_DIR "C:/libtorch-12.4/Release/share/cmake/Torch")

            set(OpenCV_INCLUDE_DIRS "C:/opencv-4.10.0-build/install/include")
            set(OpenCV_LIB_DIR "C:/opencv-4.10.0-build/install/x64/vc17/lib")

            set(Torch_INCLUDE_DIRS "C:/libtorch-12.4/Release/include")
            set(Torch_LIB_DIR "C:/libtorch-12.4/Release/lib")
         else()
            set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
            
            set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
            set(OpenCV_LIB_DIR "/usr/local/lib") 
            
            set(Torch_INCLUDE_DIRS "/usr/local/include/torch")
            set(Torch_LIB_DIR "/usr/local/lib")
         endif()
      endif()
   
- Выбрать конфигурацию сборки dev-win;
- В разделе «Сборка» выбрать «Собрать проект»;
- После успешной сборки в разделе «Сборка» выбрать «Установить mrcv».

Для операционной системы Linux (Ubuntu) без поддержки CUDA (пошаговая версия)
-----------------------------------------------------------------------------

Необходимо выполнить следующий ряд действий:

**Установить зависимости с помощью набора команд**

  .. code-block:: console

     sudo apt update
     sudo apt install -y unzip wget curl build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev


**Установить библиотеку yaml-cpp**

  .. code-block:: console

     cd ~
     git clone https://github.com/jbeder/yaml-cpp.git
     cd yaml-cpp
     cmake .
     make -j$(nproc)
     sudo make install

**Установить библиотеку OpenCV**

Выполнить команды:

.. code-block:: console

   cd ~
   git clone https://github.com/opencv/opencv.git -b "4.10.0"
   git clone https://github.com/opencv/opencv_contrib.git -b "4.10.0"
   mkdir -p opencv/build && cd opencv/build
   cmake -D CMAKE_BUILD_TYPE=Release \
         -D CMAKE_INSTALL_PREFIX=/usr/local \
         -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
         ..
   sudo make -j$(nproc)
   sudo make install

Сборка осуществляется в папке build. При возникновении ошибок необходимо очистить папки build и .cache.

**Установить библиотеку LibTorch**

Скачать соответсвующий архив с библиотекой:

  .. code-block:: console

     cd ~
     curl -L "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip" -o libtorch-library.zip

Распаковать архив libtorch-library.zip с помощью команды:

.. code-block:: console
   
   sudo unzip -o libtorch-library.zip -d /opt/

Добавить путь к libtorch в динамический компоновщик с помощью команды

.. code-block:: console

   sudo sh -c "echo '/opt/libtorch/lib' >> /etc/ld.so.conf.d/libtorch.conf"

Обновить кэш динамического компоновщика с помощью команды:

.. code-block:: console

   sudo ldconfig

Добавить путь к заголовочным файлам и библиотекам в переменные окружения, отредактировав файл ~/.bashrc, открыв его при помощи команды

.. code-block:: console

   sudo nano  ~/.bashrc

и записав конец следующие строки:

.. code-block:: console

   export TORCH_INCLUDE=/opt/libtorch/include
   export TORCH_LIB=/opt/libtorch/lib
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_LIB
   export CPATH=$CPATH:$TORCH_INCLUDE
   export Torch_DIR=/opt/libtorch/share/cmake/Torch

затем сохранив (Ctrl + O, Ctrl + X) необходимо активировать изменения при помощи команды

.. code-block:: console

   source ~/.bashrc

Убедиться в правильности установки можно используя инструкцию https://docs.pytorch.org/cppdocs/installing.html.
При нехватке системных ресурсов при сборке рекомендуется запускать сборку через make без указания параметра -j.

Для установки библиотеки MRCV необходимо выполнить следующие действия:

- Клонировать актуальную версию проекта, используя команды:

  .. code-block:: console
  
     cd ~
     git clone --branch main https://github.com/valabsoft/code-ai-400393.git

- Установить библиотеки, указанные ранее

- Выполнить команды

  .. code-block:: console

      cd ~/code-ai-400393
      mkdir -p build && cd build
      cmake ..
      make -j$(nproc)
      sudo make install
      sudo ldconfig -v

Для операционной системы Linux (Ubuntu) без поддержки CUDA (версия с помощью скрипта)
-------------------------------------------------------------------------------------

Для установки библиотеки MRCV вместе с требующимися зависимостями возможно запустить shell-скрипт. Для этого нужно создать файл с помощью последовательности команд

  .. code-block:: console

      cd ~
      nano install_cpu.sh

Вставить код, предстваленный ниже, в файл

  .. code-block:: shell

      #!/bin/bash

      set -e

      sudo apt update
      sudo apt install -y unzip wget curl build-essential cmake git \
         libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
         libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev

      cd ~
      if [ ! -d yaml-cpp ]; then
         git clone https://github.com/jbeder/yaml-cpp.git
      fi
      cd yaml-cpp
      cmake .
      make -j$(nproc)
      sudo make install

      cd ~
      if [ ! -f libtorch-library.zip ]; then
         curl -L "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip" -o libtorch-library.zip
      fi
      sudo unzip -o libtorch-library.zip -d /opt/

      TORCH_CONF="/etc/ld.so.conf.d/libtorch.conf"
      if ! grep -q "/opt/libtorch/lib" "$TORCH_CONF" 2>/dev/null; then
         echo "/opt/libtorch/lib" | sudo tee "$TORCH_CONF"
         sudo ldconfig
      fi

      BASHRC="$HOME/.bashrc"
      ENV_MARK="# BEGIN TORCH ENV"
      if ! grep -q "$ENV_MARK" "$BASHRC"; then
         echo "$ENV_MARK" >> "$BASHRC"
         echo "export TORCH_INCLUDE=/opt/libtorch/include" >> "$BASHRC"
         echo "export TORCH_LIB=/opt/libtorch/lib" >> "$BASHRC"
         echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$TORCH_LIB" >> "$BASHRC"
         echo "export CPATH=\$CPATH:\$TORCH_INCLUDE" >> "$BASHRC"
         echo "export Torch_DIR=/opt/libtorch/share/cmake/Torch" >> "$BASHRC"
         echo "# END TORCH ENV" >> "$BASHRC"
      fi

      source "$BASHRC"

      cd ~
      if [ ! -d opencv ]; then
         git clone https://github.com/opencv/opencv.git -b "4.10.0"
      fi
      if [ ! -d opencv_contrib ]; then
         git clone https://github.com/opencv/opencv_contrib.git -b "4.10.0"
      fi
      mkdir -p opencv/build && cd opencv/build

      cmake -D CMAKE_BUILD_TYPE=Release \
            -D CMAKE_INSTALL_PREFIX=/usr/local \
            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
            ..

      sudo make -j$(nproc)
      sudo make install
      sudo ldconfig

      cd ~
      if [ ! -d code-ai-400393 ]; then
         git clone --branch main https://github.com/valabsoft/code-ai-400393.git
      fi
      cd code-ai-400393
      mkdir -p build && cd build

      cmake ..
      make -j$(nproc)
      sudo make install
      sudo ldconfig -v

Сохранить содержимое файла *Ctrl + O* и закрыть файл *Ctrl + X*. Сделать файл исполняемым с помощью команды

  .. code-block:: console

      chmod +x install_cpu.sh

Запустить скрипт

   .. code-block:: console

      ./install_cpu.sh


Для операционной системы Linux (Ubuntu) с поддержкой CUDA (пошаговая версия)
----------------------------------------------------------------------------

Необходимо выполнить следующий ряд действий:

**Установить зависимости с помощью набора команд**

  .. code-block:: console

   sudo apt update
   sudo apt install -y unzip wget curl build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev


**Установить библиотеку yaml-cpp**

  .. code-block:: console

   cd ~
   git clone https://github.com/jbeder/yaml-cpp.git
   cd yaml-cpp
   cmake .
   make -j$(nproc)
   sudo make install

**Установить CUDA Toolkit 12.4**

  .. code-block:: console

      cd ~
      wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
      sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
      sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
      sudo apt-get update
      sudo apt-get -y install cuda-toolkit-12-4

**Установить cuDNN 9.3**

  .. code-block:: console

      cd ~
      wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
      sudo dpkg -i cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
      sudo cp /var/cudnn-local-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
      sudo apt-get update
      sudo apt-get -y install cudnn

**Установить библиотеку OpenCV**

Выполнить последовательность команд

.. code-block:: console

   cd ~
   git clone https://github.com/opencv/opencv.git -b "4.10.0"
   git clone https://github.com/opencv/opencv_contrib.git -b "4.10.0"
   mkdir -p opencv/build && cd opencv/build
   cmake .. \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_opencv_world=OFF \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_cudacodec=ON \
      -D BUILD_opencv_ximgproc=ON \
      -D BUILD_opencv_tracking=ON \
      -D BUILD_opencv_face=ON \
      -D BUILD_opencv_text=ON \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=8.6 \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_CUDNN=ON \
      -D WITH_OPENGL=ON \
      -D CUDA_HOST_COMPILER=/usr/bin/gcc-11 \
      -D OPENCV_GENERATE_PKGCONFIG=ON
   make -j$(nproc)
   sudo make install

Сборка осуществляется в папке build. При возникновении ошибок необходимо очистить папки build и .cache.

**Установить библиотеку LibTorch**

Скачать соответсвующий архив с библиотекой

  .. code-block:: console

   cd ~
   curl -L "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip" -o libtorch-library.zip

Распаковать архив libtorch-library.zip с помощью команды

.. code-block:: console
   
   sudo unzip -o libtorch-library.zip -d /opt/

Добавить путь к libtorch в динамический компоновщик с помощью команды

.. code-block:: console

   sudo sh -c "echo '/opt/libtorch/lib' >> /etc/ld.so.conf.d/libtorch.conf"

Обновить кэш динамического компоновщика с помощью команды

.. code-block:: console

   sudo ldconfig

Добавить путь к заголовочным файлам и библиотекам в переменные окружения, отредактировав файл ~/.bashrc, открыв его при помощи команды

.. code-block:: console

   sudo nano  ~/.bashrc

и записать в конец файла следующие строки

.. code-block:: console

   export TORCH_INCLUDE=/opt/libtorch/include
   export TORCH_LIB=/opt/libtorch/lib
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_LIB
   export CPATH=$CPATH:$TORCH_INCLUDE
   export Torch_DIR=/opt/libtorch/share/cmake/Torch

Сохранить изменения *Ctrl + O* и закрыть файл *Ctrl + X*. Активировать изменения с помощью команды

.. code-block:: console

   source ~/.bashrc

Убедиться в правильности установки можно используя инструкцию https://docs.pytorch.org/cppdocs/installing.html.
При нехватке системных ресурсов при сборке рекомендуется запускать сборку через make без указания параметра -j.

Для установки библиотеки MRCV необходимо выполнить следующие действия:

- Клонировать актуальную версию проекта, используя команды:

  .. code-block:: console
  
     cd ~
     git clone --branch main https://github.com/valabsoft/code-ai-400393.git

- Установить библиотеки, указанные ранее

- Выполнить последовательность команд

  .. code-block:: console

      cd ~/code-ai-400393
      sed -i 's/option(USE_CUDA "Use CUDA Build" OFF)/option(USE_CUDA "Use CUDA Build" ON)/' CMakeLists.txt
      mkdir -p build && cd build
      sudo cmake -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc ..
      sudo make -j$(nproc)
      sudo make install
      sudo ldconfig -v

Для операционной системы Linux (Ubuntu) с поддержкой CUDA (версия с помощью скрипта)
------------------------------------------------------------------------------------

Для установки библиотеки MRCV вместе с требующимися зависимостями возможно запустить shell-скрипт.
Для этого необходимо создать файл с помощью команды

  .. code-block:: console

      cd ~
      nano install_cuda.sh

В созданный файл вставить содержимое, представленное ниже

  .. code-block:: shell

      #!/bin/bash

      set -e

      sudo apt update
      sudo apt install -y unzip wget curl build-essential cmake git \
         libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
         libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev

      cd ~
      if [ ! -d yaml-cpp ]; then
         git clone https://github.com/jbeder/yaml-cpp.git
      fi
      cd yaml-cpp
      cmake .
      make -j$(nproc)
      sudo make install

      cd ~
      if [ ! -f cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb ]; then
      wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
      fi
      sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
      sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
      sudo apt-get update
      sudo apt-get -y install cuda-toolkit-12-4

      cd ~
      if [ ! -f cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb ]; then
      wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
      fi
      sudo dpkg -i cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
      sudo cp /var/cudnn-local-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
      sudo apt-get update
      sudo apt-get -y install cudnn

      cd ~
      if [ ! -d opencv ]; then
         git clone https://github.com/opencv/opencv.git -b "4.10.0"
      fi
      if [ ! -d opencv_contrib ]; then
         git clone https://github.com/opencv/opencv_contrib.git -b "4.10.0"
      fi
      mkdir -p opencv/build && cd opencv/build

      cmake .. \
         -D CMAKE_BUILD_TYPE=Release \
         -D CMAKE_INSTALL_PREFIX=/usr/local \
         -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
         -D BUILD_opencv_world=OFF \
         -D BUILD_opencv_python3=ON \
         -D BUILD_opencv_cudacodec=ON \
         -D BUILD_opencv_ximgproc=ON \
         -D BUILD_opencv_tracking=ON \
         -D BUILD_opencv_face=ON \
         -D BUILD_opencv_text=ON \
         -D WITH_CUDA=ON \
         -D CUDA_ARCH_BIN=8.6 \
         -D ENABLE_FAST_MATH=ON \
         -D CUDA_FAST_MATH=ON \
         -D WITH_CUBLAS=ON \
         -D WITH_CUDNN=ON \
         -D WITH_OPENGL=ON \
         -D CUDA_HOST_COMPILER=/usr/bin/gcc-11 \
         -D OPENCV_GENERATE_PKGCONFIG=ON

      make -j$(nproc)
      sudo make install
      sudo ldconfig

      cd ~
      if [ ! -f libtorch-library.zip ]; then
         curl -L "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip" -o libtorch-library.zip
      fi
      sudo unzip -o libtorch-library.zip -d /opt/

      TORCH_CONF="/etc/ld.so.conf.d/libtorch.conf"
      if ! grep -q "/opt/libtorch/lib" "$TORCH_CONF" 2>/dev/null; then
         echo "/opt/libtorch/lib" | sudo tee "$TORCH_CONF"
         sudo ldconfig
      fi

      BASHRC="$HOME/.bashrc"
      ENV_MARK="# BEGIN TORCH ENV"
      if ! grep -q "$ENV_MARK" "$BASHRC"; then
         echo "$ENV_MARK" >> "$BASHRC"
         echo "export TORCH_INCLUDE=/opt/libtorch/include" >> "$BASHRC"
         echo "export TORCH_LIB=/opt/libtorch/lib" >> "$BASHRC"
         echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$TORCH_LIB" >> "$BASHRC"
         echo "export CPATH=\$CPATH:\$TORCH_INCLUDE" >> "$BASHRC"
         echo "export Torch_DIR=/opt/libtorch/share/cmake/Torch" >> "$BASHRC"
         echo "# END TORCH ENV" >> "$BASHRC"
      fi

      source "$BASHRC"

      cd ~
      if [ ! -d code-ai-400393 ]; then
         git clone --branch main https://github.com/valabsoft/code-ai-400393.git
      fi
      cd code-ai-400393
      sed -i 's/option(USE_CUDA "Use CUDA Build" OFF)/option(USE_CUDA "Use CUDA Build" ON)/' CMakeLists.txt
      mkdir -p build && cd build

      sudo cmake -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc ..
      sudo make -j$(nproc)
      sudo make install
      sudo ldconfig -v

Сохранить содержимое файла *Ctrl + O* и закрыть файл *Ctrl + X*.

Сделать файл исполняемым с помощью последовательности команд

  .. code-block:: console

      chmod +x install_cuda.sh

Запустить скрипт

   .. code-block:: console
      
      ./install_cuda.sh

Инструкция по установке Python версии библиотеки
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Минимальная версия Python для работы с библиотекой - 3.10.
Для установки библиотеки необходимо выполнить следующие действия:

- Проверить, установлен ли Python

  .. code-block:: console

      python3 --version

- Если Python отсутствует, установить с помощью команды

  .. code-block:: console

      sudo apt-get update
      sudo apt-get install python3

- Проверить, установлен ли Git

  .. code-block:: console

      git --version

- Если Git не установлен, установить с помощью команды

  .. code-block:: console

      sudo apt-get install git

- Клонировать репозиторий кода из ветки **main**

  .. code-block:: console

      git clone --branch main https://github.com/valabsoft/code-ai-400393.git

- Перейти в локальную копию репозитория на устройстве

  .. code-block:: console

   cd code-ai-400393/python

- Рекомендуется использовать виртуальное окружение для изоляции зависимостей. Для этого необходимо выполнить следующие действия.

Установить соответствующий пакет:

  .. code-block:: console

      sudo apt install python3.10-venv

Создайте виртуальное окружение с именем *venv*

  .. code-block:: console

      python3 -m venv venv

Активировать окружение

  .. code-block:: console

      source venv/bin/activate

После этого в терминале появится *venv*, что указывает на активное окружение.

- В директории есть файлы requirements.txt и requirements_cuda.txt с необходимыми зависимостями для версий без поддержки CUDA и с поддержкой. Если файлы находится в текущей рабочей директории и требуется версия без CUDA, то выполнить команду

  .. code-block:: console

      pip install -r requirements.txt

Если необходима версия с CUDA, то необходимо воспользоваться файлом requirements_cuda.txt, выполнив команду

  .. code-block:: console

      pip install -r requirements_cuda.txt

- Установите библиотеку с помощью команды

  .. code-block:: console

      pip install -e .

- Перейти в директорию с примерами с помощью команды

   .. code-block:: console

      cd examples

- Выбрать директорию модуля и запустить пример. Предварительно убедиться, что в папке присутствуют файлы, использующиеся в качестве исходных данных. Например:

  .. code-block:: console

      cd comparing/
      python comparing.py


Для mini-ПК
~~~~~~~~~~~

Использование функций библиотеки доступно на мини-ПК типа NVIDIA Jetson, Raspberry Pi и др. Для предварительной настройки библиотеки на этом классе устройств рекомендуется воспользоваться рекомендациям, изложенными в разделе `Установка Python версии <#mrcv-python-version>`_.

Подготовка данных для работы с примерами
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

При знакомстве с библиотекой mrcv, после изучения состава модулей библиотеки, рекомендуется обратиться к демонстрационным примерам. Все примеры библиотеки снабжены необходимыми файлами, которые используются в качестве входных данных. После сборки примера рекомендуется скопировать в папку сборки примера папку *files* из репозитория кода. После копирования исходных данных выполнить запуски примера.

*Пример*

Демонстрация работы функций модуля сегментации (для операционной системы Windows).
Код примера находится в папке *code-ai-400393\\examples\\segmentationtest*

Порядок работы:

- Собрать исполняемый файл примера. В случае успешной сборки, исполняемый файл примера будет создан в папке *code-ai-400393\\build\\examples\\segmentationtest*

- Скопировать папку *files* с исходными данными из папки *code-ai-400393\\examples\\segmentationtest\\files* в папку *code-ai-400393\\build\\examples\\segmentationtest\\files*

- Запустить исполняемый файл примера *mrcv-segmentationtest.exe*

- В результате работы функции будет открыто два окна с исходным изображением и с изображением, полученным в результате работы функции сегмнетации

*Исходное изображение*

  .. image:: /_static/seg-source.jpg
     :alt: Исходное изображение

*Результат работы функции*

  .. image:: /_static/seg-prediction.jpg
     :alt: Исходное изображение

- После работы с исходными данными примера по-умолчанию, внести изменения в код примера, указав собственные входные данные или настройки параметров функции

*Например*

Изменить путь к исходному файлу с изображением

.. code-block:: cpp

   segmentor.Predict(image, "ship", true);

или отключить показ окон, изменив параметры функции

.. code-block:: cpp

   segmentor.Predict(image, "ship", false);

Запуск примеров библиотеки
~~~~~~~~~~~~~~~~~~~~~~~~~~

Запуск демонстрационного примера augmentation (остальные примеры запускаются по аналогии)

1. Перейти в папку *build/examples* любым удобным способом

2. Выбрать папку примера

.. code-block:: console
   
   cd augmentation

3. Скопировать папку *files* из *examples/augmentation* в *build/examples/augmentation*

4. Запустить исполняемый файл

.. code-block:: console
   
   ./mrcv-augmentation