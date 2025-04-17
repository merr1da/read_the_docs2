Модуль служебных функций
========================

*В данном модуле реализованы вспомогательные функции для работы с изображениями, видео, калибровкой камер и логированием.*

Функции работы с изображениями
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Функция сравнения изображений**

*Сравнивает два изображения и возвращает процент различий.*

.. code-block:: cpp

    double mrcv::compareImages(cv::Mat img1, cv::Mat img2, bool methodCompare)

**Параметры:**

- ``img1`` - первое изображение для сравнения
- ``img2`` - второе изображение для сравнения
- ``methodCompare`` - метод сравнения (true/false)

**Возвращает:** процент различий между изображениями

**Функция морфологического преобразования**

*Применяет морфологические операции к изображению.*

.. code-block:: cpp

    int mrcv::morphologyImage(cv::Mat image, std::string out, METOD_MORF metod, int morph_size)

**Параметры:**
- ``image`` - входное изображение
- ``out`` - путь для сохранения результата
- ``metod`` - тип морфологической операции (OPEN, CLOSE, GRADIENT, ERODE, DILAT)
- ``morph_size`` - размер ядра преобразования

Функции работы с видео
~~~~~~~~~~~~~~~~~~~~~~
**Запись видеопотока**

*Записывает видео с камеры в файл.*

.. code-block:: cpp

    int mrcv::recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec)

**Параметры:**
- ``cameraID`` - идентификатор камеры
- ``recorderInterval`` - интервал записи в секундах
- ``fileName`` - имя файла для сохранения
- ``codec`` - используемый кодек (XVID, MJPG, mp4v, h265)

**Возвращает:** код результата (0 - успех)

Функции калибровки камер
~~~~~~~~~~~~~~~~~~~~~~~~
**Калибровка одиночной камеры**

*Выполняет калибровку одной камеры по изображениям шахматной доски.*

.. code-block:: cpp

    void mrcv::cameraCalibrationMono(std::vector<cv::String> images, std::string pathToImages, 
                                   CalibrationParametersMono& calibrationParameters, 
                                   int chessboardColCount, int chessboardRowCount, 
                                   float chessboardSquareSize)

**Параметры:**

- ``images`` - вектор имен файлов изображений
- ``pathToImages`` - путь к папке с изображениями
- ``calibrationParameters`` - структура для хранения параметров калибровки
- ``chessboardColCount`` - количество углов по горизонтали
- ``chessboardRowCount`` - количество углов по вертикали
- ``chessboardSquareSize`` - размер клетки шахматной доски в мм

**Калибровка стереопары**

*Выполняет калибровку стереопары камер.*

.. code-block:: cpp

    void mrcv::cameraCalibrationStereo(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR,
                                     std::string pathToImagesL, std::string pathToImagesR,
                                     CalibrationParametersStereo& calibrationParameters,
                                     int chessboardColCount, int chessboardRowCount,
                                     float chessboardSquareSize)

**Параметры:**

- ``imagesL``, ``imagesR`` - векторы имен файлов для левой и правой камер
- ``pathToImagesL``, ``pathToImagesR`` - пути к папкам с изображениями
- ``calibrationParameters`` - структура для хранения параметров калибровки
- ``chessboardColCount``, ``chessboardRowCount`` - количество углов доски
- ``chessboardSquareSize`` - размер клетки в мм

Функции логирования
~~~~~~~~~~~~~~~~~~~
**Запись в лог-файл**

*Записывает сообщение в лог-файл с указанием типа.*

.. code-block:: cpp

    void mrcv::writeLog(std::string logText, LOGTYPE logType = LOGTYPE::INFO)

**Параметры:**

- ``logText`` - текст сообщения
- ``logType`` - тип сообщения (DEBUG, ERROR, EXCEPTION, INFO, WARNING)

**Генерация уникального имени файла**

*Создает уникальное имя файла на основе текущего времени.*

.. code-block:: cpp

    std::string mrcv::generateUniqueFileName(std::string fileName, std::string fileExtension)

**Параметры:**

- ``fileName`` - префикс имени файла
- ``fileExtension`` - расширение файла

**Возвращает:** строку с уникальным именем файла

Функции работы с изображениями из интернета
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Загрузка изображений с Яндекса**

*Скачивает изображения по запросу из Яндекс.Картинок.*

.. code-block:: cpp

    int mrcv::getImagesFromYandex(
        std::string query,
        int minWidth,
        int minHeight,
        std::string nameTemplate,
        std::string outputFolder,
        bool separateDataset,
        unsigned int trainsetPercentage,
        unsigned int countFoto,
        bool money,
        std::string key,
        std::string secretKey
    )

**Параметры:**

- ``query`` - строка поискового запроса
- ``minWidth``, ``minHeight`` - минимальные размеры изображений
- ``nameTemplate`` - шаблон имени файла
- ``outputFolder`` - папка для сохранения
- ``separateDataset`` - флаг разделения на train/test
- ``trainsetPercentage`` - процент для обучающей выборки
- ``countFoto`` - количество изображений для скачивания
- ``money`` - флаг платного API
- ``key``, ``secretKey`` - ключи API

Функции построения карты диспаратности
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Построение карты диспаратности**

*Строит карту диспаратности для стереопары изображений.*

.. code-block:: cpp

    int mrcv::disparityMap(cv::Mat& map, const cv::Mat& imageLeft, const cv::Mat& imageRight,
                          int minDisparity, int numDisparities, int blockSize,
                          double lambda, double sigma, DISPARITY_TYPE disparityType,
                          int colorMap, bool saveToFile, bool showImages)

**Параметры:**

- ``map`` - выходная карта диспаратности
- ``imageLeft``, ``imageRight`` - изображения стереопары
- ``minDisparity`` - минимальная диспаратность
- ``numDisparities`` - количество уровней диспаратности
- ``blockSize`` - размер блока для сравнения
- ``lambda``, ``sigma`` - параметры фильтрации
- ``disparityType`` - тип выходной карты
- ``colorMap`` - цветовая схема
- ``saveToFile`` - флаг сохранения в файл
- ``showImages`` - флаг отображения окон

**Возвращает:** код результата (0 - успех)