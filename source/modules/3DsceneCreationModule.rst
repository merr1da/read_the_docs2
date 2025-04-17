Модуль создания 3D сцен
=======================

*Модуль предоставляет функционал для обработки стереоизображений, построения 3D сцен и работы с облаками точек. Включает методы калибровки, построения карт диспаратности, сегментации объектов и визуализации.*

Формирование стереопары
~~~~~~~~~~~~~~~~~~~~~~~
*Объединяет изображения с двух камер в единое изображение для обработки.*

.. code-block:: cpp

    int mrcv::makingStereoPair(cv::Mat& inputImageCamera01, cv::Mat& inputImageCamera02, cv::Mat& outputStereoPair)

**Параметры:**

- ``inputImageCamera01`` - изображение с левой камеры (CV_8UC3)
- ``inputImageCamera02`` - изображение с правой камеры (CV_8UC3)
- ``outputStereoPair`` - результирующее объединенное изображение

**Возвращаемые значения:**

- ``0`` - успех
- ``1`` - пустое изображение
- ``-1`` - ошибка

**Особенности:**

- Автоматический расчет размеров выходного изображения
- Горизонтальное объединение изображений
- Поддержка разных размеров входных изображений

Построение карты диспаратности
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Вычисляет карту различий между стереоизображениями.*

.. code-block:: cpp

    int mrcv::find3dPointsADS(cv::Mat& inputImageCamera01, cv::Mat& inputImageCamera02, 
        mrcv::pointsData& points3D, mrcv::settingsMetodDisparity& settingsMetodDisparity,
        cv::Mat& disparityMap, mrcv::cameraStereoParameters& cameraParameters,
        int limitOutPoints, std::vector<double> limitsOutlierArea)

**Методы построения:**

- ``MODE_BM`` - блочное сопоставление (быстрый)
- ``MODE_SGBM`` - полуглобальное сопоставление
- ``MODE_HH`` - двухпроходный алгоритм

**Особенности:**

- Фильтрация выбросов по заданной области
- Ограничение количества выходных точек
- Поддержка различных цветовых карт

Сегментация объектов
~~~~~~~~~~~~~~~~~~~~
*Обнаружение и сегментация объектов нейронной сетью.*

.. code-block:: cpp

    int mrcv::detectingSegmentsNeuralNet(cv::Mat& imageInput, cv::Mat& imageOutput,
        std::vector<cv::Mat>& replyMasks, const std::string filePathToModelYoloNeuralNet,
        const std::string filePathToClasses)

**Особенности:**

- Использование модели YOLO
- Возврат бинарных масок объектов
- Настройка порогов уверенности
- Морфологическая постобработка масок

Визуализация 3D сцены
~~~~~~~~~~~~~~~~~~~~~
*Проецирование 3D точек на 2D изображение.*

.. code-block:: cpp

    int mrcv::getImage3dSceene(mrcv::pointsData& points3D, 
        mrcv::parameters3dSceene& parameters3dSceene,
        mrcv::cameraStereoParameters& cameraParameters, 
        cv::Mat& outputImage3dSceene)

**Параметры сцены:**

- Углы поворота (angX/Y/Z)
- Смещение (tX/Y/Z)
- Масштаб (scale)
- Расстояние до центра (dZ)

Сохранение результатов
~~~~~~~~~~~~~~~~~~~~~~
*Экспорт данных о 3D точках.*

.. code-block:: cpp

    int mrcv::saveInFile3dPointsInObjectsSegments(pointsData& points3D, 
        const cv::String pathToFile)

**Формат данных:**

- Координаты пикселей и 3D координаты
- Цвет точек
- Идентификаторы сегментов