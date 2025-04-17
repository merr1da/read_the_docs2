Модуль сегментации изображений
===============================

*Модуль реализует функционал семантической сегментации изображений с использованием архитектуры FPN (Feature Pyramid Network) на базе предобученных моделей ResNet.*

Архитектура FPN
~~~~~~~~~~~~~~~

**FPNImpl** - основной класс, реализующий архитектуру Feature Pyramid Network.

.. code-block:: cpp

    class FPNImpl : public torch::nn::Module

**Компоненты:**
- Энкодер (ResNet) для извлечения признаков
- Декодер для построения пирамиды признаков
- Голова сегментации для генерации масок

**Инициализация:**

.. code-block:: cpp

    FPNImpl(int _numberClasses, std::string encoderName, std::string pretrainedPath, 
            int encoderDepth, int decoderChannelPyramid, int decoderChannelsSegmentation,
            std::string decoderMergePolicy, float decoder_dropout, double upsampling)

**Параметры:**
- ``_numberClasses`` - количество классов сегментации
- ``encoderName`` - имя модели энкодера (resnet18, resnet34, resnet50)
- ``pretrainedPath`` - путь к предобученным весам
- ``encoderDepth`` - глубина энкодера (3-5)

Класс Segmentor
~~~~~~~~~~~~~~~

**Основной интерфейс для работы с сегментацией:**

**Инициализация**

.. code-block:: cpp

    void Segmentor::Initialize(int gpu_id, int _width, int _height, 
                             std::vector<std::string>&& _listName,
                             std::string encoderName, std::string pretrainedPath)

**Параметры:**
- ``gpu_id`` - идентификатор GPU (-1 для CPU)
- ``_width``, ``_height`` - размеры входного изображения
- ``_listName`` - список классов сегментации
- ``encoderName`` - имя модели энкодера
- ``pretrainedPath`` - путь к предобученным весам

**Обучение модели**

.. code-block:: cpp

    void Segmentor::Train(float learning_rate, unsigned int epochs, int batch_size,
                        std::string train_val_path, std::string imageType, std::string save_path)

**Параметры обучения:**
- ``learning_rate`` - начальная скорость обучения
- ``epochs`` - количество эпох
- ``batch_size`` - размер батча
- ``train_val_path`` - путь к данным обучения/валидации
- ``imageType`` - расширение изображений (".jpg", ".png")
- ``save_path`` - путь для сохранения модели

**Загрузка весов**

.. code-block:: cpp

    void Segmentor::LoadWeight(std::string pathWeight)

**Предсказание**

.. code-block:: cpp

    void Segmentor::Predict(cv::Mat& image, const std::string& which_class)

**Параметры:**
- ``image`` - входное изображение
- ``which_class`` - имя класса для визуализации

Вспомогательные классы
~~~~~~~~~~~~~~~~~~~~~~

**SegDataset**
*Класс для работы с данными сегментации*

.. code-block:: cpp

    class SegDataset : public torch::data::Dataset<SegDataset>

**Функционал:**
- Загрузка изображений и масок
- Аугментация данных
- Преобразование в тензоры

**Augmentations**

*Класс для аугментации данных*

.. code-block:: cpp

    class Augmentations

**Методы:**
- ``Resize`` - изменение размера изображения и маски

Функции потерь
~~~~~~~~~~~~~~

**DiceLoss**

.. code-block:: cpp

    torch::Tensor DiceLoss(torch::Tensor prediction, torch::Tensor target, int classNum)

**CELoss**

.. code-block:: cpp

    torch::Tensor CELoss(torch::Tensor prediction, torch::Tensor target)
