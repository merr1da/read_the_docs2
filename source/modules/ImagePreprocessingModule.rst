Модуль предварительной обработки изображений
============================================

*Данный модуль содержит функции для предварительной обработки изображений, включая коррекцию яркости, контраста, резкости, фильтрацию шумов и геометрическую коррекцию.*

Функция создания изображения с сообщением об ошибке
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Создаёт изображение с текстом ошибки для отображения при возникновении проблем.*

.. code-block:: cpp

    cv::Mat mrcv::getErrorImage(std::string textError)

**Описание параметров:**

- ``textError`` - текст сообщения об ошибке.

**Возвращаемое значение:**
- Изображение размером 600x960 с текстом ошибки.

Коррекция яркости изображения
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Применяет гамма-коррекцию для изменения яркости изображения.*

.. code-block:: cpp
    
    int mrcv::changeImageBrightness(cv::Mat& imageInput, cv::Mat& imageOutput, double gamma)


**Описание параметров:**

- ``imageInput`` - входное изображение.
- ``imageOutput`` - выходное изображение после коррекции.
- ``gamma`` - параметр гамма-коррекции (значения <1 увеличивают яркость, >1 уменьшают).

**Возвращаемые коды:**

- ``0`` - успешное выполнение
- ``1`` - пустое входное изображение
- ``-1`` - необработанное исключение

Увеличение резкости (метод Лапласа)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Увеличивает резкость изображения с использованием оператора Лапласа.*

.. code-block:: cpp

    int mrcv::sharpeningImage01(cv::Mat& imageInput, cv::Mat& imageOutput, double gainFactorHighFrequencyComponent)

**Описание параметров:**

- ``gainFactorHighFrequencyComponent`` - коэффициент усиления высокочастотных компонент.

**Возвращаемые коды:**

- ``0`` - успешное выполнение
- ``1`` - пустое входное изображение
- ``-1`` - необработанное исключение

Увеличение резкости (метод Гаусса)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Увеличивает резкость изображения с использованием размытия по Гауссу.*

.. code-block:: cpp

    int mrcv::sharpeningImage02(cv::Mat& imageInput, cv::Mat& imageOutput, cv::Size filterSize, double sigmaFilter, double gainFactorHighFrequencyComponent)

**Описание параметров:**

- ``filterSize`` - размер ядра фильтра.
- ``sigmaFilter`` - параметр размытия.
- ``gainFactorHighFrequencyComponent`` - коэффициент усиления высокочастотных компонент.

Основная функция предварительной обработки
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Применяет последовательность методов обработки к изображению согласно переданному списку.*

.. code-block:: cpp

    int mrcv::preprocessingImage(cv::Mat& image, std::vector<mrcv::METOD_IMAGE_PERPROCESSIN> metodImagePerProcessing, const std::string& pathToFileCameraParametrs)

**Поддерживаемые методы обработки:**

- ``CONVERTING_BGR_TO_GRAY`` - преобразование в градации серого
- ``BRIGHTNESS_LEVEL_UP/DOWN`` - коррекция яркости
- ``BALANCE_CONTRAST_*`` - различные методы коррекции контраста
- ``SHARPENING_*`` - методы увеличения резкости
- ``NOISE_FILTERING_*`` - методы фильтрации шумов
- ``CORRECTION_GEOMETRIC_DEFORMATION`` - геометрическая коррекция

**Параметры:**

- ``metodImagePerProcessing`` - вектор методов обработки для применения.
- ``pathToFileCameraParametrs`` - путь к файлу параметров камеры для геометрической коррекции.

Увеличение контраста изображения
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Применяет различные методы увеличения контраста в разных цветовых пространствах.*

.. code-block:: cpp

    int mrcv::increaseImageContrast(cv::Mat& imageInput, cv::Mat& imageOutput, mrcv::METOD_INCREASE_IMAGE_CONTRAST metodIncreaseContrast, mrcv::COLOR_MODEL colorSpace, double clipLimitCLAHE, cv::Size gridSizeCLAHE, float percentContrastBalance, double mContrastExtantion, double eContrastExtantion)

**Поддерживаемые методы:**

- ``EQUALIZE_HIST`` - эквализация гистограммы
- ``CLAHE`` - адаптивная эквализация гистограммы
- ``CONTRAST_BALANCING`` - балансировка контраста
- ``CONTRAST_EXTENSION`` - расширение контраста

**Поддерживаемые цветовые пространства:**

- ``CM_HSV``
- ``CM_LAB``
- ``CM_YCBCR``
- ``CM_RGB``

Балансировка контраста
~~~~~~~~~~~~~~~~~~~~~~
*Выполняет балансировку контраста изображения по заданному проценту.*

.. code-block:: cpp

    int mrcv::contrastBalancing(cv::Mat& planeArray, float percent)

**Параметры:**

- ``percent`` - процент обрезки гистограммы (0-100).

Расширение контраста
~~~~~~~~~~~~~~~~~~~~
*Применяет нелинейное преобразование для расширения динамического диапазона.*

.. code-block:: cpp

    int mrcv::contrastExtantion(cv::Mat& planeArray, double m, double e)

**Параметры:**

- ``m`` - среднее значение (если <0, вычисляется автоматически)
- ``e`` - параметр экспоненты