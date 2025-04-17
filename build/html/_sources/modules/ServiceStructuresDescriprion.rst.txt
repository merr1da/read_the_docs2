Описание служебных структур 
===========================
Ниже приведены основные структуры, перечисления и параметры, используемые в библиотеке Marine Robotics Computer Vision для хранения данных, настройки алгоритмов и конфигурации камер.

Перечисления:
~~~~~~~~~~~~~

**enum class METOD_MORF** — методы морфологических преобразований:
- `OPEN`, `CLOSE`, `GRADIENT`, `ERODE`, `DILAT`

**enum class CODEC** — поддерживаемые видео-кодеки:
- `XVID`, `MJPG`, `mp4v`, `h265`

**enum class LOGTYPE** — типы записей в лог-файл:
- `DEBUG`, `ERROR`, `EXCEPTION`, `INFO`, `WARNING`

**enum class METOD_INCREASE_IMAGE_CONTRAST** — методы повышения контрастности изображения:
- `EQUALIZE_HIST`, `CLAHE`, `CONTRAST_BALANCING`, `CONTRAST_EXTENSION`

**enum class COLOR_MODEL** — цветовые модели:
- `CM_RGB`, `CM_HSV`, `CM_LAB`, `CM_YCBCR`

**enum class METOD_IMAGE_PERPROCESSIN** — методы предобработки изображений:
- `NONE`, `CONVERTING_BGR_TO_GRAY`, ... *(см. исходный код для полного списка)*

**enum class METOD_DISPARITY** — методы расчёта карты диспаратности:
- `MODE_NONE`, `MODE_BM`, `MODE_SGBM`, `MODE_SGBM_3WAY`, `MODE_HH`, `MODE_HH4`

**enum class AUGMENTATION_METHOD** — методы аугментации изображений:
- `FLIP_HORIZONTAL`, `FLIP_VERTICAL`, `ROTATE_IMAGE_90`, и др.

**enum class DISPARITY_TYPE** — типы выходных данных по диспаратности:
- `ALL`, `BASIC_DISPARITY`, `BASIC_HEATMAP`, и др.


Структуры:
~~~~~~~~~~

**cameraParameters** — параметры одной камеры

**cameraStereoParameters** — параметры стерео-камеры

**pointsData** — структура данных для хранения облака точек и их проекций

**parameters3dSceene** — параметры построения и отображения 3D сцены

**CalibrationParametersMono / Stereo** — результат калибровки камеры

**CalibrationConfig** — параметры для загрузки и настройки калибровки

**trainTricks** — параметры, влияющие на поведение нейросетевого обучения

**settingsMetodDisparity** — параметры алгоритма расчёта карты диспаратности