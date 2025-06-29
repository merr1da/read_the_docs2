Демонстрационные примеры на Python
==================================

Описание скрипта objcourse.py: определение курса на объект
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``objcourse.py`` демонстрирует использование класса ``ObjCourse`` для подсчета количества объектов интереса на изображении и расчета углового курса на объект. Этот класс предназначен для задач, связанных с обнаружением объектов и навигацией роботизированных систем, таких как подводные или надводные аппараты.

Класс ``ObjCourse`` инициализируется с путями к файлу модели (в формате ``ONNX``) и файлу с именами классов, определяющими, какие объекты могут быть обнаружены. Метод ``get_object_count`` принимает изображение в формате ``cv::Mat`` и возвращает количество обнаруженных объектов. Метод ``get_object_course`` вычисляет угловую поправку курса на объект с учетом горизонтального разрешения камеры (в пикселях) и угла обзора камеры (в градусах). Если на изображении обнаружено несколько объектов, курс рассчитывается на объект с максимальной площадью.

Методы используют предварительно обученную модель (например, основанную на архитектуре ``YOLOv5``), что обеспечивает высокую точность обнаружения. Логирование в коде фиксирует пути к файлам и результаты работы методов, что упрощает отладку и контроль выполнения.

Эта функциональность применима в задачах поиска объектов (например, кораблей) и автоматического управления роботизированными аппаратами для сопровождения цели.

Пример использования

Пример демонстрирует создание экземпляра класса ``ObjCourse``, загрузку изображения, подсчет объектов и расчет курса:

.. code-block:: python

    import cv2
    from pathlib import Path
    import logging
    from mrcv import ObjCourse

    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Пути к файлам
    model_file = Path("files/ship.onnx")
    class_file = Path("files/ship.names")
    ship_file = Path("files/ship.bmp")

    current_path = Path.cwd()
    model_path = current_path / model_file
    class_path = current_path / class_file
    ship_path = current_path / ship_file

    # Создание экземпляра класса
    objcourse = ObjCourse(str(model_path), str(class_path))

    # Загрузка изображения
    frame_ship = cv2.imread(str(ship_path), cv2.IMREAD_COLOR)
    if frame_ship is None:
        logger.error(f"Failed to load image: {ship_path}")

    # Подсчет объектов
    obj_count = objcourse.get_object_count(frame_ship)

    # Расчет курса
    obj_angle = objcourse.get_object_course(frame_ship, 640, 80)

    # Вывод результатов
    logger.info(f"Файл модели: {model_path}")
    logger.info(f"Файл классов: {class_path}")
    logger.info(f"Входное изображение: {ship_path}")
    logger.info(f"Обнаружено объектов: {obj_count}")
    logger.info(f"Курс на цель в градусах: {obj_angle}")

Пример вывода

.. code-block:: text

    2024-10-01 10:00:00 - Файл модели: /path/to/files/ship.onnx
    2024-10-01 10:00:00 - Файл классов: /path/to/files/ship.names
    2024-10-01 10:00:00 - Входное изображение: /path/to/files/ship.bmp
    2024-10-01 10:00:00 - Обнаружено объектов: 1
    2024-10-01 10:00:00 - Курс на цель в градусах: 17.0

В данном случае обнаружен один объект, а курсовая невязка составила 17 градусов вправо относительно центра кадра.

Описание скрипта ``yolov5.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Предоставляет утилиты для работы с моделями YOLOv5 — популярной архитектуры для обнаружения объектов, отличающейся высокой скоростью и точностью. Эти функции предназначены для подготовки данных и конфигураций, необходимых для обучения или использования моделей YOLOv5.

- ``yolov5_labeler_processing``: обрабатывает изображения или аннотации, принимая пути к входному и выходному каталогам. Используется для подготовки данных в формате, совместимом с YOLOv5.
- ``yolov5_generate_config``: генерирует конфигурационный файл (например, ``config.yaml``), указывая тип модели (например, ``YOLOv5s``), путь к файлу и количество классов.
- ``yolov5_generate_hyperparameters``: создаёт файл гиперпараметров (например, ``hyperparameters.yaml``), определяя параметры обучения, такие как размер изображения и другие настройки.

Эти функции упрощают настройку моделей YOLOv5 для специфических задач обнаружения объектов, таких как классификация кораблей или других объектов интереса.

Пример использования

Пример показывает, как использовать утилиты для подготовки данных и конфигураций:

.. code-block:: python

    from mrcv import (
        yolov5_labeler_processing, 
        yolov5_generate_config, 
        yolov5_generate_hyperparameters, 
        YOLOv5Model
    )

    # Обработка данных для маркировки
    yolov5_labeler_processing("path/to/input/dir", "path/to/output/dir")

    # Генерация конфигурационного файла
    yolov5_generate_config(YOLOv5Model.YOLOv5s, "config.yaml", 10)

    # Генерация файла гиперпараметров
    yolov5_generate_hyperparameters(YOLOv5Model.YOLOv5s, 640, 640, "hyperparameters.yaml", 10)

В этом примере:

- Обрабатываются данные из входного каталога и сохраняются в выходной.
- Создаётся конфигурация для модели YOLOv5s с 10 классами.
- Генерируются гиперпараметры для обучения с разрешением изображения 640×640.

Описание Скрипта ``roi.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``roi.py`` реализует систему предсказания движения объекта и оптимизации области интереса (ROI) для эффективного отслеживания. Используются классы ``Predictor`` и ``Optimizer``.

- Класс ``Predictor``: обучается на основе LSTM-сети для предсказания следующей позиции объекта по его предыдущим координатам. Метод ``train_lstm_net`` обучает модель, а ``predict_next_coordinate`` возвращает прогнозируемую позицию.
- Класс ``Optimizer``: оптимизирует размер ROI, анализируя движение объекта и отклонение предсказаний, чтобы минимизировать вычислительные затраты, сохраняя объект в кадре.

Код симулирует движение объекта, генерируя координаты, обучает предиктор, а затем в реальном времени предсказывает позиции и корректирует ROI. Это полезно для систем слежения, таких как видеонаблюдение или автономное вождение.


Пример использования


Ниже приведён пример, демонстрирующий обучение предиктора и оптимизацию ROI:

.. code-block:: python

    import cv2
    import numpy as np
    from mrcv import Predictor, Optimizer, generate_coordinates, extract_roi, to_point

    # Инициализация параметров
    img_size = (1440, 1080)
    predictor_train_points_num = 50
    object_size = 100
    hidden_size = 20
    layers_num = 1

    predictor = Predictor(hidden_size, layers_num, predictor_train_points_num, img_size, 200)
    optimizer = Optimizer(1000, 50000)

    # Генерация координат для обучения
    coordinates = [generate_coordinates(i, 1, 300, 100, img_size) for i in range(1, predictor_train_points_num + 1)]

    # Обучение предиктора
    predictor.train_lstm_net(coordinates)

    # Симуляция и визуализация ROI
    img_r = np.full((img_size[1], img_size[0], 3), 255, dtype=np.uint8)
    real_coordinate = coordinates[-1]
    predicted_coordinate = predictor.predict_next_coordinate()
    roi_size = optimizer.optimize_roi_size(
        real_coordinate, 
        predicted_coordinate, 
        object_size, 
        predictor.get_moving_average_deviation() / 2
    )
    roi = extract_roi(img_r, to_point(predicted_coordinate), (int(roi_size), int(roi_size)))

    cv2.imshow("ROI Example translating to Пример ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

В данном примере:

- Обучается предиктор для предсказания координат объекта.
- Предсказывается следующая позиция объекта.
- Оптимизируется размер области интереса (ROI).
- Визуализируется полученная область на изображении.

Описание для скрипта ``vae.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``vae.py`` демонстрирует использование вариационного автоэнкодера (VAE) для генерации синтетических изображений и их полуавтоматической маркировки. VAE — это генеративная модель, создающая данные, подобные обучающему набору.

- ``neural_network_augmentation_as_mat``: генерирует аугментированное изображение на основе входных данных.
- ``semi_automatic_labeler_image``: использует модель (например, ONNX) для автоматической маркировки сгенерированного изображения.

Этот подход расширяет набор данных для задач машинного обучения, где исходных данных недостаточно, например, для обнаружения кораблей.


Пример использования


Ниже приведён пример генерации и маркировки изображения:

.. code-block:: python

    import os
    import cv2
    from mrcv import neural_network_augmentation_as_mat, semi_automatic_labeler_image

    images_path = "vae/files/images"
    result_path = "vae/files/result"
    model_path = "vae/files/ship.onnx"
    class_path = "vae/files/ship.names"

    height = 640
    width = 640

    # Генерация изображения
    genImage = neural_network_augmentation_as_mat(images_path, height, width, 200, 2, 2, 16, 3E-4)

    # Преобразование и сохранение
    colorGenImage = cv2.cvtColor(genImage, cv2.COLOR_GRAY2BGR)
    output_path = os.path.join(result_path, "generated.jpg")
    cv2.imwrite(output_path, colorGenImage)

    # Маркировка
    semi_automatic_labeler_image(colorGenImage, 640, 640, result_path, model_path, class_path)

В этом примере:

- Генерируется изображение размером 640×640.
- Преобразуется в цветное изображение и сохраняется.
- Выполняется полуавтоматическая маркировка с использованием модели.

Описание для скрипта ``segmentation.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``segmentation.py`` реализует сегментацию изображений с использованием класса ``Segmentor``. Сегментация разделяет изображение на регионы, выделяя объекты (например, корабли) от фона.

Класс ``Segmentor`` инициализируется с параметрами: количеством классов, размерами изображения, именами классов, архитектурой и путями к весам. Метод ``predict`` выполняет сегментацию для указанного класса (например, ``"ship"``).

Эта функциональность важна для задач локализации объектов и анализа сцены, где требуется пиксельная точность.

Пример использования


Ниже приведён пример сегментации изображения:

.. code-block:: python

    from mrcv import Segmentor
    import cv2

    image_path = "segmentation/file/images/test/43.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        exit(1)

    # Создание и инициализация сегментатора
    segmentor = Segmentor()
    segmentor.initialize(-1, 512, 320, ["background", "ship"], "resnet34", "segmentation/file/weights/resnet34.pt")
    segmentor.load_weight("segmentation/file/weights/segmentor.pt")

    # Выполнение сегментации
    segmentor.predict(image, "ship")

В этом примере:

- Загружается изображение.
- Инициализируется сегментатор с указанными параметрами.
- Выполняется предсказание маски для класса ``"ship"``.

Описание для скрипта ``clustering.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``clustering.py`` демонстрирует использование класса ``DenseStereo`` для выполнения кластеризации данных, загруженных из файла. Кластеризация — это процесс группировки данных по сходству, что полезно для анализа больших наборов данных, например, в задачах обработки изображений или анализа сцен.

Класс ``DenseStereo`` инициализируется, после чего данные загружаются из указанного файла с помощью метода ``load_data_from_file``. Затем метод ``make_clustering`` выполняет кластеризацию загруженных данных.

Эта функциональность применима в задачах анализа стереоизображений, таких как группировка пикселей по диспаратности или другим характеристикам.

Пример использования


Ниже приведён пример загрузки данных и выполнения кластеризации:

.. code-block:: python

    from mrcv import DenseStereo
    import logging

    data_path = "files/claster.dat"

    def main():
        """
        Основная функция для выполнения кластеризации.
        """
        dense_stereo = DenseStereo()  # Замените на актуальную инициализацию
        dense_stereo.load_data_from_file(data_path)
        dense_stereo.make_clustering()

    if __name__ == '__main__':
        from multiprocessing import freeze_support
        freeze_support()  # Для поддержки замороженных исполняемых файлов
        main()

В этом примере создаётся экземпляр класса ``DenseStereo``, данные загружаются из файла ``files/claster.dat``, и выполняется кластеризация с использованием метода ``make_clustering``.

Описание для скрипта ``disparity.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``disparity.py`` демонстрирует построение карты диспаратности для стереопары изображений с использованием функции ``disparity_map``.  
Карта диспаратности — это изображение, в котором каждый пиксель отражает разницу в положении соответствующего пикселя между левым и правым стереоизображениями, что позволяет оценить глубину сцены.

Функция ``disparity_map`` принимает левое и правое стереоизображения, параметры алгоритма (например, минимальную и максимальную диспаратность, размер блока), тип диспаратности и цветовую карту для визуализации. В примере используется тип ``DisparityType.ALL`` для генерации всех доступных типов карт диспаратности.

Эта функциональность полезна в задачах стереозрения, таких как реконструкция 3D-сцен или оценка глубины.

Пример использования


Ниже пример загрузки стереопары, вычисления карты диспаратности и её отображения:

.. code-block:: python

    import cv2
    import os
    import numpy as np
    from enum import Enum
    from mrcv import disparity_map

    # Определение перечисления для типов диспаратности
    class DisparityType(Enum):
        BASIC_DISPARITY = 0
        BASIC_HEATMAP = 1
        FILTERED_DISPARITY = 2
        FILTERED_HEATMAP = 3
        ALL = 4

    # Пути к изображениям
    file_image_left = os.path.join("files", "example_left.jpg")
    file_image_right = os.path.join("files", "example_right.jpg")

    current_path = os.getcwd()
    path_image_left = os.path.join(current_path, file_image_left)
    path_image_right = os.path.join(current_path, file_image_right)

    # Загрузка изображений
    image_left = cv2.imread(path_image_left, cv2.IMREAD_COLOR)
    image_right = cv2.imread(path_image_right, cv2.IMREAD_COLOR)

    # Параметры функции
    min_disparity = 16
    num_disparities = 16 * 10
    block_size = 15
    lambda_val = 5000.0
    sigma = 3
    color_map = cv2.COLORMAP_TURBO
    disparity_type = DisparityType.ALL

    # Проверка загрузки изображений
    if image_left is None or image_right is None:
        print("Ошибка: Не удалось загрузить одно или оба изображения")
        exit(1)

    # Создание пустого массива для результата
    disparity_map_output = np.zeros_like(image_left)

    # Построение карты диспаратности
    disparity_map(disparity_map_output, image_left, image_right, min_disparity, num_disparities,
                  block_size, lambda_val, sigma, disparity_type, color_map, True, True)

    # Отображение результата
    cv2.namedWindow("MRCV Disparity Map translating to Карта диспаратности MRCV", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("MRCV Disparity Map translating to Карта диспаратности MRCV", disparity_map_output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

В этом примере загружаются левое и правое стереоизображения, вычисляется карта диспаратности с заданными параметрами, и результат отображается.

Описание для скрипта ``3dscene.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``3dscene.py`` демонстрирует использование функции ``find_3d_points_in_objects_segments`` для определения 3D-точек в сегментах объектов на стереопаре изображений. Эта функция выполняет реконструкцию 3D-координат точек, принадлежащих объектам интереса, что полезно для задач 3D-реконструкции и анализа сцен.

Пример загружает стереопару изображений и параметры камеры, обрабатывает их для получения 3D-точек и выводит результаты, включая изображение с центрами 3D-сегментов, проекцию 3D-сцены и маски объектов.

Эта функциональность применима в задачах автономной навигации, робототехники и компьютерного зрения.

Пример использования

Пример демонстрирует загрузку данных, обработку и отображение результатов:

.. code-block:: python

    import cv2
    from mrcv import read_camera_stereo_parameters_from_file, \
        write_log, \
        METOD_DISPARITY, \
        find_3d_points_in_objects_segments, \
        show_image, \
        save_in_file_3d_points_in_objects_segments, \
        show_disparity_map, \
        converting_undistort_rectify, \
        making_stereo_pair

    def main():
        write_log("=== НОВЫЙ ЗАПУСК ===")

        # Загрузка изображений
        input_image_camera01 = cv2.imread("./files/L1000.bmp")
        input_image_camera02 = cv2.imread("./files/R1000.bmp")
        if input_image_camera01 is None or input_image_camera02 is None:
            write_log("Не удалось загрузить изображения", "ERROR")
            return
        write_log("1. Загрузка изображений из файла (успешно)")

        # Загрузка параметров камеры
        camera_parameters, state = read_camera_stereo_parameters_from_file(
            "./files/(66a)_(960p)_NewCamStereoModule_Air.xml")
        if state == 0:
            write_log("2. Загрузка параметров стереокамеры из файла (успешно)")

        # Инициализация параметров
        settings_metod_disparity = {'metodDisparity': METOD_DISPARITY.MODE_SGBM}
        limit_out_points = 8000
        limits_outlier_area = [-4.0e3, -4.0e3, 450, 4.0e3, 4.0e3, 3.0e3]
        file_path_model_yolo_neural_net = "./files/NeuralNet/yolov5n-seg.onnx"
        file_path_classes = "./files/NeuralNet/yolov5.names"
        parameters_3d_scene = {
            'angX': 25, 'angY': 45, 'angZ': 35,
            'tX': -200, 'tY': 200, 'tZ': -600,
            'dZ': -1000, 'scale': 1.0
        }

        # Обработка изображений
        output_image, output_image_3d_scene, points_3d, reply_masks, disparity_map, state = find_3d_points_in_objects_segments(
            input_image_camera01, input_image_camera02, camera_parameters,
            settings_metod_disparity, limit_out_points, limits_outlier_area,
            file_path_model_yolo_neural_net, file_path_classes, parameters_3d_scene
        )

        # Отображение результатов
        foto_experimental_stand = cv2.imread("./files/experimantalStand.jpg")
        show_image(foto_experimental_stand, "fotoExperimantalStand translating to Фото экспериментального стенда")

        output_stereo_pair, state = making_stereo_pair(input_image_camera01, input_image_camera02)
        if state == 0:
            show_image(output_stereo_pair, "SourceStereoImage translating to Исходное стереоизображение")
            write_log("4.2 Отображение исходного изображения (успешно)")

        input_image_camera01_remap, _ = converting_undistort_rectify(
            input_image_camera01, camera_parameters['map11'], camera_parameters['map12'])
        input_image_camera02_remap, _ = converting_undistort_rectify(
            input_image_camera02, camera_parameters['map21'], camera_parameters['map22'])
        output_stereo_pair_remap, state = making_stereo_pair(input_image_camera01_remap, input_image_camera02_remap)
        show_image(output_stereo_pair_remap, "outputStereoPairRemap translating to Исправленная стереопара")

        show_disparity_map(disparity_map, "disparityMap translating to Карта диспаратности")

        for qs, mask in enumerate(reply_masks):
            show_image(mask, f"replyMasks {qs} translating to Маски ответа {qs}", 0.5)
        write_log("4.5 Отображение бинарных изображений сегментов (успешно)")

        path_to_file = "./files/3DPointsInObjectsSegments.txt"
        state = save_in_file_3d_points_in_objects_segments(points_3d, path_to_file)
        if state == 0:
            write_log("4.6 Сохранение 3D-точек в текстовый файл (успешно)")

        show_image(output_image, "outputImage translating to Выходное изображение", 1.0)
        show_image(output_image_3d_scene, "outputImage3dSceene translating to Проекция 3D-сцены", 1.0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

Описание
В этом примере загружаются стереопара изображений и параметры камеры, выполняется обработка для получения 3D-точек, и результаты визуализируются с использованием различных функций отображения.

Описание для скрипта ``imgpreprocessing.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``imgpreprocessing.py`` демонстрирует использование класса ``MRCV`` для предварительной обработки изображений с применением различных методов, таких как фильтрация шума, балансировка контраста и повышение резкости. Предобработка изображений — важный этап в задачах компьютерного зрения для улучшения качества изображений перед анализом.

Пример загружает изображение, определяет список методов предобработки и применяет их с помощью метода ``preprocessingImage``. Результат сохраняется в файл и отображается.

Эта функциональность полезна для подготовки изображений к последующему анализу, например, для обнаружения объектов или сегментации.

Пример использования

Пример показывает загрузку изображения, применение предобработки и сохранение результата:

.. code-block:: python

    import cv2
    import numpy as np
    import os
    from mrcv import MRCV, METOD_IMAGE_PERPROCESSIN, LOGTYPE

    # Создание выходной директории
    os.makedirs("./files/outImages", exist_ok=True)

    # Логирование уже настроено в mrcv.py
    MRCV.writeLog(" ")
    MRCV.writeLog(" === НОВЫЙ ЗАПУСК === ")

    # Загрузка изображения
    imageInputFilePath = "./files/img02.jfif"
    imageIn = cv2.imread(imageInputFilePath, cv2.IMREAD_COLOR)
    imageOut = imageIn.copy() if imageIn is not None else None

    if imageIn is not None:
        MRCV.writeLog(f"    загружено изображение: {imageInputFilePath} :: "
                      f"{imageIn.shape[1]}x{imageIn.shape[0]}x{imageIn.shape[2]}")
    else:
        MRCV.writeLog(f"    не удалось загрузить изображение: {imageInputFilePath}", LOGTYPE.ERROR)
        exit(1)

    # Определение методов предобработки
    methodImagePreProcessingBrightnessContrast = [
        METOD_IMAGE_PERPROCESSIN.NOISE_FILTERING_01_MEDIAN_FILTER,
        METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_10_LAB_CLAHE,
        METOD_IMAGE_PERPROCESSIN.SHARPENING_02,
        METOD_IMAGE_PERPROCESSIN.BRIGHTNESS_LEVEL_DOWN,
        METOD_IMAGE_PERPROCESSIN.NONE,
    ]

    # Предобработка изображения
    state, imageOut = MRCV.preprocessingImage(
        imageOut, methodImagePreProcessingBrightnessContrast, "./files/camera-parameters.xml")
    if state == 0:
        MRCV.writeLog(" Предобработка изображения завершена (успешно)")
    else:
        MRCV.writeLog(f" preprocessingImage, state = {state}", LOGTYPE.ERROR)

    # Сохранение выходного изображения
    imageOutputFilePath = "./files/outImages/test.png"
    cv2.imwrite(imageOutputFilePath, imageOut)
    MRCV.writeLog(f"\t результат предобработки сохранён: {imageOutputFilePath}")

    # Отображение изображений
    CoefShowWindow = 0.5
    imageIn = cv2.resize(imageIn, None, fx=CoefShowWindow, fy=CoefShowWindow, interpolation=cv2.INTER_LINEAR)
    imageOut = cv2.resize(imageOut, None, fx=CoefShowWindow, fy=CoefShowWindow, interpolation=cv2.INTER_LINEAR)

    cv2.namedWindow("imageIn translating to Входное изображение", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("imageIn translating to Входное изображение", imageIn)
    cv2.namedWindow("imageOut translating to Выходное изображение", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("imageOut translating to Выходное изображение", imageOut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Описание
В этом примере загружается изображение, выполняется предобработка с использованием указанных методов, а результат сохраняется и отображается.

Описание для скрипта ``comparing.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``comparing.py`` демонстрирует использование функции ``compare_images`` для сравнения двух изображений и вычисления их сходства. Эта функция полезна для задач, таких как обнаружение дубликатов изображений, контроль качества или анализ изменений.

Функция ``compare_images`` принимает два изображения и возвращает показатель сходства, указывающий, насколько они похожи.

Пример использования


Пример показывает загрузку двух изображений и вычисление их сходства:

.. code-block:: python

    import os
    import cv2
    from mrcv import compare_images

    # Путь к папке с изображениями
    image_dir = "files"
    current_path = os.getcwd()
    image_path = os.path.join(current_path, image_dir)

    # Загрузка изображений
    img1 = cv2.imread(os.path.join(image_path, "1.png"))
    img2 = cv2.imread(os.path.join(image_path, "2.png"))

    # Проверка, что изображения загружены
    if img1 is None or img2 is None:
        print("Ошибка: Не удалось загрузить одно или оба изображения.")

    # Сравнение изображений
    similarity = compare_images(img1, img2, True)
    print(f"Сходство: {similarity}")

Описание

В этом примере загружаются два изображения, и их сходство вычисляется и выводится с использованием функции ``compare_images``.

Описание для скрипта ``detectorautotrain.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``detectorautotrain.py`` демонстрирует использование класса ``Detector`` для автоматического обучения детектора объектов. Этот класс позволяет инициализировать детектор с определёнными параметрами и запускать процесс обучения на предоставленном наборе данных.

Метод ``auto_train`` принимает путь к набору данных, расширение файлов изображений, списки эпох, размеров пакета и скоростей обучения, а также пути к предварительно обученной модели и месту сохранения обученной модели. Это автоматизирует обучение с различными гиперпараметрами.

Данная функциональность полезна для задач обнаружения объектов, таких как идентификация кораблей или других объектов интереса.

Пример использования


Пример показывает инициализацию детектора и запуск автоматического обучения:

.. code-block:: python

    from mrcv import Detector

    voc_classes_path = "files/onwater/voc_classes.txt"
    dataset_path = "files/onwater/"
    pretrained_model_path = "files/onwater_autodetector.pt"
    model_save_path = "files/yolo4_tiny.pt"

    detector = Detector()
    detector.initialize(416, 416, voc_classes_path)
    detector.auto_train(
        dataset_path,
        ".jpg",
        epochs_list=[10, 15, 30],
        batch_sizes=[4, 8],
        learning_rates=[0.001, 1e-4],
        pretrained_path=pretrained_model_path,
        save_path=model_save_path
    )

Описание

В этом примере создаётся экземпляр класса ``Detector``, инициализируется с параметрами, и запускается автоматическое обучение на наборе данных с различными гиперпараметрами.

Описание для скрипта ``augmentation.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``augmentation.py`` демонстрирует использование функций ``augmentation`` и ``batch_augmentation`` для применения различных методов аугментации к изображениям. 

Аугментация изображений — это метод увеличения размера набора данных путём создания изменённых копий изображений, что улучшает обобщающую способность моделей машинного обучения.

Функция ``augmentation`` принимает список изображений и методы аугментации, возвращая аугментированные изображения. Функция ``batch_augmentation`` выполняет аугментацию в пакетном режиме с конфигурацией, указывающей, сохранять ли оригиналы, общее количество выходных изображений и веса для различных методов.

Данная функциональность полезна для подготовки данных в задачах компьютерного зрения, таких как обучение детекторов объектов.

Пример использования


Пример показывает применение аугментации к изображениям:

.. code-block:: python

    import os
    import cv2
    from mrcv import AugmentationMethod, BatchAugmentationConfig, augmentation, batch_augmentation

    input_images = []
    for i in range(10):
        img_path = os.path.join("files", f"img{i}.jpg")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение по пути {img_path}")
        input_images.append(img)

    # Создание копии входных изображений
    input_images_copy = [img.copy() for img in input_images]

    # Определение методов аугментации
    augmentation_methods = [
        AugmentationMethod.ROTATE_IMAGE_90,
        AugmentationMethod.FLIP_HORIZONTAL,
        AugmentationMethod.FLIP_VERTICAL,
        AugmentationMethod.ROTATE_IMAGE_45,
        AugmentationMethod.ROTATE_IMAGE_315,
        AugmentationMethod.ROTATE_IMAGE_270,
        AugmentationMethod.FLIP_HORIZONTAL_AND_VERTICAL,
    ]

    # Выполнение аугментации
    state, output_images = augmentation(input_images, augmentation_methods)
    if state != 0:
        print(f"Ошибка: Аугментация завершилась с кодом {state}")

    # Конфигурация пакетной аугментации
    config = BatchAugmentationConfig()
    config.keep_original = True
    config.total_output_count = 100
    config.random_seed = 42
    config.method_weights = {
        AugmentationMethod.FLIP_HORIZONTAL: 0.2,
        AugmentationMethod.ROTATE_IMAGE_90: 0.2,
        AugmentationMethod.BRIGHTNESS_CONTRAST_ADJUST: 0.3,
        AugmentationMethod.PERSPECTIVE_WARP: 0.2,
        AugmentationMethod.COLOR_JITTER: 0.1,
    }

    # Выполнение пакетной аугментации
    state, batch_output = batch_augmentation(
        input_images_copy,
        config,
        os.path.join("files", "batch_output")
    )
    if state != 0:
        print(f"Ошибка: Пакетная аугментация завершилась с кодом: {state}")

Описание

В этом примере загружаются изображения, применяются различные методы аугментации, и выполняется пакетная аугментация с заданной конфигурацией.

Описание для скрипта ``morphologyimage.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт ``morphologyimage.py`` демонстрирует использование функции ``morphology_image`` для применения морфологических операций к изображению.

Морфологические операции, такие как открытие, закрытие, эрозия и дилатация, используются для обработки бинарных или полутоновых изображений, помогая в сегментации, удалении шума и выделении структур.

Функция ``morphology_image`` принимает изображение, путь для сохранения результата, тип морфологической операции и размер ядра. В примере используется операция ``METOD_MORF.OPEN`` для удаления мелких объектов и шума.

Данная функциональность полезна для предобработки изображений перед анализом, например, для выделения объектов интереса.

Пример использования


Пример показывает применение морфологической операции к изображению:

.. code-block:: python

    import cv2
    from mrcv import morphology_image, METOD_MORF
    from pathlib import Path

    image_file = Path("files")
    current_path = Path.cwd()
    image_path = current_path / image_file
    input_image = str(image_path / "opening.png")
    output_image = str(image_path / "out.png")

    morph_size = 1

    # Чтение изображения в градациях серого
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

    # Проверка успешности загрузки изображения
    if image is None:
        print("Не удалось открыть или найти изображение")
    else:
        result = morphology_image(image, output_image, METOD_MORF.OPEN, morph_size)

Описание

В этом примере загружается полутоновое изображение, применяется морфологическая операция открытия с размером ядра 1, и результат сохраняется.
