# Поиск направления движения поезда и определение пути

## Описание

Задача состоит в определении направления движения поезда на видео и в определении, по какому пути (ближнему или дальнему от камеры) он движется. Направление движения поезда отображается стрелками, а сами пути обозначаются цифрами 1 и 2. Для этого используются методы компьютерного зрения:

1. **Преобразование Хаффа** для обнаружения рельс на видео.
2. **Оптический поток** для отслеживания движения поезда и определения его направления.

Проект реализован на Python, и его можно запустить как локально, так и в Docker.

## Решение

1. **Обнаружение рельс с использованием преобразования Хаффа**:
   - Мы применяем преобразование Хаффа для поиска линий, которые соответствуют рельсам на изображении. Это позволяет определить местоположение рельс на видео.

2. **Оптический поток для определения движения поезда**:
   - Используем метод оптического потока (Farneback) для отслеживания движения поезда на каждом кадре. Направление и скорость движения отображаются стрелками на видео.

### Запуск проекта

#### Локальный запуск (в терминале)

Для того чтобы запустить проект локально, выполните следующие шаги:

1. Убедитесь, что у вас установлен Python 3.7 или выше.
2. Установите необходимые зависимости:
    ```bash
    pip install -r requirements.txt
    ```
3. Для обработки видео с рельсами и поезда:
    ```bash
    python3 src/main.py data/raw/near.mp4 data/raw/empty_rails.png data/finished/output.mp4
    ```

4. Для запуска в веб-интерфейсе Gradio:
    ```bash
    python3 src/gradio_app.py
    ```
    После этого откроется интерфейс Gradio, где вы можете загрузить видео и маску рельс для анализа.

#### Запуск в Docker

1. **Создание образа Docker:**

    В каталоге проекта создайте Docker образ с помощью следующей команды:
    ```bash
    docker build -t tech-trans-q .
    ```
    или запульте сразу образ
    ```bash
    docker pull dimkablin/tech-trans-q
    ```

2. **Запуск контейнера Docker:**

    После того как образ будет собран, запустите контейнер с проектом:
    ```bash
    docker run -p 7860:7860 tech-trans-q
    ```

    Этот запуск откроет веб-интерфейс на порту 7860, который можно использовать для загрузки видео.


### Инструкция по запуску

<img src="./assets/gradio_example.jpg" alt="Project Logo" width="100%"/>

1. Сначала загрузите видео с поездом
2. Потом выберите кадр без поезда.
3. Опционально можно найти рельсы на этом кадре
4. Нажать кнопку "Обработать" для обработки видео.

Теперь осталось подождать, пока оптический поток подсчитается для выбранного участка видео.

### Структура проекта

```
project/
├── src/
│   ├── main.py              # Основной скрипт для обработки видео
│   ├── rails_detection.py   # Скрипт для детекции рельс
│   ├── video_processing.py  # Скрипт для визуализации решения
│   └── gradio_app.py        # Скрипт для запуска веб-интерфейса Gradio
├── data/
│   ├── raw/
│   │   ├── near.mp4         # Видео с рельсами и поездом
│   │   └── empty_rails.png  # Маска рельс
│   └── finished/
│       └── output.mp4       # Обработанное видео
├── Dockerfile               # Docker файл для сборки образа
├── requirements.txt         # Список зависимостей
└── README.md                # Этот файл
```

## Зависимости

Для корректной работы проекта требуется установить следующие зависимости:

- OpenCV для обработки видео и вычисления оптического потока.
- Gradio для создания интерфейса.
- numpy для работы с массивами данных.
- tqdm для отображения прогресса выполнения.

Все зависимости перечислены в файле `requirements.txt`.
