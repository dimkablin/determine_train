import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from rails_detection import detect_rails
from video_processing import process_video
import fire

def run(input_video, empty_rails, output_video):
    """
    Основная функция для детектирования рельсов и обработки видео.
    
    :param input_video: Путь к исходному видео (например, "data/raw/near.mp4").
    :param empty_rails: Путь к изображению рельсов без поездов (например, "data/raw/empty_rails.png").
    :param output_video: Путь для сохранения обработанного видео (например, "data/processed/near_processed.mp4").
    """
    # Убедимся, что директория для выходного видео существует
    output_dir = os.path.dirname(output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Считываем изображение рельсов
    img_bgr = cv2.imread(empty_rails)
    if img_bgr is None:
        print(f"Не удалось загрузить изображение по пути: {empty_rails}")
        return
    
    # Детектируем рельсы
    final_image_bgr, mask, count = detect_rails(img_bgr)
    if final_image_bgr is None:
        print("Не удалось обнаружить рельсы на изображении.")
        return
    
    # Сохраняем итоговое изображение
    output_image_path = os.path.join(os.path.dirname(output_video), 'empty_rails_detected.png')
    cv2.imwrite(output_image_path, final_image_bgr)
    print(f"Итоговое изображение сохранено по пути: {output_image_path}")
    
    # Обрабатываем видео
    processed_video_path = process_video(input_video, mask, output_video)
    if processed_video_path:
        print(f"Обработанное видео сохранено по пути: {processed_video_path}")
    else:
        print("Произошла ошибка при обработке видео.")

if __name__ == "__main__":
    fire.Fire(run)
