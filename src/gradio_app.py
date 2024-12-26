import os
import gradio as gr
import cv2
import numpy as np
from rails_detection import detect_rails
from video_processing import process_video
import shutil

def extract_frame(video_path, frame_number):
    """Извлекает кадр из видео"""
    if not video_path:
        return None, "Загрузите видео"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Ошибка открытия видео"
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        frame_number = total_frames - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, "Ошибка чтения кадра"
        
    return frame, f"Кадр {frame_number} из {total_frames}"

def detect_and_process(video_path, preview_image, time0, time1):
    # Создаём временные директории
    os.makedirs("temp/raw", exist_ok=True)
    os.makedirs("temp/processed", exist_ok=True)
    
    # Пути для файлов
    output_video_path = "temp/processed/processed_video.mp4"
    output_image_path = "temp/processed/detected_image.png"
    
    # Сохраняем кадр и детектируем рельсы
    out_bgr, mask, count = detect_rails(preview_image)
    if out_bgr is None:
        return None, None, "Не удалось обнаружить рельсы на кадре."
    
    # Сохраняем обработанное изображение
    cv2.imwrite(output_image_path, out_bgr)
    
    # Обработка видео
    processed_video_path = process_video(video_path, mask, output_video_path, time0, time1)
    if not processed_video_path:
        return output_image_path, None, "Видео не удалось обработать, но изображение готово."
    
    return output_image_path, processed_video_path, f"Обнаружено {count} пути(ей). Обработка завершена успешно."

def preview_frame(video_path, frame_number):
    """Предпросмотр кадра"""
    if video_path is None:
        return None, "Загрузите видео"
    
    frame, msg = extract_frame(video_path, frame_number)
    if frame is None:
        return None, msg
    
    # Конвертируем BGR в RGB для отображения
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def preview_rails(input_image):
    """Предпросмотр рельс"""
    out_bgr, _, _ = detect_rails(input_image)

    return out_bgr

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# Система детектирования рельсов и обработки видео")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Входные данные")
                input_video = gr.Video(label="Загрузите видео (near.mp4)", sources=["upload"])
                
                with gr.Row():
                    frame_number = gr.Number(label="Номер кадра", value=0, precision=0)
                    preview_btn = gr.Button("Показать кадр")
                
                preview_image = gr.Image(label="Предпросмотр кадра")
                detect_rail_btn = gr.Button("Найти рельсы")
            
            with gr.Column():
                gr.Markdown("## Результаты")
                output_image = gr.Image(label="Обнаруженные рельсы")
                output_video = gr.Video(label="Обработанное видео")
        
        with gr.Row():
            with gr.Column():
                time0 = gr.Number(label="Обрабатывать с t0 (сек).", value=20, precision=0)
                time1 = gr.Number(label="Обрабатывать до t1 (сек)", value=40, precision=0)
                process_btn = gr.Button("Обработать", variant="primary")
            with gr.Column():
                status_text = gr.Textbox(label="Статус", interactive=False)
        
        # Привязываем функции к кнопкам
        preview_btn.click(
            fn=preview_frame,
            inputs=[input_video, frame_number],
            outputs=[preview_image]
        )

        detect_rail_btn.click(
            fn=preview_rails,
            inputs=[preview_image],
            outputs=[output_image]
        )
        
        process_btn.click(
            fn=detect_and_process,
            inputs=[input_video, preview_image, time0, time1],
            outputs=[output_image, output_video, status_text]
        )
    
    return interface

def main():
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
