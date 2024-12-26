import cv2
import numpy as np
from tqdm import tqdm


def process_video(video_path, path_mask, output_video, time0=0, time1=None, alpha=0.05, threshold=1.5):
    """
    1) Получает на вход path_mask (одноканальный numpy-массив, где 0=фон, 1..N=пути).
    2) Считывает видео, для каждого кадра (между time0 и time1) считает оптический поток (Farneback).
    3) Для каждого пути (индекс маски != 0) собирает векторы потока, сглаживает их по времени.
    4) Рисует стрелку и/или текст на кадре (показывая сглаженное направление и скорость потока).
    5) Записывает обработанное видео в output_video.
    6) Возвращает путь к результирующему файлу.
    """

    # === Шаг 0: Инициализация видео ===
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Не удалось открыть видео:", video_path)
        return None

    # Общее число кадров и FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Вычисляем количество кадров для обработки
    start_frame = int(fps * time0)  # Начальный кадр
    end_frame = total_frames if time1 is None else int(fps * time1)  # Конечный кадр

    if start_frame >= total_frames:
        print("Время начала обработки выходит за пределы видео.")
        cap.release()
        return None
    if end_frame > total_frames:
        end_frame = total_frames  # Ограничиваем максимальным числом кадров

    print(f"Обработка с {time0} до {time1 if time1 is not None else (total_frames / fps):.2f} секунд.")
    
    # Проверим, совпадают ли размеры video (w,h) и path_mask
    mask_h, mask_w = path_mask.shape[:2]
    if (mask_w != vid_w) or (mask_h != vid_h):
        print("Предупреждение: размеры маски и видео не совпадают! Масштабируем маску...")
        path_mask = cv2.resize(path_mask, (vid_w, vid_h), interpolation=cv2.INTER_NEAREST)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (vid_w, vid_h))

    # Считываем первый кадр
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        print("Видео пустое или ошибка чтения.")
        cap.release()
        out.release()
        return None

    # Переводим в grayscale для оптического потока
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Параметры оптического потока Farneback
    flow_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=40,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # Собираем список ID путей (кроме 0 - фон)
    path_ids = sorted(set(path_mask.flatten()) - {0})
    print("Обнаруженные пути в маске:", path_ids)

    # === Сглаживание: сохраняем предыдущие значения dx, dy для каждого пути ===
    smoothed_flow = {pid: {"dx": 0, "dy": 0} for pid in path_ids}

    # === Основной цикл (только кадры между time0 и time1) ===
    for _ in tqdm(range(start_frame, end_frame - 1), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break  # дошли до конца (меньше кадров, чем ожидалось?)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)
        output_frame = frame.copy()

        # === Для каждого пути считаем среднее dx, dy и сглаживаем ===
        path_flows = {}
        for pid in path_ids:
            region_mask = (path_mask == pid)

            dx_region = flow[..., 0][region_mask]
            dy_region = flow[..., 1][region_mask]

            if len(dx_region) < 10:
                # Слишком мало пикселей, пропускаем
                continue

            mean_dx = np.mean(dx_region)
            mean_dy = np.mean(dy_region)

            # Сглаживаем с предыдущими значениями
            smoothed_dx = alpha * mean_dx + (1 - alpha) * smoothed_flow[pid]["dx"]
            smoothed_dy = alpha * mean_dy + (1 - alpha) * smoothed_flow[pid]["dy"]
            smoothed_flow[pid]["dx"] = smoothed_dx
            smoothed_flow[pid]["dy"] = smoothed_dy

            # Рассчитываем модуль скорости
            mag = np.sqrt(smoothed_dx**2 + smoothed_dy**2)
            path_flows[pid] = mag

            # Находим boundingRect области, чтобы где-то рисовать
            region_uint8 = region_mask.astype(np.uint8)
            x, y, w_rect, h_rect = cv2.boundingRect(region_uint8)
            cx = x + w_rect // 2
            cy = y + h_rect // 2

            # Рисуем стрелку
            tip_x = int(cx + smoothed_dx * 10)
            tip_y = int(cy + smoothed_dy * 10)
            color = (0, 255, 0)  # зелёный
            cv2.arrowedLine(output_frame, (cx, cy), (tip_x, tip_y), color, 2, tipLength=0.3)

            # Текст
            text_str = f"Path {pid}: dx={smoothed_dx:.1f}, dy={smoothed_dy:.1f}, mag={mag:.1f}"
            cv2.putText(output_frame, text_str, (x, max(y-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # === Логика определения пути ===
        detected_path = None
        for pid in path_ids:
            if path_flows.get(pid, 0) > threshold:
                detected_path = pid
                break  # Прерываем, чтобы отдать приоритет ближнему пути

        if detected_path:
            text_str = f"Train detected on Path {detected_path}"
            cv2.putText(output_frame, text_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            
        # Записываем кадр
        out.write(output_frame)

        # Подготовка к след. кадру
        prev_gray = gray

    cap.release()
    out.release()
    return output_video
