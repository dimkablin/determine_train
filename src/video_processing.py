import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int
    start_frame: int
    end_frame: int

@dataclass
class FlowVector:
    dx: float
    dy: float
    
    @property
    def magnitude(self) -> float:
        return np.sqrt(self.dx**2 + self.dy**2)

class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
    def get_metadata(self, time0: float = 0, time1: Optional[float] = None) -> Optional[VideoMetadata]:
        if not self.cap.isOpened():
            print("Не удалось открыть видео:", self.video_path)
            return None
            
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(fps * time0)
        end_frame = total_frames if time1 is None else int(fps * time1)
        
        if start_frame >= total_frames:
            print("Время начала обработки выходит за пределы видео.")
            return None
            
        if end_frame > total_frames:
            end_frame = total_frames
            
        return VideoMetadata(width, height, fps, total_frames, start_frame, end_frame)
        
    def set_frame_position(self, frame_number: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
        
    def release(self) -> None:
        self.cap.release()

class VideoWriter:
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.output_path = output_path
        
    def write_frame(self, frame: np.ndarray) -> None:
        self.writer.write(frame)
        
    def release(self) -> None:
        self.writer.release()

class OpticalFlowCalculator:
    def __init__(self):
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=40,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        
    def calculate(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **self.flow_params)

class PathFlowAnalyzer:
    def __init__(self, path_mask: np.ndarray, alpha: float = 0.05):
        self.path_mask = path_mask
        self.alpha = alpha
        self.path_ids = sorted(set(path_mask.flatten()) - {0})
        self.smoothed_flow = {pid: FlowVector(0, 0) for pid in self.path_ids}
        
    def analyze_flow(self, flow: np.ndarray) -> Dict[int, FlowVector]:
        path_flows = {}
        
        for pid in self.path_ids:
            region_mask = (self.path_mask == pid)
            
            dx_region = flow[..., 0][region_mask]
            dy_region = flow[..., 1][region_mask]
            
            if len(dx_region) < 10:
                continue
                
            mean_dx = np.mean(dx_region)
            mean_dy = np.mean(dy_region)
            
            # Сглаживание
            smoothed_dx = self.alpha * mean_dx + (1 - self.alpha) * self.smoothed_flow[pid].dx
            smoothed_dy = self.alpha * mean_dy + (1 - self.alpha) * self.smoothed_flow[pid].dy
            
            flow_vector = FlowVector(smoothed_dx, smoothed_dy)
            self.smoothed_flow[pid] = flow_vector
            path_flows[pid] = flow_vector
            
        return path_flows

class FrameVisualizer:
    def __init__(self, path_mask: np.ndarray):
        self.path_mask = path_mask
        
    def draw_flow_visualization(
        self,
        frame: np.ndarray,
        path_flows: Dict[int, FlowVector],
        detected_path: Optional[int] = None
    ) -> np.ndarray:
        output_frame = frame.copy()
        
        for pid, flow_vector in path_flows.items():
            region_mask = (self.path_mask == pid)
            region_uint8 = region_mask.astype(np.uint8)
            x, y, w_rect, h_rect = cv2.boundingRect(region_uint8)
            
            # Центр области
            cx = x + w_rect // 2
            cy = y + h_rect // 2
            
            # Рисуем стрелку
            tip_x = int(cx + flow_vector.dx * 10)
            tip_y = int(cy + flow_vector.dy * 10)
            cv2.arrowedLine(output_frame, (cx, cy), (tip_x, tip_y), (0, 255, 0), 2, tipLength=0.3)
            
            # Подписи значений
            text_str = f"Path {pid}: dx={flow_vector.dx:.1f}, dy={flow_vector.dy:.1f}, mag={flow_vector.magnitude:.1f}"
            cv2.putText(output_frame, text_str, (x, max(y-5, 0)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if detected_path is not None:
            text_str = f"Train detected on Path {detected_path}"
            cv2.putText(output_frame, text_str, (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        return output_frame

class TrainDetector:
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
        
    def detect_train(self, path_flows: Dict[int, FlowVector]) -> Optional[int]:
        for pid, flow_vector in path_flows.items():
            if flow_vector.magnitude > self.threshold:
                return pid
        return None

class VideoProcessor:
    def __init__(self, path_mask: np.ndarray, alpha: float = 0.05, threshold: float = 1.5):
        self.path_mask = path_mask
        self.flow_calculator = OpticalFlowCalculator()
        self.path_analyzer = PathFlowAnalyzer(path_mask, alpha)
        self.visualizer = FrameVisualizer(path_mask)
        self.train_detector = TrainDetector(threshold)
        
    def process_video(
        self,
        video_path: str,
        output_path: str,
        time0: float = 0,
        time1: Optional[float] = None
    ) -> Optional[str]:
        # Инициализация видео
        reader = VideoReader(video_path)
        metadata = reader.get_metadata(time0, time1)
        
        if metadata is None:
            reader.release()
            return None
            
        # Проверка размеров маски
        if (self.path_mask.shape[1] != metadata.width) or (self.path_mask.shape[0] != metadata.height):
            print("Предупреждение: размеры маски и видео не совпадают! Масштабируем маску...")
            self.path_mask = cv2.resize(
                self.path_mask,
                (metadata.width, metadata.height),
                interpolation=cv2.INTER_NEAREST
            )
            
        writer = VideoWriter(output_path, metadata.fps, (metadata.width, metadata.height))
        
        # Чтение первого кадра
        reader.set_frame_position(metadata.start_frame)
        ret, prev_frame = reader.read_frame()
        
        if not ret:
            print("Видео пустое или ошибка чтения.")
            reader.release()
            writer.release()
            return None
            
        # Основной цикл обработки
        for _ in tqdm(range(metadata.start_frame, metadata.end_frame - 1), desc="Processing frames"):
            ret, frame = reader.read_frame()
            if not ret:
                break
                
            # Расчет оптического потока
            flow = self.flow_calculator.calculate(prev_frame, frame)
            
            # Анализ потока по путям
            path_flows = self.path_analyzer.analyze_flow(flow)
            
            # Определение пути с поездом
            detected_path = self.train_detector.detect_train(path_flows)
            
            # Визуализация
            output_frame = self.visualizer.draw_flow_visualization(frame, path_flows, detected_path)
            
            writer.write_frame(output_frame)
            prev_frame = frame
            
        reader.release()
        writer.release()
        return output_path
