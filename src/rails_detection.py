import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def avg_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

class ImagePreprocessor:
    def __init__(self, canny1: int = 80, canny2: int = 200):
        self.canny1 = canny1
        self.canny2 = canny2

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        cv2.blur(img_bgr, (7, 7))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, self.canny1, self.canny2)

class LineDetector:
    def __init__(
        self,
        hough_threshold: int = 115,
        min_line_length: int = 100,
        max_line_gap: int = 40,
        max_slope_degrees: float = 20
    ):
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.max_slope = np.tan(np.deg2rad(max_slope_degrees))

    def detect(self, edges: np.ndarray) -> List[Line]:
        lines_p = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines_p is None:
            return []

        return self._filter_lines(lines_p)

    def _filter_lines(self, lines_p: np.ndarray) -> List[Line]:
        acceptable_lines = []
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            
            slope = abs((y2 - y1) / dx)
            if slope <= self.max_slope:
                if x2 < x1:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                acceptable_lines.append(Line(x1, y1, x2, y2))
        
        return sorted(acceptable_lines, key=lambda x: x.avg_y)

class LineGrouper:
    def __init__(self, eps_y: int = 25):
        self.eps_y = eps_y

    def group_lines(self, lines: List[Line]) -> List[Line]:
        if not lines:
            return []

        used = [False] * len(lines)
        grouped_lines = []

        for i in range(len(lines)):
            if used[i]:
                continue

            y_mid_i = lines[i].avg_y
            group = [lines[i]]
            used[i] = True

            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue
                
                if abs(lines[j].avg_y - y_mid_i) < self.eps_y:
                    group.append(lines[j])
                    used[j] = True

            grouped_lines.append(self._merge_group(group))

        return sorted(grouped_lines, key=lambda x: x.avg_y, reverse=True)

    def _merge_group(self, group: List[Line]) -> Line:
        xs = []
        ys = []
        for line in group:
            xs.extend([line.x1, line.x2])
            ys.extend([line.y1, line.y2])
        
        x_min = min(xs)
        x_max = max(xs)
        y_mean = int(np.mean(ys))
        return Line(x_min, y_mean, x_max, y_mean)

class PathVisualizer:
    def __init__(self):
        self.colors = [
            (0,0,255),    # красный
            (0,255,0),    # зелёный
            (255,0,0),    # синий
            (0,255,255),  # жёлтый
            (255,0,255),  # фиолетовый
            (255,255,0),  # голубой
        ]
        self.alpha = 0.3

    def visualize(self, img_bgr: np.ndarray, grouped_lines: List[Line]) -> Tuple[np.ndarray, np.ndarray, int]:
        h, w = img_bgr.shape[:2]
        out_bgr = img_bgr.copy()
        overlay = out_bgr.copy()
        path_mask = np.zeros((h, w), dtype=np.uint8)

        num_paths = len(grouped_lines) // 2
        path_index = 0

        for i in range(0, len(grouped_lines) - 1, 2):
            path_index += 1
            line_a, line_b = grouped_lines[i], grouped_lines[i + 1]
            
            y_top = min(line_a.avg_y, line_b.avg_y)
            y_bot = max(line_a.avg_y, line_b.avg_y)
            
            pts = np.array([
                [0, y_top],
                [w, y_top],
                [w, y_bot],
                [0, y_bot]
            ], dtype=np.int32)

            cv2.fillPoly(overlay, [pts], self.colors[i % len(self.colors)])
            cv2.fillPoly(path_mask, [pts], path_index)

        cv2.addWeighted(overlay, self.alpha, out_bgr, 1 - self.alpha, 0, out_bgr)
        
        self._draw_lines_and_labels(out_bgr, grouped_lines)
        
        return out_bgr, path_mask, num_paths

    def _draw_lines_and_labels(self, img: np.ndarray, lines: List[Line]):
        for idx, line in enumerate(lines):
            cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), (255, 255, 255), 2)
            label = f"Rail {idx+1}"
            cv2.putText(
                img,
                label,
                (line.x1, line.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

class RailDetector:
    def __init__(
        self,
        canny1: int = 80,
        canny2: int = 200,
        hough_threshold: int = 115,
        min_line_length: int = 100,
        max_line_gap: int = 40,
        max_slope_degrees: float = 20,
        eps_y: int = 25
    ):
        self.preprocessor = ImagePreprocessor(canny1, canny2)
        self.line_detector = LineDetector(hough_threshold, min_line_length, max_line_gap, max_slope_degrees)
        self.line_grouper = LineGrouper(eps_y)
        self.visualizer = PathVisualizer()

    def detect_rails(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        edges = self.preprocessor.preprocess(img_bgr)
        lines = self.line_detector.detect(edges)
        
        if not lines:
            print("Не найдено линий HoughLinesP.")
            return None, None, 0

        grouped_lines = self.line_grouper.group_lines(lines)
        
        if not grouped_lines:
            print("Нет подходящих (полугоризонтальных) линий.")
            return None, None, 0

        return self.visualizer.visualize(img_bgr, grouped_lines)
