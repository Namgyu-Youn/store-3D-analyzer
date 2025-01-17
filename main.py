import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import psutil
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class StoreLayout:
    """매장 레이아웃을 정의하는 클래스

    매장의 물리적 크기, 주요 위치, 선반 정보, 분석 구역 등을 포함
    """

    def __init__(self):
        # Store dimensions in meters
        self.width = 10.0
        self.length = 15.0
        self.height = 3.0

        # Key positions (x, y coordinates in meters)
        self.entrance_position = (5.0, 0.0)
        self.checkout_position = (8.0, 14.0)
        self.fitting_room_position = (9.0, 10.0)
        self.rest_area_position = (1.0, 12.0)
        self.staff_area_position = (0.5, 14.0)

        # 선반 정보 (위치, 크기, 높이, 레벨, 카테고리, 방향)
        self.shelves = [
            {"position": (3.0, 5.0), "width": 1.0, "length": 2.0, "height": 2.0, "levels": 4, "category": "의류", "orientation": 0},
            {"position": (5.0, 5.0), "width": 1.0, "length": 2.0, "height": 2.0, "levels": 4, "category": "잡화", "orientation": 0},
            {"position": (7.0, 5.0), "width": 1.0, "length": 2.0, "height": 2.0, "levels": 4, "category": "신발", "orientation": 0},
            {"position": (3.0, 8.0), "width": 1.0, "length": 2.0, "height": 2.0, "levels": 4, "category": "가방", "orientation": 0},
            {"position": (5.0, 8.0), "width": 1.0, "length": 2.0, "height": 2.0, "levels": 4, "category": "액세서리", "orientation": 0},
        ]

        # 분석 구역 (정규화된 좌표)
        self.zones = {
            "입구": [(0.45, 0.0), (0.55, 0.1)],
            "계산대": [(0.7, 0.9), (0.9, 1.0)],
            "시착실": [(0.8, 0.6), (1.0, 0.7)],
            "휴게공간": [(0.0, 0.8), (0.2, 0.9)],
            "의류구역": [(0.2, 0.3), (0.4, 0.5)],
            "잡화구역": [(0.4, 0.3), (0.6, 0.5)],
            "신발구역": [(0.6, 0.3), (0.8, 0.5)],
            "가방구역": [(0.2, 0.5), (0.4, 0.7)],
            "액세서리구역": [(0.4, 0.5), (0.6, 0.7)],
        }


class StoreAnalyzer:
    """매장 분석을 위한 메인 클래스

    비디오 처리, 객체 추적, 3D 모델링, 히트맵 생성 등의 기능 포함
    """

    def __init__(self, video_path: str, model_path: str):
        """
        Args:
            video_path (str): 분석할 비디오 파일 경로
            model_path (str): YOLO 모델 파일 경로
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # YOLO 모델 초기화 (GPU 메모리 관리 개선)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)

        self.layout = StoreLayout()

        # 추적 관련 변수 초기화
        self.tracks: dict[int, dict] = {}
        self.dwell_times: dict[str, list[float]] = {}
        self.trajectories: list[list[tuple[float, float, float]]] = []

        # 프레임 크기는 비디오 처리 시작시 설정
        self.frame_width: int | None = None
        self.frame_height: int | None = None
        self.heatmap: np.ndarray | None = None

        # 메모리 사용량 모니터링
        self.process = psutil.Process()

    def _initialize_new_track(self, position: tuple[float, float], current_time: float) -> None:
        """새로운 추적 객체 초기화

        Args:
            position: 초기 위치 (x, y)
            current_time: 현재 시간
        """
        track_id = max(self.tracks.keys()) + 1 if self.tracks else 0
        self.tracks[track_id] = {"positions": [position], "last_update": current_time, "current_zone": None, "zone_entry_time": {}}

    def _calculate_cost_matrix(self, current_positions: list[tuple[float, float]], track_ids: list[int]) -> np.ndarray:
        """추적 매칭을 위한 비용 행렬 계산

        Args:
            current_positions: 현재 프레임의 검출 위치들
            track_ids: 현재 활성화된 트랙 ID들

        Returns:
            np.ndarray: 비용 행렬
        """
        cost_matrix = np.zeros((len(current_positions), len(track_ids)))

        for i, pos in enumerate(current_positions):
            for j, track_id in enumerate(track_ids):
                if self.tracks[track_id]["positions"]:
                    prev_pos = self.tracks[track_id]["positions"][-1]
                    cost_matrix[i][j] = np.sqrt((pos[0] - prev_pos[0]) ** 2 + (pos[1] - prev_pos[1]) ** 2)

        return cost_matrix

    def _update_matched_tracks(
        self,
        current_positions: list[tuple[float, float]],
        track_ids: list[int],
        cost_matrix: np.ndarray,
        row_ind: np.ndarray,
        col_ind: np.ndarray,
        current_time: float,
    ) -> None:
        """매칭된 트랙 정보 업데이트

        Args:
            current_positions: 현재 프레임의 검출 위치들
            track_ids: 현재 활성화된 트랙 ID들
            cost_matrix: 매칭 비용 행렬
            row_ind: 행 인덱스 (헝가리안 알고리즘 결과)
            col_ind: 열 인덱스 (헝가리안 알고리즘 결과)
            current_time: 현재 시간
        """
        # 매칭된 검출과 트랙 기록
        matched_detections = set()
        matched_tracks = set()

        # 거리 임계값 (픽셀)
        DISTANCE_THRESHOLD = 100.0

        for det_idx, track_idx in zip(row_ind, col_ind, strict=False):
            if cost_matrix[det_idx][track_idx] < DISTANCE_THRESHOLD:
                track_id = track_ids[track_idx]
                self.tracks[track_id]["positions"].append(current_positions[det_idx])
                self.tracks[track_id]["last_update"] = current_time
                matched_detections.add(det_idx)
                matched_tracks.add(track_id)

                # 구역 정보 업데이트
                self._update_zone_info(track_id, current_positions[det_idx], current_time)

        # 매칭되지 않은 검출에 대해 새로운 트랙 생성
        for i in range(len(current_positions)):
            if i not in matched_detections:
                self._initialize_new_track(current_positions[i], current_time)

        # 매칭되지 않은 트랙 처리 (오래된 트랙 제거)
        for track_id in track_ids:
            if track_id not in matched_tracks:
                if current_time - self.tracks[track_id]["last_update"] > 1.0:  # 1초 이상 업데이트 없으면 제거
                    # 구역 체류 시간 계산
                    if self.tracks[track_id]["current_zone"]:
                        zone = self.tracks[track_id]["current_zone"]
                        entry_time = self.tracks[track_id]["zone_entry_time"].get(zone)
                        if entry_time:
                            dwell_time = current_time - entry_time
                            if zone not in self.dwell_times:
                                self.dwell_times[zone] = []
                            self.dwell_times[zone].append(dwell_time)

                    # 트랙 제거
                    del self.tracks[track_id]

    def _update_zone_info(self, track_id: int, position: tuple[float, float], current_time: float) -> None:
        """현재 위치에 따른 구역 정보 업데이트

        Args:
            track_id: 트랙 ID
            position: 현재 위치 (x, y)
            current_time: 현재 시간
        """
        # 정규화된 좌표로 변환
        norm_x = position[0] / self.frame_width
        norm_y = position[1] / self.frame_height

        # 현재 구역 찾기
        current_zone = None
        for zone_name, ((x1, y1), (x2, y2)) in self.layout.zones.items():
            if x1 <= norm_x <= x2 and y1 <= norm_y <= y2:
                current_zone = zone_name
                break

        # 구역 변경 감지 및 체류 시간 업데이트
        prev_zone = self.tracks[track_id]["current_zone"]
        if current_zone != prev_zone:
            # 이전 구역 체류 시간 계산
            if prev_zone:
                entry_time = self.tracks[track_id]["zone_entry_time"].get(prev_zone)
                if entry_time:
                    dwell_time = current_time - entry_time
                    if prev_zone not in self.dwell_times:
                        self.dwell_times[prev_zone] = []
                    self.dwell_times[prev_zone].append(dwell_time)

            # 새로운 구역 정보 업데이트
            self.tracks[track_id]["current_zone"] = current_zone
            if current_zone:
                self.tracks[track_id]["zone_entry_time"][current_zone] = current_time

    def _init_video_capture(self) -> cv2.VideoCapture:
        """비디오 캡처 초기화 및 검증

        Returns:
            cv2.VideoCapture: 초기화된 비디오 캡처 객체
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video dimensions: {self.frame_width}x{self.frame_height}")
        return cap

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """단일 프레임 처리 및 객체 검출

        Args:
            frame (np.ndarray): 처리할 비디오 프레임

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 박스 좌표, 신뢰도, 클래스 ID
        """
        # CUDA 메모리 관리 개선
        with torch.no_grad():
            # 이미지 전처리
            # 1. BGR에서 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. 크기 조정 (640x640)
            frame_resized = cv2.resize(frame_rgb, (640, 640))

            # 3. 배치 차원 추가 및 채널 순서 변경 (HWC -> BCHW)
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()  # (3, 640, 640)
            frame_tensor = frame_tensor.unsqueeze(0)  # (1, 3, 640, 640)

            # 4. 정규화 (0-255 -> 0-1)
            frame_tensor = frame_tensor / 255.0

            # GPU로 전송
            frame_tensor = frame_tensor.to(self.device)

            # 모델 추론
            results = self.model(frame_tensor, classes=[0])[0]  # class 0 is person

            # 결과 추출
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()

            # 박스 좌표를 원본 이미지 크기로 변환
            if len(boxes) > 0:
                boxes[:, [0, 2]] *= self.frame_width / 640
                boxes[:, [1, 3]] *= self.frame_height / 640

            # CUDA 캐시 정리
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return boxes, confidences, class_ids

    def update_tracking(self, detections: np.ndarray) -> None:
        current_frame_time = time.time()

        # 히트맵 초기화 (필요한 경우)
        if self.heatmap is None:
            self.heatmap = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)

        # 현재 검출된 위치를 중심점으로 변환
        current_positions = []
        for box in detections:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            current_positions.append((center_x, center_y))

            # 히트맵 업데이트 - 가우시안 블러를 적용하여 부드러운 히트맵 생성
            x, y = int(center_x), int(center_y)
            if 0 <= y < self.heatmap.shape[0] and 0 <= x < self.heatmap.shape[1]:
                # 가우시안 커널 크기 설정
                kernel_size = 31
                sigma = 7
                temp_heatmap = np.zeros_like(self.heatmap)
                temp_heatmap[y, x] = 1
                self.heatmap += cv2.GaussianBlur(temp_heatmap, (kernel_size, kernel_size), sigma)

        self._match_and_update_tracks(current_positions, current_frame_time)

    def _match_and_update_tracks(self, current_positions: list[tuple[float, float]], current_frame_time: float) -> None:
        """추적 매칭 및 업데이트 로직

        Args:
            current_positions: 현재 프레임의 검출 위치
            current_frame_time: 현재 프레임 시간
        """
        if not self.tracks:
            # 첫 프레임이거나 모든 트랙이 삭제된 경우 새로 초기화
            for i, pos in enumerate(current_positions):
                self._initialize_new_track(pos, current_frame_time)
            return

        track_ids = list(self.tracks.keys())
        cost_matrix = self._calculate_cost_matrix(current_positions, track_ids)

        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            self._update_matched_tracks(current_positions, track_ids, cost_matrix, row_ind, col_ind, current_frame_time)

    def create_store_3d(self) -> o3d.geometry.TriangleMesh:
        """3D 매장 모델 생성

        Returns:
            o3d.geometry.TriangleMesh: 결합된 3D 매장 모델
        """
        meshes = []  # 모든 메시를 저장할 리스트

        # 바닥 생성
        floor = o3d.geometry.TriangleMesh.create_box(width=self.layout.width, height=0.1, depth=self.layout.length)
        floor.paint_uniform_color([0.8, 0.8, 0.8])
        meshes.append(floor)

        # 선반 추가
        for shelf in self.layout.shelves:
            shelf_mesh = self._create_shelf_mesh(shelf)
            meshes.append(shelf_mesh)

        # 모든 메시 결합
        combined = meshes[0]
        for mesh in meshes[1:]:
            combined += mesh

        return combined

    def _create_shelf_mesh(self, shelf: dict) -> o3d.geometry.TriangleMesh:
        """선반 메시 생성

        Args:
            shelf (Dict): 선반 정보

        Returns:
            o3d.geometry.TriangleMesh: 선반 메시
        """
        shelf_mesh = o3d.geometry.TriangleMesh.create_box(width=shelf["width"], height=shelf["height"], depth=shelf["length"])

        # 선반 위치 및 회전 설정
        shelf_mesh.translate([shelf["position"][0], 0, shelf["position"][1]])
        shelf_mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, shelf["orientation"] * np.pi / 180, 0]))
        shelf_mesh.paint_uniform_color([0.7, 0.5, 0.3])

        return shelf_mesh

    def analyze_video(self) -> None:
        """비디오 분석 메인 함수"""
        logger.info("Starting video analysis...")
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        try:
            cap = self._init_video_capture()
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 프레임 처리
                boxes, confidences, class_ids = self.process_frame(frame)
                self.update_tracking(boxes)

                frame_count += 1
                if frame_count % 100 == 0:  # 주기적으로 메모리 사용량 로깅
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    logger.info(f"Frame {frame_count}: Memory usage: {current_memory:.2f} MB")

            cap.release()

            # 결과 저장
            self.save_results()

        except Exception as e:
            logger.error(f"Error during video analysis: {e}")
            raise
        finally:
            if "cap" in locals():
                cap.release()

            final_memory = self.process.memory_info().rss / 1024 / 1024
            logger.info(f"Final memory usage: {final_memory:.2f} MB")

    def save_results(self) -> None:
        """분석 결과 저장"""
        # 히트맵 저장
        if self.heatmap is not None:
            # 히트맵 정규화 (0-1 범위로)
            normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 1, cv2.NORM_MINMAX)

            # 컬러맵 적용을 위해 0-255 범위로 변환
            heatmap_255 = (normalized_heatmap * 255).astype(np.uint8)

            # COLORMAP_JET 적용
            colored_heatmap = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)

            # 알파 블렌딩을 위한 마스크 생성
            alpha = normalized_heatmap[..., np.newaxis]

            # 최종 히트맵 (부분 투명도 적용)
            final_heatmap = (colored_heatmap * alpha).astype(np.uint8)

            # 저장
            cv2.imwrite("heatmap.png", final_heatmap)

            # 디버깅을 위한 정보 출력
            logger.info(
                f"Heatmap stats - Min: {self.heatmap.min():.2f}, Max: {self.heatmap.max():.2f}, "
                f"Mean: {self.heatmap.mean():.2f}, Shape: {self.heatmap.shape}"
            )

            # 체류 시간 분석 결과 저장
            with open("dwell_times.txt", "w", encoding="utf-8") as f:
                f.write("Zone Dwell Time Analysis\n")
                f.write("======================\n\n")
                for zone, times in self.dwell_times.items():
                    if times:
                        avg_time = sum(times) / len(times)
                        max_time = max(times)
                        min_time = min(times)
                        f.write(f"Zone: {zone}\n")
                        f.write(f"Average dwell time: {avg_time:.2f} seconds\n")
                        f.write(f"Maximum dwell time: {max_time:.2f} seconds\n")
                        f.write(f"Minimum dwell time: {min_time:.2f} seconds\n")
                        f.write(f"Number of visits: {len(times)}\n\n")


def main():
    """메인 실행 함수"""
    try:
        video_path = "data/store.mp4"
        model_path = "yolov8n.pt"

        logger.info("Initializing Store Analyzer...")
        analyzer = StoreAnalyzer(video_path, model_path)

        logger.info("Starting video analysis...")
        analyzer.analyze_video()

        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
