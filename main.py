import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import json
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

@dataclass
class StoreLayout:
    def __init__(self):
        # Store dimensions in meters
        self.width = 10.0
        self.length = 15.0
        self.height = 3.0

        # Key positions
        self.entrance_position = (5.0, 0.0)
        self.checkout_position = (8.0, 14.0)
        self.fitting_room_position = (9.0, 10.0)
        self.rest_area_position = (1.0, 12.0)
        self.staff_area_position = (0.5, 14.0)

        # Shelf information
        self.shelves = [
            {
                "position": (3.0, 5.0),
                "width": 1.0,
                "length": 2.0,
                "height": 2.0,
                "levels": 4,
                "category": "의류",
                "orientation": 0
            },
            {
                "position": (5.0, 5.0),
                "width": 1.0,
                "length": 2.0,
                "height": 2.0,
                "levels": 4,
                "category": "잡화",
                "orientation": 0
            },
            {
                "position": (7.0, 5.0),
                "width": 1.0,
                "length": 2.0,
                "height": 2.0,
                "levels": 4,
                "category": "신발",
                "orientation": 0
            },
            {
                "position": (3.0, 8.0),
                "width": 1.0,
                "length": 2.0,
                "height": 2.0,
                "levels": 4,
                "category": "가방",
                "orientation": 0
            },
            {
                "position": (5.0, 8.0),
                "width": 1.0,
                "length": 2.0,
                "height": 2.0,
                "levels": 4,
                "category": "액세서리",
                "orientation": 0
            }
        ]

        # Analysis zones (normalized coordinates)
        self.zones = {
            "입구": [(0.45, 0.0), (0.55, 0.1)],
            "계산대": [(0.7, 0.9), (0.9, 1.0)],
            "시착실": [(0.8, 0.6), (1.0, 0.7)],
            "휴게공간": [(0.0, 0.8), (0.2, 0.9)],
            "의류구역": [(0.2, 0.3), (0.4, 0.5)],
            "잡화구역": [(0.4, 0.3), (0.6, 0.5)],
            "신발구역": [(0.6, 0.3), (0.8, 0.5)],
            "가방구역": [(0.2, 0.5), (0.4, 0.7)],
            "액세서리구역": [(0.4, 0.5), (0.6, 0.7)]
        }

class StoreAnalyzer:
    def __init__(self, video_path: str, model_path: str):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.layout = StoreLayout()

        # Initialize tracking related variables
        self.tracks = {}
        self.dwell_times = {}
        self.trajectories = []

        # Store dimensions will be set when processing first frame
        self.frame_width = None
        self.frame_height = None

        # Initialize heatmap
        self.heatmap = None

    def process_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        results = self.model(frame, classes=[0])[0]  # class 0 is person
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        return boxes, confidences, class_ids

    def update_tracking(self, detections):
        current_frame_time = time.time()

        # Convert detections to a list of (x, y) center points
        current_positions = []
        for box in detections:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            current_positions.append((center_x, center_y))

            # Update heatmap
            y = int(center_y)
            x = int(center_x)
            if self.heatmap is None:
                self.heatmap = np.zeros((480, 640), dtype=np.float32)
            if 0 <= y < self.heatmap.shape[0] and 0 <= x < self.heatmap.shape[1]:
                self.heatmap[y, x] += 1

        # Match current detections with existing tracks
        if not self.tracks:
            # Initialize tracks if none exist
            for i, pos in enumerate(current_positions):
                track_id = len(self.tracks)
                self.tracks[track_id] = {
                    'positions': [pos],
                    'last_update': current_frame_time,
                    'current_zone': None,
                    'zone_entry_time': {}
                }
        else:
            # Match detections with existing tracks using Hungarian algorithm
            track_ids = list(self.tracks.keys())
            cost_matrix = np.zeros((len(current_positions), len(track_ids)))

            for i, pos in enumerate(current_positions):
                for j, track_id in enumerate(track_ids):
                    if self.tracks[track_id]['positions']:
                        prev_pos = self.tracks[track_id]['positions'][-1]
                        cost_matrix[i][j] = np.sqrt(
                            (pos[0] - prev_pos[0])**2 +
                            (pos[1] - prev_pos[1])**2
                        )

            if cost_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Update matched tracks
                matched_detections = set()
                matched_tracks = set()

                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i][j] < 100:  # Distance threshold
                        track_id = track_ids[j]
                        self.tracks[track_id]['positions'].append(current_positions[i])
                        self.tracks[track_id]['last_update'] = current_frame_time
                        matched_detections.add(i)
                        matched_tracks.add(track_id)

                        # Update zone information
                        self._update_zone_info(track_id, current_positions[i], current_frame_time)

                # Handle unmatched detections (new tracks)
                for i in range(len(current_positions)):
                    if i not in matched_detections:
                        track_id = max(self.tracks.keys()) + 1 if self.tracks else 0
                        self.tracks[track_id] = {
                            'positions': [current_positions[i]],
                            'last_update': current_frame_time,
                            'current_zone': None,
                            'zone_entry_time': {}
                        }

                # Handle unmatched tracks (lost tracks)
                for track_id in track_ids:
                    if track_id not in matched_tracks:
                        if current_frame_time - self.tracks[track_id]['last_update'] > 1.0:
                            # Calculate final dwell times for lost track
                            if self.tracks[track_id]['current_zone']:
                                zone = self.tracks[track_id]['current_zone']
                                entry_time = self.tracks[track_id]['zone_entry_time'].get(zone)
                                if entry_time:
                                    dwell_time = current_frame_time - entry_time
                                    if zone not in self.dwell_times:
                                        self.dwell_times[zone] = []
                                    self.dwell_times[zone].append(dwell_time)
                            del self.tracks[track_id]

    def _update_zone_info(self, track_id, position, current_time):
        # Convert position to relative coordinates
        rel_x = position[0] / self.frame_width
        rel_y = position[1] / self.frame_height

        # Check current zone
        current_zone = None
        for zone_name, zone_coords in self.layout.zones.items():
            if (zone_coords[0][0] <= rel_x <= zone_coords[1][0] and
                zone_coords[0][1] <= rel_y <= zone_coords[1][1]):
                current_zone = zone_name
                break

        track = self.tracks[track_id]

        # Handle zone entry/exit
        if current_zone != track['current_zone']:
            # Exit old zone
            if track['current_zone']:
                entry_time = track['zone_entry_time'].get(track['current_zone'])
                if entry_time:
                    dwell_time = current_time - entry_time
                    if track['current_zone'] not in self.dwell_times:
                        self.dwell_times[track['current_zone']] = []
                    self.dwell_times[track['current_zone']].append(dwell_time)

            # Enter new zone
            if current_zone:
                track['zone_entry_time'][current_zone] = current_time

            track['current_zone'] = current_zone

    def is_in_zone(self, point: Tuple[float, float], zone: List[Tuple[float, float]]) -> bool:
        x, y = point
        (x1, y1), (x2, y2) = zone
        return x1 <= x <= x2 and y1 <= y <= y2

    def create_store_3d(self) -> o3d.geometry.TriangleMesh:
        store = o3d.geometry.TriangleMesh()

        # Create floor
        floor = o3d.geometry.TriangleMesh.create_box(
            width=self.layout.width,
            height=0.1,
            depth=self.layout.length
        )
        floor.paint_uniform_color([0.8, 0.8, 0.8])
        store += floor

        # Create walls
        wall_thickness = 0.1
        walls = []
        # Front wall with entrance
        front_wall_left = o3d.geometry.TriangleMesh.create_box(
            width=self.layout.entrance_position[0] - 1,
            height=self.layout.height,
            depth=wall_thickness
        )
        front_wall_right = o3d.geometry.TriangleMesh.create_box(
            width=self.layout.width - self.layout.entrance_position[0] - 1,
            height=self.layout.height,
            depth=wall_thickness
        )
        front_wall_right.translate([self.layout.entrance_position[0] + 1, 0, 0])
        walls.extend([front_wall_left, front_wall_right])

        # Add shelves
        for shelf in self.layout.shelves:
            shelf_mesh = o3d.geometry.TriangleMesh.create_box(
                width=shelf["width"],
                height=shelf["height"],
                depth=shelf["length"]
            )
            shelf_mesh.translate([
                shelf["position"][0],
                0,
                shelf["position"][1]
            ])
            shelf_mesh.rotate(
                o3d.geometry.get_rotation_matrix_from_xyz(
                    [0, shelf["orientation"] * np.pi / 180, 0]
                )
            )
            shelf_mesh.paint_uniform_color([0.7, 0.5, 0.3])
            store += shelf_mesh

            # Add shelf levels
            level_height = shelf["height"] / shelf["levels"]
            for level in range(1, shelf["levels"]):
                level_mesh = o3d.geometry.TriangleMesh.create_box(
                    width=shelf["width"],
                    height=0.02,
                    depth=shelf["length"]
                )
                level_mesh.translate([
                    shelf["position"][0],
                    level * level_height,
                    shelf["position"][1]
                ])
                level_mesh.rotate(
                    o3d.geometry.get_rotation_matrix_from_xyz(
                        [0, shelf["orientation"] * np.pi / 180, 0]
                    )
                )
                level_mesh.paint_uniform_color([0.6, 0.4, 0.2])
                store += level_mesh

        return store

    def visualize_3d_with_heatmap(self):
        store = self.create_store_3d()

        # Convert heatmap to point cloud
        normalized_heatmap = self.heatmap / np.max(self.heatmap)
        points = []
        colors = []

        for y in range(self.heatmap.shape[0]):
            for x in range(self.heatmap.shape[1]):
                if normalized_heatmap[y, x] > 0.1:  # Threshold to reduce noise
                    # Convert pixel coordinates to store coordinates
                    store_x = (x / self.heatmap.shape[1]) * self.layout.width
                    store_z = (y / self.heatmap.shape[0]) * self.layout.length
                    points.append([store_x, 0.1, store_z])  # Slightly above floor

                    # Use red color intensity based on heatmap value
                    colors.append([normalized_heatmap[y, x], 0, 0])

        heatmap_points = o3d.geometry.PointCloud()
        heatmap_points.points = o3d.utility.Vector3dVector(points)
        heatmap_points.colors = o3d.utility.Vector3dVector(colors)

        # Save to file
        o3d.io.write_triangle_mesh("store_3d.obj", store)
        o3d.io.write_point_cloud("heatmap_3d.ply", heatmap_points)

        # Save metadata
        metadata = {
            "scale": {
                "pixels_to_meters_x": self.layout.width / 640,
                "pixels_to_meters_y": self.layout.length / 480
            },
            "zones": self.zones,
            "dwell_times": {k: np.mean(v) for k, v in self.dwell_times.items() if v}
        }
        with open("store_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def analyze_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0

        # Get video dimensions
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.heatmap = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, confidences, class_ids = self.process_frame(frame)
            self.update_tracking(boxes)
            frame_idx += 1

        cap.release()

        # Save results
        self.save_results()

    def visualize_dwell_times(self):
        plt.figure(figsize=(12, 6))
        zones = list(self.dwell_times.keys())
        avg_times = [np.mean(times) if times else 0 for times in self.dwell_times.values()]

        plt.bar(zones, avg_times)
        plt.title("평균 체류 시간 (초)")
        plt.xlabel("구역")
        plt.ylabel("시간 (초)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("dwell_time.png")
        plt.close()

    def visualize_trajectories(self):
        plt.figure(figsize=(12, 8))
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw zones
        for zone_name, zone_coords in self.zones.items():
            (x1, y1), (x2, y2) = zone_coords
            pt1 = (int(x1 * 640), int(y1 * 480))
            pt2 = (int(x2 * 640), int(y2 * 480))
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(img, zone_name, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw trajectories
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.trajectories)))
        for trajectory, color in zip(self.trajectories, colors):
            points = np.array([(int(x), int(y)) for x, y, _ in trajectory])
            color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

            for i in range(len(points) - 1):
                cv2.line(img, tuple(points[i]), tuple(points[i + 1]), color_bgr, 2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("고객 동선")
        plt.axis('off')
        plt.savefig("trajectories.png")
        plt.close()

    def save_results(self):
        # Save 3D store model
        store_mesh = self.create_store_3d()
        o3d.io.write_triangle_mesh("store_3d.obj", store_mesh)

        # Save heatmap as point cloud
        heatmap_points = np.argwhere(self.heatmap > 0)
        heatmap_colors = np.zeros((len(heatmap_points), 3))
        heatmap_colors[:, 0] = self.heatmap[heatmap_points[:, 0], heatmap_points[:, 1]] / np.max(self.heatmap)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(heatmap_points)
        pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
        o3d.io.write_point_cloud("heatmap_3d.ply", pcd)

        # Save metadata
        metadata = {
            "scale": {
                "pixels_per_meter_x": self.frame_width / self.layout.width,
                "pixels_per_meter_y": self.frame_height / self.layout.length
            },
            "zones": self.layout.zones,
            "dwell_times": self.dwell_times
        }
        with open("store_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Visualize average dwell times
        plt.figure(figsize=(10, 6))
        zones = list(self.dwell_times.keys())
        avg_times = [np.mean(times) if times else 0 for times in self.dwell_times.values()]
        plt.bar(zones, avg_times)
        plt.xlabel("Zone")
        plt.ylabel("Average Dwell Time (seconds)")
        plt.title("Average Dwell Time by Zone")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("dwell_time.png")
        plt.close()

        # Visualize customer movement patterns
        plt.figure(figsize=(10, 6))
        for track_id, track_data in self.tracks.items():
            positions = track_data.get("positions", [])
            if positions:  # Only plot if there are positions
                positions = np.array(positions)
                if positions.ndim == 1:  # Single position
                    plt.plot(positions[0], positions[1], 'o', label=f'Track {track_id}')
                else:  # Multiple positions
                    plt.plot(positions[:, 0], positions[:, 1], '-', label=f'Track {track_id}')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Customer Movement Patterns")
        plt.legend()
        plt.grid(True)
        plt.savefig("trajectories.png")
        plt.close()

def main():
    analyzer = StoreAnalyzer("data/store.mp4", "yolov8n.pt")
    analyzer.analyze_video()

if __name__ == "__main__":
    main()