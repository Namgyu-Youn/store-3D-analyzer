import cv2
import numpy as np
from pathlib import Path

class CCTVPreprocessor:
    def __init__(self, video_paths):
        self.video_paths = video_paths

    def extract_frames(self, interval=1):
        """지정된 간격으로 프레임 추출"""
        frames_dict = {}

        for path in self.video_paths:
            cap = cv2.VideoCapture(path)
            frames = []
            count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if count % interval == 0:
                    # 노이즈 제거 및 이미지 개선
                    frame = cv2.GaussianBlur(frame, (5,5), 0)
                    frame = cv2.fastNlMeansDenoisingColored(frame)
                    frames.append(frame)

                count += 1

            cap.release()
            frames_dict[Path(path).stem] = frames

        return frames_dict

    def calibrate_cameras(self, frames_dict, checkerboard_size=(9,6)):
        """카메라 캘리브레이션"""
        calibration_data = {}

        for camera_id, frames in frames_dict.items():
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objpoints = []
            imgpoints = []

            objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)

            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners2)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            calibration_data[camera_id] = {
                'matrix': mtx,
                'distortion': dist,
                'rotation_vectors': rvecs,
                'translation_vectors': tvecs
            }

        return calibration_data