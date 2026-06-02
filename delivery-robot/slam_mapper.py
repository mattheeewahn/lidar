"""
간이 SLAM - CygLiDAR D1으로 점유 격자 지도(Occupancy Grid Map) 생성
- Hector SLAM 스타일의 스캔 매칭 기반
- 라이다 데이터를 누적하여 2D 지도 생성
"""

import numpy as np
import math
import time
import sys
import os

# 같은 폴더의 모듈 import 보장
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_driver import CygLidarD1


class SLAMMapper:
    """
    간이 2D SLAM
    - 점유 격자 지도 (Occupancy Grid Map) 생성
    - 스캔 매칭으로 로봇 위치 추정
    """

    def __init__(self, map_size=200, resolution=0.05):
        """
        Args:
            map_size: 맵 격자 크기 (200x200 = 10m x 10m)
            resolution: 격자 한 칸 크기 (m), 0.05 = 5cm
        """
        self.map_size = map_size
        self.resolution = resolution

        # 점유 격자 맵 (log-odds 표현)
        # 0 = 미탐사, 양수 = 장애물, 음수 = 빈 공간
        self.log_odds_map = np.zeros((map_size, map_size), dtype=np.float32)

        # 로봇 위치/방향 (맵 중앙에서 시작)
        self.robot_x = map_size / 2.0  # 격자 좌표
        self.robot_y = map_size / 2.0
        self.robot_theta = 0.0  # 라디안

        # log-odds 파라미터
        self.l_occ = 0.9   # 장애물 확률 증가량
        self.l_free = -0.7  # 빈 공간 확률 감소량
        self.l_max = 5.0
        self.l_min = -5.0

        # 라이다
        self.lidar = CygLidarD1()
        self.angles_deg = self.lidar.get_angles()  # 0~120도
        self.angles_rad = np.radians(self.angles_deg - 60)  # 중앙 기준 -60~+60도

        # 이전 스캔 (스캔 매칭용)
        self.prev_scan = None

        # 스캔 이력 (경로 시각화용)
        self.trajectory = []

        self.running = False

    def connect(self):
        self.lidar.connect()
        self.lidar.start_scan()
        time.sleep(0.5)
        print("[SLAM] 라이다 연결 및 스캔 시작")

    def scan_to_points(self, distances):
        """
        거리 데이터를 로봇 로컬 좌표 포인트로 변환

        Args:
            distances: mm 단위 거리 배열 (160개)

        Returns:
            points: (N, 2) 배열, 로봇 기준 (x, y) 좌표 (m)
        """
        # mm → m 변환, 유효 거리만 사용 (10cm ~ 8m)
        valid = (distances > 100) & (distances < 8000)
        dist_m = distances[valid] / 1000.0
        angles = self.angles_rad[valid]

        x = dist_m * np.cos(angles)
        y = dist_m * np.sin(angles)

        return np.column_stack((x, y))

    def scan_match(self, current_points):
        """
        간이 스캔 매칭: ICP 기반 위치 보정
        이전 스캔과 현재 스캔을 비교하여 이동량 추정

        Returns:
            (dx, dy, dtheta) 이동량
        """
        if self.prev_scan is None or len(current_points) < 10:
            return 0.0, 0.0, 0.0

        if len(self.prev_scan) < 10:
            return 0.0, 0.0, 0.0

        # 간단한 중심점 비교 기반 이동 추정
        curr_center = np.mean(current_points, axis=0)
        prev_center = np.mean(self.prev_scan, axis=0)

        dx = (prev_center[0] - curr_center[0]) * 0.3  # 감쇠 계수
        dy = (prev_center[1] - curr_center[1]) * 0.3

        # 각도 변화 추정 (좌/우 포인트 분포 차이)
        if len(current_points) > 20 and len(self.prev_scan) > 20:
            curr_left = np.mean(current_points[:len(current_points)//2, 1])
            curr_right = np.mean(current_points[len(current_points)//2:, 1])
            prev_left = np.mean(self.prev_scan[:len(self.prev_scan)//2, 1])
            prev_right = np.mean(self.prev_scan[len(self.prev_scan)//2:, 1])

            dtheta = ((curr_left - curr_right) - (prev_left - prev_right)) * 0.1
        else:
            dtheta = 0.0

        return dx, dy, dtheta

    def update_map(self, distances):
        """
        한 프레임의 라이다 데이터로 맵 업데이트

        Args:
            distances: mm 단위 거리 배열
        """
        points = self.scan_to_points(distances)
        if len(points) < 5:
            return

        # 스캔 매칭으로 위치 보정
        dx, dy, dtheta = self.scan_match(points)
        self.robot_theta += dtheta
        self.robot_x += (dx * math.cos(self.robot_theta) - dy * math.sin(self.robot_theta)) / self.resolution
        self.robot_y += (dx * math.sin(self.robot_theta) + dy * math.cos(self.robot_theta)) / self.resolution

        # 맵 범위 제한
        self.robot_x = np.clip(self.robot_x, 5, self.map_size - 5)
        self.robot_y = np.clip(self.robot_y, 5, self.map_size - 5)

        # 월드 좌표로 변환 후 맵에 반영
        cos_t = math.cos(self.robot_theta)
        sin_t = math.sin(self.robot_theta)

        for i in range(len(points)):
            # 로컬 → 월드
            wx = points[i, 0] * cos_t - points[i, 1] * sin_t
            wy = points[i, 0] * sin_t + points[i, 1] * cos_t

            # 월드 → 격자 좌표
            end_x = int(self.robot_x + wx / self.resolution)
            end_y = int(self.robot_y + wy / self.resolution)

            # 맵 범위 확인
            if 0 <= end_x < self.map_size and 0 <= end_y < self.map_size:
                # 레이캐스팅: 로봇~끝점 사이를 빈 공간으로
                self._bresenham_free(int(self.robot_x), int(self.robot_y), end_x, end_y)
                # 끝점은 장애물
                self.log_odds_map[end_y, end_x] = min(
                    self.log_odds_map[end_y, end_x] + self.l_occ, self.l_max
                )

        # 궤적 기록
        self.trajectory.append((self.robot_x, self.robot_y, self.robot_theta))
        self.prev_scan = points.copy()

    def _bresenham_free(self, x0, y0, x1, y1):
        """브레젠햄 알고리즘으로 빈 공간 마킹 (끝점 제외)"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if x0 == x1 and y0 == y1:
                break
            if 0 <= x0 < self.map_size and 0 <= y0 < self.map_size:
                self.log_odds_map[y0, x0] = max(
                    self.log_odds_map[y0, x0] + self.l_free, self.l_min
                )
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def get_occupancy_grid(self):
        """
        점유 격자 맵 반환 (0~255 이미지 형태)
        - 0 (검정) = 장애물
        - 128 (회색) = 미탐사
        - 255 (흰색) = 빈 공간

        Returns:
            (map_size, map_size) uint8 배열
        """
        # log-odds → 확률
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds_map))

        grid = np.full((self.map_size, self.map_size), 128, dtype=np.uint8)
        grid[prob > 0.65] = 0      # 장애물
        grid[prob < 0.35] = 255    # 빈 공간

        return grid

    def get_robot_position(self):
        """로봇 위치 반환 (격자 좌표)"""
        return int(self.robot_x), int(self.robot_y), self.robot_theta

    def run_mapping(self, duration=30):
        """
        일정 시간 동안 매핑 실행

        Args:
            duration: 매핑 시간 (초)
        """
        self.running = True
        start = time.time()
        print(f"[SLAM] 매핑 시작 ({duration}초)")

        while self.running and (time.time() - start) < duration:
            distances = self.lidar.read_frame()
            if distances is not None:
                self.update_map(distances)
            time.sleep(0.05)  # ~20Hz

        print("[SLAM] 매핑 완료")

    def stop(self):
        self.running = False

    def close(self):
        self.stop()
        self.lidar.close()
        print("[SLAM] 종료")
