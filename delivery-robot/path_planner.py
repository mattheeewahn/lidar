"""
A* 경로 탐색 + 경로 추종 컨트롤러
- SLAM으로 생성된 지도 위에서 목적지까지 경로 계산
- 경로를 따라 아두이노에 모터 명령 전송
"""

import numpy as np
import heapq
import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motor_controller import MotorController


class PathPlanner:
    """A* 기반 경로 탐색기"""

    def __init__(self, occupancy_grid, resolution=0.05):
        """
        Args:
            occupancy_grid: 점유 격자 맵 (0=장애물, 255=빈공간, 128=미탐사)
            resolution: 격자 해상도 (m/cell)
        """
        self.grid = occupancy_grid
        self.resolution = resolution
        self.map_size = occupancy_grid.shape[0]

        # 장애물 팽창 (로봇 크기 고려, 로봇 폭 ~30cm → 6셀 팽창)
        self.inflated_grid = self._inflate_obstacles(radius=6)

    def _inflate_obstacles(self, radius=6):
        """장애물 주변을 팽창시켜 로봇이 벽에 부딪히지 않게"""
        from scipy.ndimage import binary_dilation

        # 장애물 마스크
        obstacle_mask = self.grid < 64  # 장애물 + 약간의 마진

        # 원형 구조 요소
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        structure = (x**2 + y**2) <= radius**2

        # 팽창
        inflated = binary_dilation(obstacle_mask, structure=structure)

        # 결과 맵 생성
        result = self.grid.copy()
        result[inflated] = 0

        return result

    def plan(self, start, goal):
        """
        A* 경로 탐색

        Args:
            start: (x, y) 시작 격자 좌표
            goal: (x, y) 목표 격자 좌표

        Returns:
            path: [(x, y), ...] 경로 리스트, 실패 시 None
        """
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])

        # 범위 확인
        if not self._is_valid(sx, sy) or not self._is_valid(gx, gy):
            print("[PathPlanner] 시작 또는 목표가 맵 범위 밖")
            return None

        # 목표가 장애물 위인지 확인
        if self.inflated_grid[gy, gx] < 64:
            print("[PathPlanner] 목표 지점이 장애물 위입니다")
            return None

        # A* 알고리즘
        open_set = []
        heapq.heappush(open_set, (0, sx, sy))

        came_from = {}
        g_score = np.full((self.map_size, self.map_size), np.inf)
        g_score[sy, sx] = 0

        f_score = np.full((self.map_size, self.map_size), np.inf)
        f_score[sy, sx] = self._heuristic(sx, sy, gx, gy)

        closed_set = set()

        # 8방향 이동
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]

        iterations = 0
        max_iterations = self.map_size * self.map_size

        while open_set and iterations < max_iterations:
            iterations += 1
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) in closed_set:
                continue
            closed_set.add((cx, cy))

            # 목표 도달
            if cx == gx and cy == gy:
                return self._reconstruct_path(came_from, gx, gy)

            for dx, dy, cost in neighbors:
                nx, ny = cx + dx, cy + dy

                if not self._is_valid(nx, ny):
                    continue
                if (nx, ny) in closed_set:
                    continue
                if self.inflated_grid[ny, nx] < 64:  # 장애물
                    continue

                # 미탐사 지역은 약간의 페널티
                move_cost = cost
                if self.inflated_grid[ny, nx] == 128:
                    move_cost *= 1.5

                tentative_g = g_score[cy, cx] + move_cost

                if tentative_g < g_score[ny, nx]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[ny, nx] = tentative_g
                    f = tentative_g + self._heuristic(nx, ny, gx, gy)
                    f_score[ny, nx] = f
                    heapq.heappush(open_set, (f, nx, ny))

        print(f"[PathPlanner] 경로를 찾지 못했습니다 (iterations: {iterations})")
        return None

    def _heuristic(self, x1, y1, x2, y2):
        """유클리드 거리 휴리스틱"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _is_valid(self, x, y):
        """격자 좌표가 맵 범위 내인지"""
        return 0 <= x < self.map_size and 0 <= y < self.map_size

    def _reconstruct_path(self, came_from, gx, gy):
        """경로 역추적"""
        path = [(gx, gy)]
        current = (gx, gy)
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # 경로 단순화 (웨이포인트 줄이기)
        return self._simplify_path(path)

    def _simplify_path(self, path, step=5):
        """경로를 일정 간격으로 샘플링하여 단순화"""
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        for i in range(step, len(path) - 1, step):
            simplified.append(path[i])
        simplified.append(path[-1])
        return simplified


class PathFollower:
    """
    경로 추종 컨트롤러
    - 웨이포인트를 따라 로봇을 이동시키는 명령 생성
    - 아두이노에 W/A/S/D/Q 명령 전송
    """

    def __init__(self, motor: MotorController, slam_mapper):
        """
        Args:
            motor: MotorController 인스턴스
            slam_mapper: SLAMMapper 인스턴스 (현재 위치 조회용)
        """
        self.motor = motor
        self.slam = slam_mapper
        self.running = False
        self.current_path = None
        self.waypoint_idx = 0
        self.reached_goal = False

        # 추종 파라미터
        self.arrive_threshold = 3  # 웨이포인트 도달 판정 (격자 셀 수)
        self.angle_threshold = 0.3  # 방향 전환 임계값 (라디안, ~17도)
        self.speed = 150

    def set_path(self, path):
        """새로운 경로 설정"""
        self.current_path = path
        self.waypoint_idx = 0
        self.reached_goal = False
        print(f"[PathFollower] 경로 설정: {len(path)}개 웨이포인트")

    def follow(self):
        """
        경로 추종 루프 실행 (별도 스레드에서 호출)
        """
        self.running = True
        print("[PathFollower] 경로 추종 시작")

        while self.running and not self.reached_goal:
            if self.current_path is None or self.waypoint_idx >= len(self.current_path):
                self.reached_goal = True
                self.motor.stop()
                print("[PathFollower] 목적지 도달!")
                break

            # 현재 위치
            rx, ry, rtheta = self.slam.get_robot_position()

            # 다음 웨이포인트
            wx, wy = self.current_path[self.waypoint_idx]

            # 거리 계산
            dx = wx - rx
            dy = wy - ry
            distance = math.sqrt(dx**2 + dy**2)

            # 웨이포인트 도달 판정
            if distance < self.arrive_threshold:
                self.waypoint_idx += 1
                print(f"[PathFollower] 웨이포인트 {self.waypoint_idx}/{len(self.current_path)} 도달")
                continue

            # 목표 방향 계산
            target_angle = math.atan2(dy, dx)

            # 현재 방향과의 차이
            angle_diff = target_angle - rtheta
            # -π ~ π 정규화
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            # 조향 결정
            if abs(angle_diff) > self.angle_threshold:
                # 방향 전환 필요
                if angle_diff > 0:
                    self.motor.send_command('A', self.speed)  # 좌회전
                else:
                    self.motor.send_command('D', self.speed)  # 우회전
                time.sleep(0.2)
                self.motor.send_command('Q', 0)  # 정지
                time.sleep(0.1)
            else:
                # 직진
                self.motor.send_command('W', self.speed)
                time.sleep(0.3)
                self.motor.send_command('Q', 0)
                time.sleep(0.1)

            # 라이다 업데이트 (위치 추적 유지)
            distances = self.slam.lidar.read_frame()
            if distances is not None:
                self.slam.update_map(distances)

            time.sleep(0.05)

        self.motor.stop()
        print("[PathFollower] 추종 종료")

    def stop(self):
        """경로 추종 중지"""
        self.running = False
        self.motor.send_command('Q', 0)

    def get_status(self):
        """현재 상태 반환"""
        return {
            'running': self.running,
            'reached_goal': self.reached_goal,
            'waypoint_idx': self.waypoint_idx,
            'total_waypoints': len(self.current_path) if self.current_path else 0
        }
