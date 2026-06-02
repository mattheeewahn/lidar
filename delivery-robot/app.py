"""
Flask 웹서버 - 배달로봇 통합 컨트롤러
- 수동 조종 (W/A/S/D/Q)
- 카메라 스트리밍
- SLAM 매핑
- 지도 클릭 → 자율 주행
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request, Response
from motor_controller import MotorController
from slam_mapper import SLAMMapper
from path_planner import PathPlanner, PathFollower
import threading
import numpy as np
import cv2
import time
import base64
import math

app = Flask(__name__)

# ─── 전역 객체 ───
motor = MotorController(port='/dev/ttyACM0', baudrate=9600)
slam = SLAMMapper(map_size=200, resolution=0.05)
path_follower = None
camera = None

# 상태
state = {
    'mode': 'manual',           # manual / mapping / navigating
    'mapping_progress': 0,
    'has_map': False,
    'path': None,
    'goal': None,
}

# 스레드 락
state_lock = threading.Lock()


# ─── 카메라 스트리밍 ───
def get_camera():
    """카메라 초기화 (싱글톤)"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def generate_frames():
    """카메라 프레임 생성기 (MJPEG 스트리밍)"""
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30fps


# ─── 라우트 ───

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """카메라 MJPEG 스트리밍"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ─── 수동 조종 ───

@app.route('/control', methods=['POST'])
def control():
    """수동 모드 조종 명령 (아두이노에 W/A/S/D/Q 전송)"""
    if state['mode'] != 'manual':
        return jsonify({'error': '현재 자율주행 중입니다'}), 400

    data = request.json
    cmd = data.get('cmd', 'Q')
    speed = data.get('speed', 150)

    # 아두이노가 받는 명령: W, A, S, D, Q
    valid_cmds = ['W', 'A', 'S', 'D', 'Q']
    cmd = cmd.upper()
    if cmd not in valid_cmds:
        return jsonify({'error': f'유효하지 않은 명령: {cmd}'}), 400

    motor.send_command(cmd, speed)
    return jsonify({'status': 'ok', 'cmd': cmd, 'speed': speed})


@app.route('/stop', methods=['POST'])
def emergency_stop():
    """비상 정지"""
    global path_follower

    with state_lock:
        if path_follower:
            path_follower.stop()
            path_follower = None
        slam.stop()
        motor.send_command('Q', 0)
        state['mode'] = 'manual'
        state['path'] = None
        state['goal'] = None

    return jsonify({'status': 'stopped', 'mode': 'manual'})


# ─── SLAM 매핑 ───

@app.route('/start_mapping', methods=['POST'])
def start_mapping():
    """SLAM 매핑 시작"""
    global slam

    if state['mode'] != 'manual':
        return jsonify({'error': '수동 모드에서만 매핑 가능'}), 400

    duration = request.json.get('duration', 60)

    with state_lock:
        state['mode'] = 'mapping'
        state['mapping_progress'] = 0

    # 매핑을 별도 스레드에서 실행
    def mapping_thread():
        try:
            slam.connect()
            slam.running = True
            start_time = time.time()

            while slam.running and (time.time() - start_time) < duration:
                distances = slam.lidar.read_frame()
                if distances is not None:
                    slam.update_map(distances)

                elapsed = time.time() - start_time
                with state_lock:
                    state['mapping_progress'] = min(100, int(elapsed / duration * 100))

                time.sleep(0.05)

            with state_lock:
                state['mode'] = 'manual'
                state['has_map'] = True
                state['mapping_progress'] = 100

            print("[App] 매핑 완료!")

        except Exception as e:
            print(f"[App] 매핑 오류: {e}")
            with state_lock:
                state['mode'] = 'manual'

    t = threading.Thread(target=mapping_thread, daemon=True)
    t.start()

    return jsonify({'status': 'mapping_started', 'duration': duration})


@app.route('/stop_mapping', methods=['POST'])
def stop_mapping():
    """매핑 중지"""
    slam.stop()
    with state_lock:
        state['mode'] = 'manual'
        state['has_map'] = True
    return jsonify({'status': 'mapping_stopped'})


@app.route('/get_map')
def get_map():
    """현재 맵 이미지 반환 (PNG base64)"""
    grid = slam.get_occupancy_grid()
    rx, ry, rtheta = slam.get_robot_position()

    # 컬러 맵 생성
    color_map = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    color_map[grid == 0] = [40, 40, 40]       # 장애물: 어두운 회색
    color_map[grid == 128] = [100, 100, 100]   # 미탐사: 중간 회색
    color_map[grid == 255] = [220, 220, 220]   # 빈 공간: 밝은 회색

    # 로봇 위치 표시 (파란색 원)
    cv2.circle(color_map, (int(rx), int(ry)), 3, (255, 100, 0), -1)

    # 로봇 방향 표시 (화살표)
    arrow_len = 8
    ax = int(rx + arrow_len * math.cos(rtheta))
    ay = int(ry + arrow_len * math.sin(rtheta))
    cv2.arrowedLine(color_map, (int(rx), int(ry)), (ax, ay), (255, 100, 0), 1)

    # 경로 표시 (초록색)
    if state['path']:
        for i in range(len(state['path']) - 1):
            p1 = state['path'][i]
            p2 = state['path'][i + 1]
            cv2.line(color_map, (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])), (0, 255, 0), 1)

    # 목적지 표시 (빨간색)
    if state['goal']:
        cv2.circle(color_map, (int(state['goal'][0]), int(state['goal'][1])),
                   4, (0, 0, 255), -1)

    # 궤적 표시 (파란 점선)
    for tx, ty, _ in slam.trajectory[-100:]:
        cv2.circle(color_map, (int(tx), int(ty)), 1, (255, 150, 50), -1)

    # 이미지를 확대 (200x200 → 600x600)
    color_map = cv2.resize(color_map, (600, 600), interpolation=cv2.INTER_NEAREST)

    # PNG → base64
    _, buffer = cv2.imencode('.png', color_map)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': img_base64,
        'robot': {'x': rx, 'y': ry, 'theta': rtheta},
        'map_size': slam.map_size,
        'resolution': slam.resolution,
        'has_path': state['path'] is not None,
        'goal': state['goal'],
    })


# ─── 자율 주행 (목적지 클릭) ───

@app.route('/navigate', methods=['POST'])
def navigate():
    """목적지 설정 및 자율 주행 시작"""
    global path_follower

    if not state['has_map']:
        return jsonify({'error': '먼저 매핑을 수행하세요'}), 400

    data = request.json
    # 클라이언트에서 보내는 좌표는 600x600 이미지 기준 → 200x200 격자로 변환
    goal_x = data.get('x', 0) * slam.map_size / 600.0
    goal_y = data.get('y', 0) * slam.map_size / 600.0

    # 로봇 현재 위치
    rx, ry, _ = slam.get_robot_position()

    # 경로 계획
    grid = slam.get_occupancy_grid()
    planner = PathPlanner(grid, slam.resolution)
    path = planner.plan((rx, ry), (goal_x, goal_y))

    if path is None:
        return jsonify({'error': '경로를 찾을 수 없습니다'}), 400

    with state_lock:
        state['path'] = path
        state['goal'] = (goal_x, goal_y)
        state['mode'] = 'navigating'

    # 경로 추종 시작
    path_follower = PathFollower(motor, slam)
    path_follower.set_path(path)

    def nav_thread():
        try:
            path_follower.follow()
        except Exception as e:
            print(f"[App] 네비게이션 오류: {e}")
        finally:
            with state_lock:
                state['mode'] = 'manual'
            print("[App] 네비게이션 완료, 수동 모드로 전환")

    t = threading.Thread(target=nav_thread, daemon=True)
    t.start()

    return jsonify({
        'status': 'navigating',
        'goal': {'x': goal_x, 'y': goal_y},
        'path_length': len(path)
    })


@app.route('/nav_status')
def nav_status():
    """네비게이션 상태 확인"""
    if path_follower:
        status = path_follower.get_status()
    else:
        status = {'running': False, 'reached_goal': False, 'waypoint_idx': 0, 'total_waypoints': 0}

    return jsonify({
        'mode': state['mode'],
        'mapping_progress': state['mapping_progress'],
        'has_map': state['has_map'],
        **status
    })


@app.route('/status')
def status_route():
    """전체 상태"""
    return jsonify(state)


# ─── 메인 ───

import math

if __name__ == '__main__':
    motor.connect()
    print("=" * 50)
    print("  배달로봇 서버 시작")
    print("  http://0.0.0.0:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, threaded=True)
