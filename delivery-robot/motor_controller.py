"""
아두이노 메가와 시리얼 통신으로 모터 제어
- 라즈베리파이 → 아두이노 메가 → BTS7960 x4 → 모터 4개
- 아두이노는 단순 문자 명령 수신: W(전진), A(좌회전), S(후진), D(우회전), Q(정지)
"""

import serial
import time


class MotorController:
    """
    아두이노에 단일 문자 명령 전송
    
    프로토콜: 단일 문자 전송
    - W = 전진 (Forward)
    - S = 후진 (Backward)
    - A = 좌회전 (Turn Left)
    - D = 우회전 (Turn Right)
    - Q = 정지 (Stop)
    """
    
    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        
    def connect(self):
        """아두이노 시리얼 연결"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # 아두이노 리셋 대기
            print(f"[Motor] 아두이노 연결됨: {self.port}")
        except serial.SerialException as e:
            print(f"[Motor] 연결 실패: {e}")
            self.serial = None
        
    def send_command(self, cmd, speed=150):
        """
        아두이노에 명령 전송
        
        Args:
            cmd: 명령어 문자 (W, A, S, D, Q)
            speed: 미사용 (아두이노 코드에서 고정 속도 사용)
        """
        if self.serial is None:
            print("[Motor] 시리얼 미연결")
            return
            
        # 유효한 명령만 전송
        valid_cmds = ['W', 'A', 'S', 'D', 'Q']
        cmd = cmd.upper()
        if cmd not in valid_cmds:
            print(f"[Motor] 유효하지 않은 명령: {cmd}")
            return
            
        self.serial.write(cmd.encode())
        
    def forward(self, speed=150):
        """전진"""
        self.send_command('W')
        
    def backward(self, speed=150):
        """후진"""
        self.send_command('S')
        
    def turn_left(self, speed=120):
        """좌회전"""
        self.send_command('A')
        
    def turn_right(self, speed=120):
        """우회전"""
        self.send_command('D')
        
    def stop(self):
        """정지"""
        self.send_command('Q')
        
    def close(self):
        """연결 종료"""
        self.stop()
        if self.serial:
            self.serial.close()
            print("[Motor] 연결 종료")
