"""
CygLiDAR D1 드라이버 (2D 모드)
"""

import serial
import struct
import time
import numpy as np


class CygLidarD1:
    HEADER = bytes([0x5A, 0x77, 0xFF])
    CMD_START_2D = bytes([0x5A, 0x77, 0xFF, 0x02, 0x00, 0x01, 0x00, 0x03])
    CMD_STOP = bytes([0x5A, 0x77, 0xFF, 0x02, 0x00, 0x02, 0x00, 0x02])

    def __init__(self, port='/dev/ttyUSB0', baudrate=3000000):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.num_points = 160

    def connect(self):
        self.serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=1
        )
        time.sleep(0.5)
        print(f"[LiDAR] 연결됨: {self.port}")

    def start_scan(self):
        self.serial.write(self.CMD_START_2D)
        print("[LiDAR] 2D 스캔 시작")

    def stop_scan(self):
        self.serial.write(self.CMD_STOP)
        print("[LiDAR] 스캔 중지")

    def read_frame(self):
        buf = bytearray()
        while True:
            byte = self.serial.read(1)
            if not byte:
                return None
            buf.append(byte[0])
            if len(buf) >= 3:
                if buf[-3:] == bytearray(self.HEADER):
                    break

        length_bytes = self.serial.read(2)
        if len(length_bytes) < 2:
            return None
        payload_length = struct.unpack('<H', length_bytes)[0]

        payload = self.serial.read(payload_length)
        if len(payload) < payload_length:
            return None

        if payload[0] != 0x01:
            return None

        distances = []
        data = payload[1:]
        for i in range(0, min(len(data), self.num_points * 2), 2):
            dist = struct.unpack('<H', data[i:i+2])[0]
            distances.append(dist)

        if len(distances) < self.num_points:
            return None

        return np.array(distances, dtype=np.float32)

    def get_angles(self):
        return np.linspace(0, 120, self.num_points)

    def close(self):
        if self.serial:
            self.stop_scan()
            self.serial.close()
            print("[LiDAR] 연결 종료")
