import serial
import struct
import time

port = '/dev/ttyUSB0'
baudrate = 3000000

print(f"연결 시도: {port}")
ser = serial.Serial(port=port, baudrate=baudrate, timeout=2)
time.sleep(0.5)

# 2D 스캔 시작 명령
CMD_START = bytes([0x5A, 0x77, 0xFF, 0x02, 0x00, 0x01, 0x00, 0x03])
ser.write(CMD_START)
print("스캔 시작 명령 전송")

time.sleep(1)

# 버퍼에 뭐가 있는지 확인
available = ser.in_waiting
print(f"수신 버퍼: {available} 바이트")

if available > 0:
    raw = ser.read(min(available, 100))
    print(f"원시 데이터: {raw.hex()}")
    print("라이다 데이터 수신 OK!")
else:
    print("데이터 없음! 라이다 연결/포트/보드레이트 확인 필요")

ser.close()
