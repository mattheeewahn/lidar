import serial
import time

port = '/dev/ttyUSB0'

for baud in [3000000, 115200, 230400, 256000, 500000, 921600]:
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=2)
        time.sleep(0.3)
        CMD = bytes([0x5A, 0x77, 0xFF, 0x02, 0x00, 0x01, 0x00, 0x03])
        ser.write(CMD)
        time.sleep(1)
        avail = ser.in_waiting
        print(f"보드레이트 {baud}: {avail} 바이트 수신")
        if avail > 0:
            raw = ser.read(min(avail, 50))
            print(f"  데이터: {raw.hex()}")
        ser.close()
    except Exception as e:
        print(f"보드레이트 {baud}: 에러 - {e}")
