"""
ball_plate_bridge.py  -  cau noi Camera/OpenCV  ->  ESP32-S3 (Serial)

Dung lai nguyen xi nhan dien trong ball_on_plate_opencv_v6.py (de CUNG THU MUC),
khong sua file do. Script nay chi them:
  * gui toa do bi xuong ESP32 qua Serial  (D,seq,t_ms,x,y  /  I,seq khi mat bi/tam)
  * doc telemetry tu ESP32, ghi log CSV
  * phim tat + click chuot de dat diem dat, go lenh truc tiep vao ESP32 (KP 0.5 ...)

Cai dat:   pip install pyserial opencv-python numpy
Chay:      python ball_plate_bridge.py --port COM5
           python ball_plate_bridge.py --no-serial      (chi xem nhan dien)

Phim (cua so camera/plate dang focus):
  a = ARM (bat dieu khien)     s hoac SPACE = DISARM (tam ve phang)
  c = diem dat ve tam          click chuot len cua so "Plate" = dat diem dat
  o = bat/tat quy dao tron (giai doan 2)
  r = tim lai tam              q = thoat
Go lenh o console (vi du):  KP 0.5   KD 0.35   SAVE   ?
"""
import argparse
import csv
import math
import sys
import threading
import time

import cv2
import numpy as np

import ball_on_plate_opencv_v7 as bp

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None


# --------------------------------------------------------------------------------
#  Serial
# --------------------------------------------------------------------------------
def find_port():
    ports = list(list_ports.comports())
    for p in ports:                                   # Espressif native USB
        if p.vid == 0x303A:
            return p.device
    for p in ports:                                   # CP210x / CH340 / FTDI
        if p.vid in (0x10C4, 0x1A86, 0x0403):
            return p.device
    if len(ports) == 1:
        return ports[0].device
    return None


class Link:
    """Gui lenh + doc telemetry o thread rieng."""

    def __init__(self, port, baud):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0.05
        self.ser.dtr = False          # tranh reset ESP32 khi mo cong
        self.ser.rts = False
        self.ser.open()
        self.lock = threading.Lock()
        self.status = None            # idx: 0 state, 1 esp_ms, 2 x, 3 y, 4 vx, 5 vy, 6 ux, 7 uy, 8 ix, 9 iy
        self.running = True
        self.ok = True
        threading.Thread(target=self._rx, daemon=True).start()

    def send(self, text):
        if not self.ok:
            return
        try:
            with self.lock:
                self.ser.write((text + "\n").encode("ascii", "ignore"))
        except Exception as e:                        # rut cap USB...
            self.ok = False
            print("Mat ket noi Serial:", e)

    def _rx(self):
        buf = b""
        while self.running:
            try:
                data = self.ser.read(256)
            except Exception:
                self.ok = False
                break
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode("ascii", "ignore").strip()
                if not s:
                    continue
                if s.startswith("S,"):
                    try:
                        f = s.split(",")
                        self.status = (int(f[1]), int(f[2])) + tuple(float(v) for v in f[3:11])
                    except (ValueError, IndexError):
                        pass
                else:
                    print("ESP32:", s)

    def close(self):
        self.running = False
        try:
            self.send("DISARM")
            time.sleep(0.05)
            self.ser.close()
        except Exception:
            pass


# --------------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------------
STATE_NAME = {0: "DISARM", 1: "RUN", 2: "ARM-NO BALL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="vd COM5 hoac /dev/ttyACM0 (bo trong = tu tim)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cam", type=int, default=bp.CAM_INDEX)
    ap.add_argument("--no-serial", action="store_true", help="chi chay nhan dien, khong gui")
    ap.add_argument("--traj-r", type=float, default=2.5, help="ban kinh quy dao tron (cm)")
    ap.add_argument("--traj-t", type=float, default=12.0, help="chu ky quy dao tron (s)")
    args = ap.parse_args()

    W, H = bp.PLATE_W_CM, bp.PLATE_H_CM
    PPC = bp.PIXELS_PER_CM

    link = None
    if not args.no_serial:
        if serial is None:
            print("Thieu pyserial:  pip install pyserial")
            return
        port = args.port or find_port()
        if port is None:
            print("Khong tu tim duoc cong. Dung --port COMx (xem Device Manager).")
            return
        try:
            link = Link(port, args.baud)
        except Exception as e:
            print(f"Khong mo duoc {port}: {e}")
            return
        print(f"Da mo {port} @ {args.baud}")
        time.sleep(1.0)
        link.send("DISARM")

    stream = bp.CameraStream(args.cam)
    if not stream.running:
        print("Khong mo duoc camera. Doi --cam 0/1/2 ...")
        if link:
            link.close()
        return

    # diem dat + quy dao
    sp = [W / 2.0, H / 2.0]
    sp_dirty = True
    traj_on, traj_t0 = False, 0.0

    PLATE_WIN = "Plate top view"
    cv2.namedWindow(PLATE_WIN)

    def on_mouse(event, mx, my, flags, param):
        nonlocal sp_dirty
        if event == cv2.EVENT_LBUTTONDOWN and not traj_on:
            sp[0] = float(np.clip(mx / PPC, 1.5, W - 1.5))
            sp[1] = float(np.clip(my / PPC, 1.5, H - 1.5))
            sp_dirty = True

    cv2.setMouseCallback(PLATE_WIN, on_mouse)

    # console: go lenh thang vao ESP32
    if link is not None:
        def console():
            for line in sys.stdin:
                line = line.strip()
                if line:
                    link.send(line)
        threading.Thread(target=console, daemon=True).start()

    log_name = time.strftime("log_%Y%m%d_%H%M%S.csv")
    logf = open(log_name, "w", newline="")
    logw = csv.writer(logf)
    logw.writerow(["t_s", "valid", "x_cm", "y_cm", "sp_x", "sp_y", "esp_state", "ux_deg", "uy_deg", "ix_deg", "iy_deg"])

    tracker = bp.PlateTracker()
    last_seq, ball_prev, ball_miss = 0, None, 0
    ball_r_cm = bp.BALL_DIAMETER_MM / 20.0     # hoc ban kinh bi that (cm) - dong bo voi v6
    seq, last_sp_send, last_print = 0, 0.0, 0.0
    fps, last_tick = 0.0, time.time()
    t_start = time.perf_counter()

    print("a=ARM  s/SPACE=DISARM  c=tam  o=quy dao tron  r=tim lai tam  q=thoat | click len cua so Plate = dat diem dat")
    print("Log:", log_name)

    while True:
        frame, last_seq = stream.read(last_seq)
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.001)
            continue

        t_cap = time.perf_counter()                   # moc thoi gian chup anh
        now = time.time()
        dt = now - last_tick
        last_tick = now
        if dt > 0:
            fps = 1.0 / dt if fps == 0 else 0.85 * fps + 0.15 / dt

        tracker.update(frame)
        q = tracker.quad
        ball = None
        ball_state = "LOST"
        # v6 co them state TRK2 / PRED / HOLD va buoc cuu bang find_ball_relaxed
        if q is not None and tracker.state in ("TRACK", "TRK2", "PRED", "MARKS", "LOCK"):
            ball = bp.find_ball_cam(tracker.D, tracker.hsv, q, ball_prev, ball_r_cm)
            ball_state = "OK"
            if ball is None and ball_prev is not None:      # bi o goc / bi cat: cuu quanh vi tri cu
                ball = bp.find_ball_relaxed(tracker.D, tracker.hsv, q, ball_prev, ball_r_cm)
                ball_state = "REC"

        # ----- toa do tho (KHONG lam muot o day: bo loc alpha-beta tren ESP32 lo viec do) -----
        valid, x_cm, y_cm = False, 0.0, 0.0
        if ball is not None:
            cx, cy, r_s, clipped = ball                     # v6 tra ve 4 gia tri
            raw = bp.to_canon(q, [[cx, cy]])[0]
            if -0.5 <= raw[0] <= W + 0.5 and -0.5 <= raw[1] <= H + 0.5:
                valid, x_cm, y_cm = True, float(raw[0]), float(raw[1])
                ball_prev, ball_miss = (cx, cy), 0
                if not clipped:                             # hoc ban kinh bi that khi thay tron ven
                    cm_px_b = np.linalg.norm(q[1] - q[0]) / W
                    ball_r_cm = 0.9 * ball_r_cm + 0.1 * (r_s / cm_px_b)
        if not valid:
            ball_miss += 1
            if ball_miss > bp.BALL_MISS_MAX:
                ball_prev = None

        # ----- quy dao / diem dat -----
        t_ms = int((t_cap - t_start) * 1000.0) & 0xFFFFFFFF
        if traj_on:
            w = 2.0 * math.pi / args.traj_t
            ph = w * (t_cap - traj_t0)
            sp[0] = W / 2.0 + args.traj_r * math.cos(ph)
            sp[1] = H / 2.0 + args.traj_r * math.sin(ph)
            vref = (-args.traj_r * w * math.sin(ph), args.traj_r * w * math.cos(ph))
            if link:
                link.send(f"SP,{sp[0]:.3f},{sp[1]:.3f},{vref[0]:.3f},{vref[1]:.3f}")
        elif link and (sp_dirty or t_cap - last_sp_send > 1.0):
            link.send(f"SP,{sp[0]:.3f},{sp[1]:.3f}")
            sp_dirty, last_sp_send = False, t_cap

        # ----- gui du lieu do -----
        if link:
            if valid:
                link.send(f"D,{seq},{t_ms},{x_cm:.3f},{y_cm:.3f}")
            else:
                link.send(f"I,{seq}")
        seq += 1

        st = link.status if link else None
        logw.writerow([f"{t_cap - t_start:.3f}", int(valid), f"{x_cm:.3f}", f"{y_cm:.3f}",
                       f"{sp[0]:.3f}", f"{sp[1]:.3f}"] +
                      ([st[0], st[6], st[7], st[8], st[9]] if st else ["", "", "", "", ""]))

        # ----- hien thi -----
        display = frame.copy()
        corners = tracker.corners_full()
        plate_view = None
        if corners is not None:
            bp.draw_plate(display, corners)
            plate_view = bp.rectify_plate(frame, corners)
            sx, sy = int(sp[0] * PPC), int(sp[1] * PPC)
            cv2.drawMarker(plate_view, (sx, sy), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
            if valid:
                p = (int(x_cm * PPC), int(y_cm * PPC))
                cv2.circle(plate_view, p, int(bp.BALL_DIAMETER_MM / 20.0 * PPC), (0, 255, 0), 2)
                cv2.circle(plate_view, p, 4, (0, 0, 255), -1)
                cv2.putText(plate_view, f"X={x_cm:.2f} Y={y_cm:.2f} cm", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            else:
                cv2.putText(plate_view, f"BALL NOT FOUND ({ball_state})", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "KHONG TIM THAY TAM", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        line = f"{tracker.state} FPS:{fps:.0f} ball:{ball_state}"
        if st:
            line += f"  ESP:{STATE_NAME.get(st[0], '?')} u=({st[6]:+.1f},{st[7]:+.1f})deg"
        elif link is None:
            line += "  (no-serial)"
        if traj_on:
            line += "  TRAJ"
        cv2.putText(display, line, (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.imshow("Camera - plate detection", display)
        if plate_view is not None:
            cv2.imshow(PLATE_WIN, plate_view)

        if t_cap - last_print > 1.0 and st:
            print(f"[{STATE_NAME.get(st[0], '?')}] ball=({x_cm:.2f},{y_cm:.2f}) v=({st[4]:+.1f},{st[5]:+.1f}) cm/s "
                  f"u=({st[6]:+.2f},{st[7]:+.2f}) I=({st[8]:+.2f},{st[9]:+.2f}) deg  fps={fps:.0f}")
            last_print = t_cap

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a") and link:
            link.send("ARM")
        elif key in (ord("s"), 32) and link:
            link.send("DISARM")
        elif key == ord("c"):
            traj_on = False
            sp[0], sp[1], sp_dirty = W / 2.0, H / 2.0, True
        elif key == ord("o"):
            traj_on = not traj_on
            traj_t0 = time.perf_counter()
            if not traj_on:
                sp[0], sp[1], sp_dirty = W / 2.0, H / 2.0, True
                if link:
                    link.send(f"SP,{sp[0]:.3f},{sp[1]:.3f}")
        elif key == ord("r"):
            tracker.reset()
            ball_prev = None

    stream.release()
    cv2.destroyAllWindows()
    logf.close()
    if link:
        link.close()


if __name__ == "__main__":
    main()