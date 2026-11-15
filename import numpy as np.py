import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==========================================
# 1. KHỞI TẠO THÔNG SỐ CƠ KHÍ CHUẨN
# ==========================================
L4 = 10.0          # Trụ trung tâm cao 10cm
H_SERVO = 2.5      # Chiều cao trục quay Servo = 2.5cm
L1 = 1.5           # Độ dài tay đòn Servo = 1.5cm
HF = 1.0           # Đoạn đế nối từ tấm xuống khớp trên = 1cm

# Tọa độ Servo & Điểm nối trên tấm
Ox, Oy = 4.0, 3.0   # Khoảng cách từ trụ L4 tới 2 Servo
L3x, L3y = 6.0, 4.0 # Khoảng cách từ trụ L4 tới 2 điểm nối trên tấm

# Độ dài thực tế thanh truyền L2 (Mặt cắt 2D chuẩn)
# L2x = sqrt((6 - (4 - 1.5))^2 + (9 - 2.5)^2) = sqrt(3.5^2 + 6.5^2)
L2x = np.sqrt((L3x - (Ox - L1))**2 + ((L4 - HF) - H_SERVO)**2)
L2y = np.sqrt((L3y - (Oy - L1))**2 + ((L4 - HF) - H_SERVO)**2)

# Kích thước tấm nền (Length X = 16.5cm, Width Y = 10cm)
W_plate, H_plate = 16.5, 10.0 

# ==========================================
# 2. HÀM GIẢI INVERSE KINEMATICS (IK)
# ==========================================
def solve_ik(theta_x_deg, theta_y_deg):
    th_x = np.radians(theta_x_deg) # Pitch (nghiêng quanh Y)
    th_y = np.radians(theta_y_deg) # Roll  (nghiêng quanh X)
    
    # 1. Tọa độ đỉnh trụ trung tâm B
    B = np.array([0.0, 0.0, L4])
    
    # 2. Tọa độ khớp trên Hx, Hy (khi tấm nghiêng)
    # Trục X: nghiêng theo theta_y
    Hx = np.array([L3x * np.cos(th_y), 0.0, (L4 - HF) + L3x * np.sin(th_y)])
    # Trục Y: nghiêng theo theta_x
    Hy = np.array([0.0, L3y * np.cos(th_x), (L4 - HF) + L3y * np.sin(th_x)])
    
    # 3. Vị trí trục Servo Sx, Sy
    Sx = np.array([Ox, 0.0, H_SERVO])
    Sy = np.array([0.0, Oy, H_SERVO])
    
    # 4. Giải góc alpha_x (Servo X) - Tay đòn L1 QUAY VÀO TRONG (hướng về -X)
    # Dx = [Ox - L1*cos(alpha), 0, H_SERVO + L1*sin(alpha)]
    # Khoảng cách |Hx - Dx| = L2x
    # PT: (Hx_x - Ox + L1*cos(a))^2 + (Hx_z - Sz - L1*sin(a))^2 = L2x^2
    K1 = Hx[0] - Sx[0]
    K2 = Hx[2] - Sx[2]
    
    # Duyệt góc alpha từ -60 đến 60 độ để tìm nghiệm chính xác
    alphas = np.linspace(np.radians(-60), np.radians(60), 300)
    best_ax, min_err_x = 0.0, 1e6
    for a in alphas:
        Dx_test = np.array([Sx[0] - L1 * np.cos(a), 0.0, Sx[2] + L1 * np.sin(a)])
        err = abs(np.linalg.norm(Hx - Dx_test) - L2x)
        if err < min_err_x:
            min_err_x = err
            best_ax = a
            
    # 5. Giải góc alpha_y (Servo Y) - Tay đòn L1 QUAY VÀO TRONG (hướng về -Y)
    best_ay, min_err_y = 0.0, 1e6
    for a in alphas:
        Dy_test = np.array([0.0, Sy[1] - L1 * np.cos(a), Sy[2] + L1 * np.sin(a)])
        err = abs(np.linalg.norm(Hy - Dy_test) - L2y)
        if err < min_err_y:
            min_err_y = err
            best_ay = a
            
    Dx = np.array([Sx[0] - L1 * np.cos(best_ax), 0.0, Sx[2] + L1 * np.sin(best_ax)])
    Dy = np.array([0.0, Sy[1] - L1 * np.cos(best_ay), Sy[2] + L1 * np.sin(best_ay)])
    
    return B, Hx, Hy, Sx, Sy, Dx, Dy, np.degrees(best_ax), np.degrees(best_ay)

# ==========================================
# 3. VẼ MÔ PHỎNG 3D VỚI SLIDER ĐIỀU KHIỂN
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Slider điều khiển góc nghiêng Pitch/Roll
ax_pitch = plt.axes([0.25, 0.1, 0.5, 0.03])
ax_roll  = plt.axes([0.25, 0.05, 0.5, 0.03])
s_pitch = Slider(ax_pitch, 'Góc Pitch (trục X)', -15.0, 15.0, valinit=0.0)
s_roll  = Slider(ax_roll,  'Góc Roll (trục Y)',  -15.0, 15.0, valinit=0.0)

def update(val):
    ax.clear()
    th_x = s_pitch.val
    th_y = s_roll.val
    
    B, Hx, Hy, Sx, Sy, Dx, Dy, ax_deg, ay_deg = solve_ik(th_x, th_y)
    
    # 1. Trụ đứng trung tâm L4
    ax.plot([0, 0], [0, 0], [0, L4], 'k-', lw=5, label='Trụ giữa L4 (10cm)')
    
    # 2. Nhánh cơ khí Servo X (Màu đỏ)
    ax.plot([Sx[0], Sx[0]], [0, 0], [0, Sx[2]], 'r--', lw=2) # Đế Servo
    ax.plot([Sx[0], Dx[0]], [Sx[1], Dx[1]], [Sx[2], Dx[2]], 'r-o', lw=4, label=f'Servo X L1 (quay vào trong): {ax_deg:.1f}°')
    ax.plot([Dx[0], Hx[0]], [Dx[1], Hx[1]], [Dx[2], Hx[2]], 'g-o', lw=3, label='Thanh L2_x')
    ax.plot([Hx[0], Hx[0]], [Hx[1], Hx[1]], [Hx[2], Hx[2]+HF], 'k-', lw=2) # Khớp HF
    
    # 3. Nhánh cơ khí Servo Y (Màu xanh dương)
    ax.plot([0, 0], [Sy[1], Sy[1]], [0, Sy[2]], 'b--', lw=2) # Đế Servo
    ax.plot([Sy[0], Dy[0]], [Sy[1], Dy[1]], [Sy[2], Dy[2]], 'b-o', lw=4, label=f'Servo Y L1 (quay vào trong): {ay_deg:.1f}°')
    ax.plot([Dy[0], Hy[0]], [Dy[1], Hy[1]], [Dy[2], Hy[2]], 'm-o', lw=3, label='Thanh L2_y')
    ax.plot([Hy[0], Hy[0]], [Hy[1], Hy[1]], [Hy[2], Hy[2]+HF], 'k-', lw=2) # Khớp HF
    
    # 4. Mặt tấm nghiêng (16.5 x 10 cm)
    px = np.array([-W_plate/2, W_plate/2, W_plate/2, -W_plate/2, -W_plate/2])
    py = np.array([-H_plate/2, -H_plate/2, H_plate/2, H_plate/2, -H_plate/2])
    pz = np.zeros_like(px)
    
    # Xoay mặt tấm theo góc theta_x, theta_y
    rad_x, rad_y = np.radians(th_x), np.radians(th_y)
    # Ma trận xoay đơn giản cho tấm
    pz_rot = px * np.sin(rad_y) + py * np.sin(rad_x) + L4
    px_rot = px * np.cos(rad_y)
    py_rot = py * np.cos(rad_x)
    
    ax.plot(px_rot, py_rot, pz_rot, 'gray', lw=2)
    
    # Cấu hình không gian 3D
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([0, 12])
    ax.set_xlabel('Trục X (cm)')
    ax.set_ylabel('Trục Y (cm)')
    ax.set_zlabel('Trục Z (cm)')
    ax.set_title('Mô Phỏng 3D Ball On Plate (Tay đòn L1 quay vào trong)')
    ax.legend(loc='upper left', fontsize=8)
    fig.canvas.draw_idle()

s_pitch.on_changed(update)
s_roll.on_changed(update)

# Vẽ lần đầu
update(0)
plt.show()