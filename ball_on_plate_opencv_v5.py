import cv2
import numpy as np
import time
import threading
import itertools

# ===================== CAU HINH =====================
CAM_INDEX = 2
PLATE_W_CM = 16.5
PLATE_H_CM = 10.0
BALL_DIAMETER_MM = 18.0
PIXELS_PER_CM = 40
WARP_W = int(PLATE_W_CM * PIXELS_PER_CM)   # 660
WARP_H = int(PLATE_H_CM * PIXELS_PER_CM)   # 400

# Bo cuc 3 hinh ve (nhin tu tren xuong, canh dai nam ngang):
#   TL = hinh TRON, BL = hinh TAM GIAC (dinh chu L), BR = hinh VUONG, TR = de trong
# Khoang cach tam hinh ve toi mep tam (cm) - chi dung de dat "hat giong", sai ~0.3 cm van duoc
MARK_INSET_CM = 0.8

DET_SCALE = 0.5            # xu ly tren anh 320x240
BLUE_S_MIN = 60            # nhay muc xanh (phim , / . de chinh)
MARK_MIN_AREA = 10
MARK_MAX_AREA = 900

WL = 0.5                   # trong so do sang so voi mau (giam anh huong bong do)
EDGE_THR = 14              # do tuong phan toi thieu giua giay va nen de coi la mep
EDGE_W = 3                 # cua so trung binh 2 phia mep (px anh nho)
N_SAMPLES = 20             # so diem do tren moi canh
T_LO, T_HI = 0.18, 0.82    # chi do o giua canh (tranh goc/hinh ve)
INIT_R = (14, 8)           # ban kinh tim mep khi khoi tao (2 vong, anh nho)
MARK_R = (7, 3)            # khi thay du 3 hinh ve: chi tim mep rat gan khung suy ra tu hinh ve
EDGE_THR_SEEDED = 8        # co hat giong tot thi cho phep mep yeu (nen trang gan giong giay)
MARK_TOL_CM = 0.6          # khung tim duoc phai khop vi tri hinh ve da hoc (cm)
MARK_SEARCH_CM = 1.3       # ban kinh tim hinh ve quanh vi tri du doan (cm)
MARK_MIN_PIX = 6
BALL_D_THR = 42            # bi = vung 'khac mau giay' rat manh (bong do va nep nhan khong vuot nguong)
TRACK_R = (10, 4)          # bam: vong rong (bat chuyen dong nhanh) roi vong hep (chinh xac)
RECOVER_R = (24, 12, 5)    # cuu: vua mat bam -> tim rong hon quanh vi tri du doan, chua can khoi tao lai
INTERIOR_TOL = 30          # noi that tam phai giong mau giay (khoang cach Lab)
ALPHA_STILL = 0.55         # dung yen: lam muot manh de het rung
ALPHA_MOVE = 1.0           # chuyen dong nhanh: bo lam muot de khong bi tre
VEL_GAIN = 0.7             # muc tin vao du doan van toc (0 = tat)
LOST_MAX = 15

# --- Bi ---
S_MAX = 70
BALL_SCALE = 0.5
BALL_PPCM = PIXELS_PER_CM * BALL_SCALE
BALL_R = (BALL_DIAMETER_MM / 10.0) * BALL_PPCM / 2.0
BALL_AREA = np.pi * BALL_R ** 2
BALL_MISS_MAX = 6
EMA_ALPHA = 0.45

# ====================================================
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
K11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
TW = np.array([[WL, 1.0, 1.0]], np.float32)
CANON = np.float32([[0, 0], [PLATE_W_CM, 0], [PLATE_W_CM, PLATE_H_CM], [0, PLATE_H_CM]])
_gu, _gv = np.meshgrid(np.linspace(0.12, 0.88, 16), np.linspace(0.12, 0.88, 10))
GRID = np.stack([_gu.ravel(), _gv.ravel()], axis=1).astype(np.float32)


class CameraStream:
    """Doc camera o thread rieng, luon giu frame moi nhat -> khong bi tre."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(index)
        self.running = self.cap.isOpened()
        self.frame = None
        self.seq = 0
        self.lock = threading.Lock()
        if self.running:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
                    self.seq += 1

    def read(self, last_seq):
        with self.lock:
            if self.seq == last_seq:
                return None, last_seq
            return self.frame, self.seq

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ------------------------------------------------------------------
#  TANG 1 (vi mo -> chi tiet): tim cac hinh ve xanh lam "neo"
# ------------------------------------------------------------------
def find_marks(blue):
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []
    for c in contours:
        a = cv2.contourArea(c)
        if not MARK_MIN_AREA <= a <= MARK_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) / max(1, min(w, h)) > 2.2 or a / float(w * h) < 0.35:
            continue
        m = cv2.moments(c)
        if m["m00"] < 1:
            continue
        marks.append((m["m10"] / m["m00"], m["m01"] / m["m00"], a, c))
    marks.sort(key=lambda t: t[2], reverse=True)
    return marks[:5]


def ring_medians(lab, blue, marks):
    """Mau giay (Lab) o vanh quanh TUNG hinh ve (None neu vanh khong du diem)."""
    blue_d = cv2.dilate(blue, K3)
    out = []
    for mk in marks:
        m = np.zeros(blue.shape, np.uint8)
        cv2.drawContours(m, [mk[3]], -1, 255, -1)
        ring = cv2.bitwise_and(cv2.dilate(m, K11), cv2.bitwise_not(cv2.dilate(m, K3)))
        ring[blue_d > 0] = 0
        px = lab[ring > 0]
        out.append(np.median(px, axis=0).astype(np.float32) if len(px) >= 25 else None)
    return out


def make_D(lab, ref):
    """Ban do 'khac giay': 0 = giong mau giay, lon = khac. Thay cho nguong trang co dinh."""
    r = np.clip(np.round(ref), 0, 255).astype(int)
    return cv2.transform(cv2.absdiff(lab, (int(r[0]), int(r[1]), int(r[2]), 0)), TW)


# ------------------------------------------------------------------
#  TANG 2: cac gia thuyet vai tro hinh ve + hat giong affine
# ------------------------------------------------------------------
def role_hypotheses(triple):
    """Tu 3 hinh ve -> (TL tron, BL tam giac, BR vuong). Dua vao hinh hoc: chu L, ty le canh, chieu quay."""
    pts = [np.array([m[0], m[1]], np.float32) for m in triple]
    out = []
    for vi in range(3):
        V = pts[vi]
        o = [pts[j] for j in range(3) if j != vi]
        for TL, BR in ((o[0], o[1]), (o[1], o[0])):
            va, vb = TL - V, BR - V
            la, lb = np.linalg.norm(va), np.linalg.norm(vb)
            if la < 8 or lb < 8:
                continue
            if abs(float(np.dot(va, vb)) / (la * lb)) > 0.6:      # goc ~ 53..127 do
                continue
            if not 1.2 <= lb / la <= 2.6:                          # canh dai / canh ngan
                continue
            cross = (V - TL)[0] * (BR - V)[1] - (V - TL)[1] * (BR - V)[0]
            if cross >= 0:                                         # sai chieu quay (guong)
                continue
            out.append((TL, V, BR))
    return out


def affine_seed(TL, BL, BR):
    i, W, H = MARK_INSET_CM, PLATE_W_CM, PLATE_H_CM
    src = np.float32([[i, i], [i, H - i], [W - i, H - i]])
    A = cv2.getAffineTransform(src, np.float32([TL, BL, BR]))
    return (CANON @ A[:, :2].T + A[:, 2]).astype(np.float32)


# ------------------------------------------------------------------
#  TANG 3 (chi tiet): bam mep - do mep theo phap tuyen + fit duong thang ben vung
#  Bi / hinh ve / vat che mep chi la diem ngoai nhom (outlier) nen bi loai khi fit.
# ------------------------------------------------------------------
def fit_line(pts, n_total):
    """RANSAC tren moi cap diem: duong nao co nhieu diem thang hang nhat thi thang.
    Cung tron cua bi nam sat mep chi la mot cum diem cong, khong the thang duoc phan mep that."""
    P = np.ascontiguousarray(pts, dtype=np.float32)
    n = len(P)
    ii, jj = np.triu_indices(n, 1)
    if len(ii) > 90:                                   # chi can ~90 cap diem (deu dan) la du
        sel = np.linspace(0, len(ii) - 1, 90).astype(int)
        ii, jj = ii[sel], jj[sel]
    d = P[jj] - P[ii]
    L = np.linalg.norm(d, axis=1)
    keep = L > 4.0
    inl = None
    if keep.any():
        ii, jj, d, L = ii[keep], jj[keep], d[keep], L[keep]
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1) / L[:, None]
        r = np.abs(((P[None, :, :] - P[ii][:, None, :]) * nrm[:, None, :]).sum(axis=2))
        cnt = (r < 1.2).sum(axis=1)
        best = int(np.argmax(cnt))
        if cnt[best] >= 5:
            inl = r[best] < 1.2
    if inl is None:
        inl = np.ones(n, bool)
    vx, vy, x0, y0 = cv2.fitLine(np.ascontiguousarray(P[inl]), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    nrm = np.array([-vy, vx], np.float32)
    inl = np.abs((P - np.array([x0, y0], np.float32)) @ nrm) < 1.5
    return np.array([x0, y0], np.float32), np.array([vx, vy], np.float32), inl.sum() / float(n_total)


def intersect(l1, l2):
    p1, d1, _ = l1
    p2, d2, _ = l2
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-3:
        return None
    a = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return p1 + a * d1


def refine_quad(D, quad, radii, thr=EDGE_THR, partial=False, excl=None):
    """Day 4 canh cua quad ve dung mep that. Tra ve (quad moi, do ho tro moi canh).
    partial=True: canh nao khong thay mep ro thi giu nguyen duong du doan (ho tro = 0) thay vi that bai."""
    q = quad.astype(np.float32)
    sup = np.zeros(4, np.float32)
    need = max(6, int(0.4 * N_SAMPLES))
    for R in radii:
        c = q.mean(axis=0)
        lines = []
        sup = np.zeros(4, np.float32)
        K = 2 * R + 1
        s = np.arange(-R, R + 1, dtype=np.float32)
        t = np.linspace(T_LO, T_HI, N_SAMPLES, dtype=np.float32)
        js = np.arange(EDGE_W - 1, K - EDGE_W)
        sv = (js - R + 0.5).astype(np.float32)
        for i in range(4):
            p0, p1 = q[i], q[(i + 1) % 4]
            d = p1 - p0
            L = float(np.linalg.norm(d))
            if L < 15:
                return None, None
            d = d / L
            n = np.array([d[1], -d[0]], np.float32)
            if float(np.dot(n, (p0 + p1) / 2 - c)) < 0:
                n = -n
            base = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
            X = (base[:, 0:1] + s[None, :] * n[0]).astype(np.float32)
            Y = (base[:, 1:2] + s[None, :] * n[1]).astype(np.float32)
            S = cv2.remap(D, X, Y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            cs = np.concatenate([np.zeros((N_SAMPLES, 1), np.float32), np.cumsum(S, axis=1)], axis=1)
            resp = (cs[:, js + 1 + EDGE_W] - cs[:, js + 1]) / EDGE_W \
                - (cs[:, js + 1] - cs[:, js + 1 - EDGE_W]) / EDGE_W      # ngoai - trong
            adj = resp - 0.25 * np.abs(sv)[None, :]                       # uu tien mep gan
            idx = np.argmax(adj, axis=1)
            best = resp[np.arange(N_SAMPLES), idx]
            valid = best > thr
            allpts = base + sv[idx][:, None] * n[None, :]
            if excl is not None:                       # bo cac diem do nam sat bi (bi che/lam sai mep)
                valid &= np.hypot(allpts[:, 0] - excl[0], allpts[:, 1] - excl[1]) > excl[2]
            ln = None
            if valid.sum() >= need:
                pts = allpts[valid]
                ln = fit_line(pts, N_SAMPLES)
                if partial and ln[2] < 0.35:
                    ln = None
            if ln is None:
                if not partial:
                    return None, None
                ln = (p0.copy(), d.copy(), 0.0)
            lines.append(ln)
            sup[i] = ln[2]
        newq = []
        for i in range(4):
            pt = intersect(lines[(i - 1) % 4], lines[i])
            if pt is None or np.linalg.norm(pt - q[i]) > 3.5 * R + 6:
                return None, None
            newq.append(pt)
        q = np.array(newq, np.float32)
    return q, sup


def interior_points(q):
    u, v = GRID[:, 0:1], GRID[:, 1:2]
    return (1 - u) * (1 - v) * q[0] + u * (1 - v) * q[1] + u * v * q[2] + (1 - u) * v * q[3]


def interior_frac(D, q):
    P = interior_points(q)
    xi = np.clip(np.rint(P[:, 0]).astype(int), 0, D.shape[1] - 1)
    yi = np.clip(np.rint(P[:, 1]).astype(int), 0, D.shape[0] - 1)
    return float((D[yi, xi] < INTERIOR_TOL).mean())


def quad_ok(D, q, sup, strict):
    """Cac phep kiem tra hinh hoc + noi that. strict=True khi khoi tao."""
    if q is None:
        return False, 0.0, 0.0
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return False, 0.0, 0.0
    h, w = D.shape[:2]
    area = cv2.contourArea(q)
    if not 0.04 * w * h <= area <= 0.92 * w * h:
        return False, 0.0, 0.0
    top, bot = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    lef, rig = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    ratio = ((top + bot) / 2.0) / max(1e-3, (lef + rig) / 2.0)
    if not 1.1 <= ratio <= 2.7:
        return False, 0.0, 0.0
    fr = interior_frac(D, q)
    if strict:
        ok = sup.min() >= 0.35 and sup.mean() >= 0.55 and fr >= 0.60
    else:
        ok = sup.min() >= 0.25 and sup.mean() >= 0.45 and fr >= 0.45
    return ok, float(sup.mean()), fr


# ------------------------------------------------------------------
#  NEO BANG HINH VE: 3 hinh ve luon nam o cho co dinh tren tam -> hat giong cuc chac
# ------------------------------------------------------------------
def to_canon(q, pts):
    """Toa do anh -> toa do tam (cm) theo khung q."""
    Hm = cv2.getPerspectiveTransform(q.astype(np.float32), CANON)
    return cv2.perspectiveTransform(np.float32(pts).reshape(1, -1, 2), Hm)[0]


def seed_from_marks(pts3, mark_cm):
    A = cv2.getAffineTransform(np.float32(mark_cm), np.float32(pts3))
    return (CANON @ A[:, :2].T + A[:, 2]).astype(np.float32)


def nominal_mark_cm():
    i, W, H = MARK_INSET_CM, PLATE_W_CM, PLATE_H_CM
    return np.float32([[i, i], [i, H - i], [W - i, H - i]])


def inset_ok(mc):
    W, H, lo, hi = PLATE_W_CM, PLATE_H_CM, 0.2, 2.2
    return bool(lo <= mc[0, 0] <= hi and lo <= mc[0, 1] <= hi
                and lo <= mc[1, 0] <= hi and H - hi <= mc[1, 1] <= H - lo
                and W - hi <= mc[2, 0] <= W - lo and H - hi <= mc[2, 1] <= H - lo)


def locate_marks(hsv, q, mark_cm):
    """Tim 3 hinh ve quanh vi tri du doan (TL, BL, BR). Tra ve list 3 phan tu (x,y) hoac None."""
    Hm = cv2.getPerspectiveTransform(CANON, q.astype(np.float32))
    pos = cv2.perspectiveTransform(np.float32(mark_cm).reshape(1, -1, 2), Hm)[0]
    cm_px = np.linalg.norm(q[1] - q[0]) / PLATE_W_CM
    half = max(5, int(MARK_SEARCH_CM * cm_px))
    sh, sw = hsv.shape[:2]
    out = []
    for x, y in pos:
        x0, x1 = int(max(x - half, 0)), int(min(x + half, sw))
        y0, y1 = int(max(y - half, 0)), int(min(y + half, sh))
        if x1 - x0 < 4 or y1 - y0 < 4:
            out.append(None)
            continue
        patch = cv2.inRange(hsv[y0:y1, x0:x1], (100, BLUE_S_MIN, 20), (150, 255, 255))
        patch = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, K3)
        cs, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, bd = None, 1e9
        for c in cs:                                      # chon vet xanh GAN vi tri du doan nhat (khong phai to nhat:
            if cv2.contourArea(c) < MARK_MIN_PIX:         # day dien xanh/vat xanh khac co the to hon hinh ve nho)
                continue
            m = cv2.moments(c)
            if m["m00"] < 1:
                continue
            cx, cy = x0 + m["m10"] / m["m00"], y0 + m["m01"] / m["m00"]
            d = float(np.hypot(cx - x, cy - y))
            if d < bd:
                best, bd = (cx, cy), d
        out.append(best if bd <= 0.9 * half else None)
    return out


def consistent(q, found, mark_cm):
    """True/False: khung q co khop vi tri hinh ve da hoc khong. None: khong du hinh ve de ket luan."""
    idx = [i for i, f in enumerate(found) if f is not None]
    if len(idx) < 2:
        return None
    c = to_canon(q, [found[i] for i in idx])
    dev = np.linalg.norm(c - np.float32(mark_cm)[idx], axis=1)
    return bool((dev < MARK_TOL_CM).all())


def geom_ok(D, q, min_inter):
    if q is None or not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return False, 0.0
    h, w = D.shape[:2]
    if not 0.03 * w * h <= cv2.contourArea(q) <= 0.92 * w * h:
        return False, 0.0
    top, bot = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    lef, rig = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    ratio = ((top + bot) / 2.0) / max(1e-3, (lef + rig) / 2.0)
    if not 1.1 <= ratio <= 2.7:
        return False, 0.0
    fr = interior_frac(D, q)
    return fr >= min_inter, fr


def detect_init(small_blur, lab):
    """Khoi tao: tim hinh ve -> thu moi gia thuyet vai tro -> day mep (mep yeu van chap nhan nho hat giong)."""
    hsv = cv2.cvtColor(small_blur, cv2.COLOR_BGR2HSV)
    marks, blue = [], None
    for smin in (BLUE_S_MIN + 30, BLUE_S_MIN):
        b = cv2.inRange(hsv, (100, smin, 20), (150, 255, 255))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, K3)
        blue = b if blue is None else cv2.bitwise_or(blue, b)
        for mk in find_marks(b):
            if all((mk[0] - o[0]) ** 2 + (mk[1] - o[1]) ** 2 > 36 for o in marks):
                marks.append(mk)
    marks = sorted(marks, key=lambda t: t[2], reverse=True)[:8]
    if len(marks) < 3:
        return None, marks
    meds = ring_medians(lab, blue, marks)

    best, best_score = None, 0.0
    for ids in itertools.combinations(range(len(marks)), 3):
        if any(meds[i] is None for i in ids):
            continue
        hyps = role_hypotheses([marks[i] for i in ids])
        if not hyps:
            continue
        ref = np.median(np.array([meds[i] for i in ids]), axis=0).astype(np.float32)
        D = make_D(lab, ref)
        for TL, BL, BR in hyps:
            q, sup = refine_quad(D, affine_seed(TL, BL, BR), INIT_R, EDGE_THR_SEEDED, partial=True)
            if q is None:
                continue
            strong = int((sup >= 0.5).sum())
            ok, fr = geom_ok(D, q, 0.60 if strong >= 2 else 0.75)
            if not ok:
                continue
            mc = to_canon(q, [TL, BL, BR]).astype(np.float32)
            if not inset_ok(mc):
                continue
            score = (float(sup.sum()) + 0.5) * fr
            if score > best_score:
                best_score = score
                best = (q, ref, D, mc if strong >= 3 else nominal_mark_cm())
    return best, marks


def similarity_apply(p_old, p_new, quad):
    """Tu 2 cap diem (hinh ve truoc/sau) suy ra phep dong dang (xoay+co+dich) roi ap len khung truoc."""
    z = np.array([complex(p[0], p[1]) for p in p_old])
    w = np.array([complex(p[0], p[1]) for p in p_new])
    den = z[1] - z[0]
    a = (w[1] - w[0]) / den if abs(den) > 1e-3 else 1 + 0j
    if not 0.8 <= abs(a) <= 1.25:                   # ty le vo ly -> chi tinh tien
        a = 1 + 0j
    b = w.mean() - a * z.mean()
    r = a * np.array([complex(p[0], p[1]) for p in quad]) + b
    return np.stack([r.real, r.imag], axis=1).astype(np.float32)


class PlateTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.quad = None          # TL, TR, BR, BL (toa do anh nho)
        self.anchor_marks = self.anchor_quad = None
        self.ref = None
        self.mark_cm = nominal_mark_cm()
        self.vel = np.zeros((4, 2), np.float32)
        self.lost = 0
        self.n = 0
        self.D = None
        self.hsv = None
        self.marks = []
        self.state = "SEARCH"
        self.sup_min = 0.0
        self.inter = 0.0
        self.nfound = 0
        self.occl = 0
        self.anchor_marks = None  # vi tri 3 hinh ve + khung tai thoi diem mep da xac nhan chac (khung moc)
        self.anchor_quad = None
        self.ball_hint = None     # (x, y, r) bi o frame truoc: bo cac diem do mep nam sat bi

    def update(self, frame):
        small = cv2.resize(frame, None, fx=DET_SCALE, fy=DET_SCALE, interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (3, 3), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        self.n += 1
        self.hsv = hsv

        if self.quad is not None:
            D = make_D(lab, self.ref)
            pred = self.quad + VEL_GAIN * self.vel
            found = locate_marks(hsv, pred, self.mark_cm)
            idx = [i for i, f in enumerate(found) if f is not None]
            nf = len(idx)
            self.nfound = nf
            excl = self.ball_hint
            Hm_prev = cv2.getPerspectiveTransform(CANON, self.quad.astype(np.float32))
            old_pos = cv2.perspectiveTransform(self.mark_cm.reshape(1, -1, 2), Hm_prev)[0]
            q, sup, src = None, np.zeros(4, np.float32), ""

            if nf == 3:                                  # du 3 hinh ve: neo chac nhat
                seed = seed_from_marks(found, self.mark_cm)
                # goc TR khong co hinh ve -> suy ra tu khung truoc dich theo 3 hinh ve (giu phoi canh)
                if self.anchor_marks is not None:           # tinh tu khung moc -> khong tich luy sai so
                    A = cv2.getAffineTransform(self.anchor_marks, np.float32(found))
                    seed[1] = cv2.transform(self.anchor_quad.reshape(1, -1, 2), A)[0, 1]
                else:
                    A = cv2.getAffineTransform(np.float32(old_pos), np.float32(found))
                    seed[1] = cv2.transform(self.quad.reshape(1, -1, 2).astype(np.float32), A)[0, 1]
                label = "TRACK"
            elif nf == 2:                                # 1 hinh ve bi bi che -> dong dang tu 2 hinh ve con lai
                if self.anchor_marks is not None:
                    seed = similarity_apply(self.anchor_marks[idx], [found[i] for i in idx], self.anchor_quad)
                else:
                    seed = similarity_apply(old_pos[idx], [found[i] for i in idx], self.quad)
                label = "TRK2"
            else:                                        # >=2 hinh ve bi che -> giu khung du doan, mep chi de chinh them
                seed = pred.astype(np.float32)
                label = "PRED"

            if nf >= 2:
                q, sup = refine_quad(D, seed, MARK_R, EDGE_THR_SEEDED, True, excl)
                cons = consistent(q, found, self.mark_cm) if q is not None else False
                if q is not None and cons is not False and geom_ok(D, q, 0.5)[0]:
                    src = label
                else:                                    # mep khong dang tin -> tin vao hinh ve
                    q, sup, src = seed, np.zeros(4, np.float32), "MARKS"
                    if not geom_ok(D, q, 0.5)[0]:
                        q = None
            else:
                q, sup = refine_quad(D, seed, TRACK_R, EDGE_THR_SEEDED, True, excl)
                if q is None or not geom_ok(D, q, 0.45)[0]:
                    q = seed if geom_ok(D, seed, 0.45)[0] else None
                    sup = np.zeros(4, np.float32)
                src = label
            self.occl = self.occl + 1 if nf <= 1 else 0

            # che lau (>=10 frame thay <2 hinh ve): thu khoi tao lai song song, co the tam da bi dich chuyen that
            if q is not None and self.occl >= 10:
                res0, marks0 = detect_init(blur, lab)
                if res0 is not None:
                    q, self.ref, D, self.mark_cm = res0
                    sup, src, self.occl = np.zeros(4, np.float32), "LOCK", 0
                    self.quad = q.copy()
                    self.anchor_marks = self.anchor_quad = None

            if q is not None:
                move = float(np.abs(q - self.quad).max())
                alpha = ALPHA_STILL if move < 1.5 else ALPHA_MOVE
                new_quad = alpha * q + (1.0 - alpha) * self.quad
                self.vel = np.clip(0.6 * self.vel + 0.4 * (new_quad - self.quad), -15, 15)
                self.quad = new_quad
                self.lost = 0
                self.D, self.state = D, src
                self.sup_min = float(sup.min())
                self.inter = interior_frac(D, self.quad)
                if src == "TRACK" and nf == 3 and sup[0] >= 0.6 and sup[1] >= 0.6:   # 2 canh quanh goc TR deu co mep chac
                    self.anchor_marks = np.float32(found)
                    self.anchor_quad = q.astype(np.float32).copy()
                if src == "TRACK" and nf == 3 and sup.min() >= 0.6:     # hoc lai vi tri hinh ve
                    mc = to_canon(q, found).astype(np.float32)
                    if inset_ok(mc):
                        self.mark_cm = 0.95 * self.mark_cm + 0.05 * mc
                if self.n % 8 == 0:                      # cap nhat mau giay theo anh sang
                    P = interior_points(self.quad)
                    xi = np.clip(np.rint(P[:, 0]).astype(int), 0, lab.shape[1] - 1)
                    yi = np.clip(np.rint(P[:, 1]).astype(int), 0, lab.shape[0] - 1)
                    self.ref = 0.7 * self.ref + 0.3 * np.median(lab[yi, xi], axis=0).astype(np.float32)
                return self.quad
            self.lost += 1
            self.vel[:] = 0

        res, marks = detect_init(blur, lab)
        self.marks = marks
        if res is not None:
            q, ref, D, mc = res
            self.quad, self.ref, self.mark_cm = q, ref, mc
            self.lost = 0
            self.vel[:] = 0
            self.D, self.state = D, "LOCK"
            f0 = locate_marks(hsv, q, mc)
            if all(f is not None for f in f0):
                self.anchor_marks, self.anchor_quad = np.float32(f0), q.astype(np.float32).copy()
            else:
                self.anchor_marks = self.anchor_quad = None
            return self.quad
        if self.quad is not None:
            self.state = "HOLD"
        if self.lost > LOST_MAX:
            self.quad = None
            self.state = "SEARCH"
        return self.quad

    def corners_full(self):
        if self.quad is None:
            return None
        return ((self.quad + 0.5) / DET_SCALE - 0.5).astype(np.float32)


def refine_ball(D, cx, cy, r):
    """Tinh chinh tam bi bang cac diem nep SAC (D tut manh trong ~2px). Bong do tut tu tu nen khong tinh.
    Nho do tam khong bi lech ve phia bong va khong bi mat phan vanh sang."""
    n = 24
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).astype(np.float32)
    rs = np.arange(max(2.0, 0.7 * r), 1.7 * r + 4, 0.5, dtype=np.float32)
    X = (cx + np.outer(np.cos(ang), rs)).astype(np.float32)
    Y = (cy + np.outer(np.sin(ang), rs)).astype(np.float32)
    S = cv2.remap(D, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    g = S[:, :-4] - S[:, 4:]                       # do tut theo huong ra ngoai (~2 px)
    pts = []
    for k in range(n):
        idx = np.where(g[k] > 28)[0]
        if len(idx):
            rr = rs[idx[-1] + 2]                   # nep ngoai cung
            pts.append((cx + rr * np.cos(ang[k]), cy + rr * np.sin(ang[k])))
    if len(pts) < 9:
        return cx, cy, r
    (ex, ey), (ma, mi), _ = cv2.fitEllipse(np.array(pts, np.float32).reshape(-1, 1, 2))
    rad = 0.25 * (ma + mi)
    if np.hypot(ex - cx, ey - cy) > 0.6 * r or not 0.6 * r <= rad <= 1.6 * r:
        return cx, cy, r
    return float(ex), float(ey), float(rad)


def fit_circle_fixed_r(pts, r, c0):
    """Tim tam duong tron ban kinh r da biet sao cho cac diem vanh nam tren no (Gauss-Newton)."""
    c = np.array(c0, np.float64)
    pts = pts.astype(np.float64)
    for _ in range(10):
        v = c - pts
        d = np.linalg.norm(v, axis=1) + 1e-6
        J = v / d[:, None]
        dc = -np.linalg.lstsq(J, d - r, rcond=None)[0]
        c += dc
        if np.linalg.norm(dc) < 0.01:
            break
    return c


def find_ball_cam(D, hsv, q, prev=None, r_cm=None):
    """Tim bi tren anh camera (nho). Tra ve (cx, cy, r, bi_cat) hoac None.
    - Bi 'khac mau giay' rat manh; bong do mem khong vuot nguong.
    - Bi bi mep tam cat (nam sat/vuot mep): chap nhan manh nho, tim tam bang duong tron co ban kinh da biet
      khop voi cac diem vanh NAM TRONG tam (khong tinh phan bi bi cat ngang mep hay dinh muc xanh)."""
    h, w = D.shape[:2]
    poly = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(poly, np.rint(q).astype(np.int32), 255)
    band = cv2.bitwise_and(poly, cv2.bitwise_not(cv2.erode(poly, K5)))     # dai 2 px sat mep tam
    _, cand = cv2.threshold(D, BALL_D_THR, 255, cv2.THRESH_BINARY)
    cand = cv2.bitwise_and(cand, poly)
    blue = cv2.inRange(hsv, (100, BLUE_S_MIN, 20), (150, 255, 255))
    blue_d = cv2.dilate(blue, K3)
    cand[blue_d > 0] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, K3)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, K9)
    cand = cv2.bitwise_and(cand, poly)
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cm_px = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0 / PLATE_W_CM
    r_nom = (r_cm if r_cm else BALL_DIAMETER_MM / 20.0) * cm_px
    nom = np.pi * r_nom ** 2
    blue_near = cv2.dilate(blue, K7)
    best, best_cost = None, 1e9
    for c in contours:
        if cv2.contourArea(c) < 8:
            continue
        pts = c[:, 0, :]
        on_edge = band[pts[:, 1], pts[:, 0]] > 0
        clipped = bool(on_edge.sum() >= 3)
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        ratio = area / nom
        if clipped:
            if not 0.10 <= ratio <= 3.0:
                continue
            rim = pts[(~on_edge) & (blue_near[pts[:, 1], pts[:, 0]] == 0)]
            if len(rim) < 6:
                continue
            m = cv2.moments(hull)
            if m["m00"] < 1:
                continue
            c0 = (m["m10"] / m["m00"], m["m01"] / m["m00"])
            ctr = fit_circle_fixed_r(rim, r_nom, c0)
            if np.hypot(ctr[0] - c0[0], ctr[1] - c0[1]) > 1.6 * r_nom:
                continue
            cand_ball = (float(ctr[0]), float(ctr[1]), float(r_nom), True)
            cost = 0.8
        else:
            if not 0.18 <= ratio <= 3.0:
                continue
            if ratio >= 0.35:                                    # manh to du moi kiem tra do tron
                (_, _), r_enc = cv2.minEnclosingCircle(hull)
                if area / (np.pi * r_enc * r_enc + 1e-6) < 0.55:
                    continue
            m = cv2.moments(hull)
            if m["m00"] < 1:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            cand_ball = (cx, cy, float(np.sqrt(area / np.pi)), False)
            cost = abs(np.log(min(ratio, 1.0))) * 0.5 + (0.0 if ratio >= 0.35 else 0.3)
        if prev is not None:
            cost += 0.02 * np.hypot(cand_ball[0] - prev[0], cand_ball[1] - prev[1])
        if cost < best_cost:
            best_cost, best = cost, cand_ball
    if best is None:
        return None
    if best[3]:
        return best
    cx, cy, r = refine_ball(D, best[0], best[1], r_nom)       # quet vanh theo ban kinh bi that, khong theo manh (co the thieu)
    return cx, cy, r, False


def rectify_plate(frame, corners):
    dst = np.array([[0, 0], [WARP_W - 1, 0],
                    [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(frame, m, (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)


def draw_plate(display, corners):
    pts = corners.astype(np.int32)
    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
    for name, p in zip(("TL", "TR", "BR", "BL"), pts):
        cv2.circle(display, tuple(int(v) for v in p), 4, (0, 255, 255), -1)
        cv2.putText(display, name, (int(p[0]) + 6, int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def main():
    global BLUE_S_MIN
    stream = CameraStream(CAM_INDEX)
    if not stream.running:
        print("Khong mo duoc camera. Hay doi CAM_INDEX = 1 hoac 2...")
        return

    tracker = PlateTracker()
    last_seq, smooth, ball_prev, ball_miss = 0, None, None, 0
    ball_r_cm = BALL_DIAMETER_MM / 20.0
    fps, trk_ms, ball_ms, last_print = 0.0, 0.0, 0.0, 0.0
    last_tick = time.time()
    show_debug = False

    print("Q: thoat | R: tim lai tam | D: xem ban do 'khac mau giay' | , . : chinh do nhay muc xanh")
    print("Quy uoc: TL=hinh tron, BL=tam giac, BR=vuong, TR=trong (nhin tu tren); X doc canh dai, Y doc canh ngan")

    while True:
        frame, last_seq = stream.read(last_seq)
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.001)
            continue

        now = time.time()
        dt = now - last_tick
        last_tick = now
        if dt > 0:
            fps = 1.0 / dt if fps == 0 else 0.85 * fps + 0.15 / dt

        t0 = time.perf_counter()
        tracker.update(frame)
        t1 = time.perf_counter()
        q = tracker.quad
        ball = None
        if q is not None and tracker.state in ("TRACK", "TRK2", "PRED", "MARKS", "LOCK"):
            ball = find_ball_cam(tracker.D, tracker.hsv, q, ball_prev, ball_r_cm)
        t2 = time.perf_counter()
        trk_ms = 0.9 * trk_ms + 0.1 * (t1 - t0) * 1000.0
        ball_ms = 0.9 * ball_ms + 0.1 * (t2 - t1) * 1000.0

        display = frame.copy()
        plate_view = None
        corners = tracker.corners_full()

        if corners is not None:
            draw_plate(display, corners)
            plate_view = rectify_plate(frame, corners)
            if ball is not None:
                ball_miss = 0
                cx, cy, r_s, clipped = ball
                ball_prev = (cx, cy)
                cm_px_b = np.linalg.norm(q[1] - q[0]) / PLATE_W_CM
                if not clipped:                          # hoc ban kinh bi that (cm) khi nhin thay tron ven
                    ball_r_cm = 0.9 * ball_r_cm + 0.1 * (r_s / cm_px_b)
                tracker.ball_hint = (cx, cy, 1.7 * r_s)
                raw = to_canon(q, [[cx, cy]])[0]         # px anh -> cm tren mat tam (qua homography)
                smooth = raw if smooth is None else EMA_ALPHA * raw + (1 - EMA_ALPHA) * smooth
                x_cm, y_cm = float(smooth[0]), float(smooth[1])
                cm_px = np.linalg.norm(q[1] - q[0]) / PLATE_W_CM
                r_px = int(r_s / cm_px * PIXELS_PER_CM)
                p = (int(x_cm * PIXELS_PER_CM), int(y_cm * PIXELS_PER_CM))
                cv2.circle(plate_view, p, max(r_px, 3), (0, 255, 0), 2)
                cv2.circle(plate_view, p, 4, (0, 0, 255), -1)
                cv2.putText(plate_view, f"X={x_cm:.2f} cm  Y={y_cm:.2f} cm", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                fx, fy = (cx + 0.5) / DET_SCALE - 0.5, (cy + 0.5) / DET_SCALE - 0.5
                cv2.circle(display, (int(fx), int(fy)), int(r_s / DET_SCALE), (0, 255, 0), 2)
                cv2.putText(display, f"Ball: X={x_cm:.2f}, Y={y_cm:.2f} cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                if now - last_print >= 0.2:
                    print(f"X={x_cm:5.2f} cm   Y={y_cm:5.2f} cm")
                    last_print = now
            else:
                ball_miss += 1
                if ball_miss > BALL_MISS_MAX:
                    smooth, ball_prev = None, None
                if ball_miss > 6:
                    tracker.ball_hint = None
                cv2.putText(plate_view, "BALL NOT FOUND", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, f"KHONG TIM THAY TAM (thay {len(tracker.marks)} hinh ve, can 3)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.putText(display,
                    f"{tracker.state} FPS:{fps:.0f} trk:{trk_ms:.1f} ball:{ball_ms:.1f}ms marks:{tracker.nfound}",
                    (10, display.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.imshow("Camera - plate detection", display)
        if plate_view is not None:
            cv2.imshow("Plate top view - ball coordinate", plate_view)
        if show_debug and tracker.D is not None:
            dv = cv2.applyColorMap(cv2.convertScaleAbs(tracker.D, alpha=4), cv2.COLORMAP_JET)
            if tracker.quad is not None:
                cv2.polylines(dv, [tracker.quad.astype(np.int32)], True, (255, 255, 255), 1)
            cv2.imshow("Debug: khac mau giay (xanh = giong giay)", dv)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
            smooth = None
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("Debug: khac mau giay (xanh = giong giay)")
        elif key == ord(','):
            BLUE_S_MIN = max(20, BLUE_S_MIN - 5)
        elif key == ord('.'):
            BLUE_S_MIN = min(160, BLUE_S_MIN + 5)

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
