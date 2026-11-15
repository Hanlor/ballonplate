"""
smc_study.py - Kiem tra mo hinh theo TECHNICAL_SPECIFICATION + so sanh PID / SMC bang mo phong (khong can MATLAB)

  python smc_study.py --spec      kiem tra dieu khien duoc, on dinh, LQR, gioi han tin hieu, dong hoc nguoc (IK)
  python smc_study.py --compare   Monte Carlo 60 he ngau nhien: PID vs SMC vs +bu do ro  (~1-2 phut)

Mo phong CO tre toan vong (camera+servo), do ro servo, ma sat tinh, lech phang, nhieu camera, loc alpha-beta
-> gan voi he that hon mo hinh ly tuong cua spec.  Day la MO PHONG; ket qua tren he that co the khac.
Quy uoc: u (do) duong => bi tang toc theo +X/+Y (== ux = -theta_g trong spec).
"""
import argparse, math, sys
import numpy as np
from collections import deque
KI_BOOST = 4.0
# ---------------- controllers (per axis, plate degrees; ref r, rd) ----------------
class PID:
    def __init__(s,KP,KD,KI,LAT,lim=6.5,ilim=5.0): s.p=(KP,KD,KI,LAT); s.lim=lim; s.ilim=ilim; s.reset()
    def reset(s): s.I=0.0; s.stuck=0.0
    def step(s,x,v,r,rd,d):
        KP,KD,KI,LAT=s.p; xs=x+v*LAT; e=r-xs
        u0=KP*e+KD*(rd-v)
        s.stuck = s.stuck+d if (abs(e)>0.8 and abs(v-rd)<1.5) else 0.0
        ki=KI*(KI_BOOST if s.stuck>0.4 else 1.0)
        if abs(v-rd)<8.0 and not ((u0+s.I>s.lim and e>0) or (u0+s.I<-s.lim and e<0)):
            s.I=max(-s.ilim,min(s.ilim,s.I+ki*e*d))
        return max(-s.lim,min(s.lim,u0+s.I))
# ---------------- plant + estimator ----------------
def sim(ctrl,plant,ref,T=12.0,x0=0.0,dt=0.002,fps=30.0,seed=0,edge=7.0):
    k,L,c,b=plant['k'],plant['L'],plant['c'],plant['b']; tau=plant.get('tau',0.04); bl=plant.get('bl',3.0)
    stic=plant.get('stic',4.0); noise=plant.get('noise',0.06); gear=plant.get('gear',5.0); A=plant.get('A',0.7); B=plant.get('B',0.33)
    Lc=max(L-tau,0.02); rng=np.random.default_rng(seed); n=int(T/dt)
    x,v=x0,0.0; buf=deque(); ex_,ev_,lt=None,0.0,0.0
    sc=slag=pos=0.0; u=0.0; nextf=0.0; slew=300.0*dt; half=bl/2
    blest=plant.get('blest',0.0); dirc=0.0; lastraw=0.0; eps=plant.get('bleps',0.4)
    ctrl.reset(); noises=rng.normal(0,noise,int(T*fps)+10); ni=0
    T_=[];X_=[];U_=[];R_=[]; fail=None
    for i in range(n):
        t=i*dt
        if t>=nextf: nextf+=1/fps; buf.append((t+Lc,x+noises[ni])); ni+=1
        while buf and buf[0][0]<=t:
            _,z=buf.popleft()
            if ex_ is None: ex_,ev_,lt=z,0.0,t; continue
            d=t-lt
            if d<0.004: continue
            lt=t; xp=ex_+ev_*d; r=z-xp; ex_=xp+A*r; ev_=ev_+(B/d)*r
            rr,rd=ref(t)
            u=ctrl.step(ex_,ev_,rr,rd,d); raw=u*gear
            if raw-lastraw>eps: dirc=1.0; lastraw=raw
            elif lastraw-raw>eps: dirc=-1.0; lastraw=raw
            sc=raw+dirc*blest/2.0
        diff=sc-slag
        if abs(diff)>0.5: slag+=max(-slew,min(slew,diff/tau*dt))
        if slag-pos>half: pos=slag-half
        elif slag-pos<-half: pos=slag+half
        a=k*(pos/gear+b+(plant.get('db',0.0) if t>=plant.get('tdb',1e9) else 0.0))-c*v
        if abs(v)<0.3 and abs(a)<stic: a=0.0; v=0.0
        if 'kick' in plant and abs(t-plant['kick'][0])<dt/2: v+=plant['kick'][1]
        v+=a*dt; x+=v*dt
        if abs(x)>edge: fail=t; break
        if i%5==0: T_.append(t);X_.append(x);U_.append(u);R_.append(ref(t)[0])
    return np.array(T_),np.array(X_),np.array(U_),np.array(R_),fail
def reg_cost(res,x0,T):
    t,x,u,r,fail=res
    if fail is not None: return 40+20*(1-fail/T),None
    e=x-r; iae=float(np.mean(np.abs(e))); tail=float(np.sqrt(np.mean(e[t>T-4]**2)))
    side=1.0 if x0>0 else -1.0; os_=float(max(0.0,np.max(-side*e)))
    eff=float(np.mean(np.abs(np.diff(u)))/0.01)
    return iae+2*tail+0.7*os_+0.01*eff, dict(iae=iae,tail=tail,os=os_,eff=eff)
REG=lambda t:(0.0,0.0)
def track_cost(res,R,w,T):
    t,x,u,r,fail=res
    if fail is not None: return None
    m=t>3.0; e=(x-r)[m]; return float(np.sqrt(np.mean(e**2))), float(np.max(np.abs(e)))

class SMC2:
    """SMC lop bien + tich phan BU LECH ngoai mat truot (nhu firmware moi):
       s = de + lam*e + li*q ;  u = -(lam*de + li*e + KSW*sat(s/phi))/K + I ;  I += -ki*e*dt (gating, chong windup)"""
    def __init__(s,lam,KSW,phi,ki,LAT,li=0.0,K=9.5,lim=6.5,ilim=5.0): s.p=(lam,KSW,phi,ki,LAT,li); s.K=K; s.lim=lim; s.ilim=ilim; s.reset()
    def reset(s): s.q=0.0; s.I=0.0; s.stuck=0.0
    def step(s,x,v,r,rd,d):
        lam,KSW,phi,ki,LAT,li=s.p; xs=x+v*LAT; e=xs-r; de=v-rd
        sv=de+lam*e+li*s.q; sat=max(-1.0,min(1.0,sv/phi))
        un=-(lam*de+li*e+KSW*sat)/s.K
        s.stuck = s.stuck+d if (abs(e)>0.8 and abs(de)<1.5) else 0.0
        boost=KI_BOOST if s.stuck>0.4 else 1.0
        ut=un+s.I
        satu=(ut>s.lim and e<0) or (ut<-s.lim and e>0)
        if abs(de)<8.0 and not satu:
            s.I=max(-s.ilim,min(s.ilim,s.I-boost*ki*e*d))
            if li>0: qm=s.K*s.ilim/li; s.q=max(-qm,min(qm,s.q+boost*e*d))
        return max(-s.lim,min(s.lim,un+s.I))


# ============================== KIEM TRA SPEC ==============================
def spec_check():
    Kg = 5 / 7 * 981.0
    A = np.array([[0, 1], [0, 0]]); B = np.array([[0], [Kg]])
    print("Kg = %.4f cm/s2/rad = %.3f cm/s2 tren 1 do" % (Kg, Kg * math.pi / 180))
    print("Tri rieng A:", np.linalg.eigvals(A), "-> tich phan kep: tiem can bien on dinh (khong on dinh tiem can)")
    print("hang [B AB] =", np.linalg.matrix_rank(np.hstack([B, A @ B])), "(=2 -> dieu khien duoc)")
    umax = 6 * math.pi / 180
    print("|u|max = %.5f rad -> gia toc bi toi da %.1f cm/s2" % (umax, Kg * umax))

    def lqr(Q, R):
        H = np.block([[A, -B @ B.T / R], [-Q, -A.T]]); w, v = np.linalg.eig(H); idx = np.where(w.real < 0)[0]
        P = np.real(v[2:, idx] @ np.linalg.inv(v[:2, idx])); return (B.T @ P) / R
    print("\nLQR (mo hinh KHONG tre, trong so Bryson), K quy doi do tam:")
    plant = dict(k=9.5, L=0.15, c=0.2, b=0.8, bl=3.0, stic=4.0)
    for xmax, vmax in ((3, 15), (5, 30), (8, 40)):
        K = lqr(np.diag([1 / xmax ** 2, 1 / vmax ** 2]), 1 / umax ** 2)
        kp, kd = K[0, 0] * 180 / math.pi, K[0, 1] * 180 / math.pi
        res = sim(PID(kp, kd, 0.0, 0.0), plant, REG, T=12, x0=3.5)
        _, m = reg_cost(res, 3.5, 12)
        tail = "roi khoi tam" if m is None else "dao dong du %.2f cm khi co tre 150 ms + do ro" % m["tail"]
        print("  xmax=%g vmax=%g: Kp=%.2f deg/cm Kd=%.2f deg/(cm/s) -> %s" % (xmax, vmax, kp, kd, tail))

    def ik(theta_deg, L3, sx, L1=1.5, L2=6.52, L4=10.0, HF=1.0, Sz=2.5):
        th = math.radians(theta_deg); a0 = L3 * math.cos(th) - sx; b0 = (L4 - HF) + L3 * math.sin(th) - Sz
        rhs = (L2 ** 2 - a0 ** 2 - b0 ** 2 - L1 ** 2) / (2 * L1); R = math.hypot(a0, b0); phi = math.atan2(b0, a0)
        if abs(rhs / R) > 1: return None
        return math.degrees(-phi + math.acos(rhs / R))
    print("\nDong hoc nguoc theo spec (nhanh tay don huong len):")
    for name, L3, sx in (("Servo X", 6.0, 4.0), ("Servo Y", 4.0, 3.0)):
        al = [ik(t, L3, sx) for t in (-6, -3, 0, 3, 6)]
        slope = (ik(0.5, L3, sx) - ik(-0.5, L3, sx))
        lin = max(abs(ik(t, L3, sx) - al[2] - slope * t) for t in (-6, -3, 3, 6))
        print("  %s: alpha(-6,-3,0,3,6 do tam) = %s do | do doc %.2f do servo/do tam | lech so voi tuyen tinh <= %.2f do servo"
              % (name, [round(a, 1) for a in al], slope, lin))
    print("  (do doc X/Y = %.2f; he thuc do khao sat = 5.03/3.40 = %.2f)" %
          ((ik(.5, 6.0, 4.0) - ik(-.5, 6.0, 4.0)) / (ik(.5, 4.0, 3.0) - ik(-.5, 4.0, 3.0)), 5.03 / 3.40))


# ============================== SO SANH ==============================
PID_BEST = (0.495, 0.445, 0.114, 0.007)             # KP KD KI LAT   (toi uu tren bo kich ban co lech phang lon)
SMC_BEST = (1.478, 73.18, 25.26, 0.1916, 0.004)     # LAM KSW PHI KIS LATS (K=9.5)
SIN = lambda R, T: (lambda t: (R * math.sin(2 * math.pi / T * t), R * 2 * math.pi / T * math.cos(2 * math.pi / T * t)))


def compare(n=60):
    rng = np.random.default_rng(555); plants = []
    for i in range(n):
        plants.append((dict(k=rng.uniform(7, 13), L=rng.uniform(0.10, 0.22), c=rng.uniform(0.05, 0.5), b=rng.uniform(-2, 2),
                            bl=rng.uniform(2, 5), stic=rng.uniform(2, 7)), rng.choice([-1, 1]) * rng.uniform(2.5, 4.0)))
    cfgs = [("PID, khong bu do ro", lambda: PID(*PID_BEST), 0.0), ("PID + bu do ro BL=3", lambda: PID(*PID_BEST), 3.0),
            ("SMC, khong bu do ro", lambda: SMC2(*SMC_BEST), 0.0), ("SMC + bu do ro BL=3", lambda: SMC2(*SMC_BEST), 3.0)]
    print("%d he ngau nhien: k 7-13, tre 100-220 ms, do ro 2-5 do servo, ma sat tinh 2-7, lech phang <=2 do" % n)
    print("%-24s %5s %7s %7s %8s %10s %10s" % ("", "roi", "J_med", "J_p90", "tail cm", "bam T=10s", "bam T=6s"))
    for name, make, bl in cfgs:
        Js, tails, t10, t6, fails = [], [], [], [], 0
        for i, (pl, x0) in enumerate(plants):
            p = dict(pl, blest=bl)
            res = sim(make(), p, REG, T=14, x0=x0, seed=100 + i); J, m = reg_cost(res, x0, 14); Js.append(J)
            if m is None: fails += 1
            else: tails.append(m["tail"])
            for T_, l, sd in ((10, t10, 200), (6, t6, 300)):
                r = sim(make(), p, SIN(2.5, T_), T=14, x0=0.0, seed=sd + i); t = track_cost(r, 2.5, 0, 14); l.append(40 if t is None else t[0])
        print("%-24s %4d%% %7.2f %7.2f %8.2f %10.2f %10.2f" % (name, 100 * fails // n, np.median(Js), np.percentile(Js, 90),
              np.median(tails), np.median(t10), np.median(t6)))
    print("\n(bam = sai so rms khi bam quy dao sin bien do 2.5 cm; J = diem hoi tu co ban, thap tot)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", action="store_true"); ap.add_argument("--compare", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    if not (a.spec or a.compare): ap.print_help()
    if a.spec: spec_check()
    if a.compare: compare(a.n)
