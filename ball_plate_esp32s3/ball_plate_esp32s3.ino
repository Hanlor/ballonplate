/*
  BALL ON PLATE - firmware ESP32-S3 (2 servo MG995)
  --------------------------------------------------------------------------
  Laptop (Python/OpenCV) gui toa do bi qua Serial, ESP32 chay bo dieu khien
  PD + I tren moi truc roi xuat PWM cho 2 servo.  KHONG can MATLAB.

  GIAO THUC (moi dong ket thuc bang '\n', 115200 baud)
    Laptop -> ESP32
      D,seq,t_ms,x,y        phep do hop le: x,y (cm) tren tam, t_ms = thoi diem chup anh (ms)
      I,seq                 khong thay bi / khong thay tam  (INVALID)
      ARM | DISARM          bat / tat dieu khien vong kin (DISARM = tam ve phang)
      SP x y [vx vy]        diem dat (cm) va van toc dat (cm/s, dung cho bam quy dao)
      TILT x y              (chi khi DISARM) nghieng tam bang tay, don vi do, de hieu chuan
      SV d1 d2              (chi khi DISARM) dat THANG goc servo 1/2 (do, so voi trung tinh, +-30),
                            bo qua GEAR/MIX/LIM -> dung de khao sat servo <-> goc tam (servo_survey.py)
      KP KI KD LAT GEAR LIM ILIM ALPHA BETA <so>     chinh tham so
      MODE 0|1              0 = PID (mac dinh), 1 = SMC tich phan co lop bien (sliding mode)
      LAM KSW PHI KM KIS LATS LI   tham so SMC:  s = de + LAM*e + LI*int(e);  u = -(LAM*de + LI*e + KSW*sat(s/PHI))/KM + I
                            LAM [1/s], KSW [cm/s2] (do manh cong tac), PHI [cm/s] (be day lop bien),
                            KM [cm/s2 tren 1 do tam] (he so mo hinh bi; KM nho -> manh hon),
                            KIS = tich phan bu lech NGOAI mat truot (do/(cm*s)), LATS = thoi gian du doan (s), LI = tich phan trong s (mac dinh 0)
      BL1 BL2 <do servo>    BU DO RO servo 1/2 (backlash). 0 = tat. Thu 3.  Chi tac dung khi dang dieu khien.
      DITH <do>             rung nhe tam (do, 0 = tat, thu 0.3-0.6) de pha ma sat/do ro; KHONG luu NVS
      C1 <us> | C2 <us>     xung trung tinh servo 1/2 (us)
      MIX a b c d           ma tran tron truc: servo1 = a*ux + b*uy ; servo2 = c*ux + d*uy
      SAVE | RESET | ?      luu NVS / ve mac dinh / in tham so
    ESP32 -> Laptop
      S,state,t_ms,x,y,vx,vy,ux,uy,ix,iy     (20 Hz)  state: 0=DISARM 1=DIEU KHIEN 2=ARM nhung mat bi
      OK / ERR ...                           phan hoi lenh

  QUY UOC
    ux, uy = goc nghieng tam (do) de GIA TOC bi theo +X / +Y.  Chieu dung/sai do MIX quyet dinh.
    GEAR   = so do quay cua servo cho 1 do nghieng tam (phu thuoc tay don, do bang dien thoai).

  AN TOAN
    - Khoi dong o DISARM, tam ve phang (xung trung tinh C1/C2).
    - Mat du lieu > 150 ms hoac nhan goi I  ->  tam ve phang (+ gia tri I da hoc), cho bi xuat hien lai.
    - Gioi han nghieng LIM (do) va gioi han xung cung +-MAX_DEV_US quanh trung tinh.

  Board: ESP32S3 Dev Module.  Neu dung cong USB native: bat "USB CDC On Boot: Enabled".
  Chay duoc tren Arduino-ESP32 core 2.0.x va 3.x.
*/
#include <Arduino.h>
#include <Preferences.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// ============================== CAU HINH PHAN CUNG ==============================
#define PIN_SERVO1 4          // GPIO an toan tren moi bien the S3 (tranh 0,3,19,20,26-37,45,46)
#define PIN_SERVO2 5
#define SERVO_FREQ 50         // Hz
#define SERVO_RES  14         // bit (S3 toi da 14)
#define US_PER_DEG 10.0f      // 10 us / do servo (xap xi; sai so duoc GEAR hieu chuan bu)
#define MAX_DEV_US 350.0f     // gioi han CUNG quanh trung tinh (~35 do servo). Giam neu co khi bi ket.
#define SLEW_US_PER_TICK 60.0f // toc do doi xung toi da moi 20 ms (~300 do servo/s, MG995 khong nhanh hon)
#define SERIAL_BAUD 115200
#define TIMEOUT_MS 150        // het han du lieu -> mat bi
#define STATUS_PERIOD_MS 50   // 20 Hz

#ifndef ESP_ARDUINO_VERSION_MAJOR
#define ESP_ARDUINO_VERSION_MAJOR 2
#endif

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  #define PWM_ATTACH(pin, ch)       ledcAttach(pin, SERVO_FREQ, SERVO_RES)
  #define PWM_WRITE(pin, ch, duty)  ledcWrite(pin, duty)
#else
  #define PWM_ATTACH(pin, ch)       do { ledcSetup(ch, SERVO_FREQ, SERVO_RES); ledcAttachPin(pin, ch); } while (0)
  #define PWM_WRITE(pin, ch, duty)  ledcWrite(ch, duty)
#endif

// ============================== THAM SO ==============================
struct Params {
  uint32_t magic;
  float kp, ki, kd;      // do tam / cm ; do tam / (cm*s) ; do tam / (cm/s)
  float lat;             // s, bu tre camera: dung x + v*lat
  float alpha, beta;     // bo loc alpha-beta
  float gear;            // do servo / do tam
  float lim;             // do, nghieng toi da moi truc
  float ilim;            // do, gioi han thanh phan I
  float c1, c2;          // us trung tinh
  float m[4];            // ma tran tron
};
static const uint32_t PARAM_MAGIC = 0xB0B00002;

static Params P;
static void setDefaults() {
  P.magic = PARAM_MAGIC;
  P.kp = 0.30f; P.ki = 0.05f; P.kd = 0.40f;
  P.lat = 0.05f;
  P.alpha = 0.70f; P.beta = 0.33f;
  P.gear = 2.0f;
  P.lim = 6.0f;
  P.ilim = 5.0f;
  P.c1 = 1500.0f; P.c2 = 1500.0f;
  P.m[0] = 1; P.m[1] = 0; P.m[2] = 0; P.m[3] = 1;
}

static float g_dith = 0.0f;   // bien do rung (do tam), chi runtime, khong luu NVS

// Tham so mo rong (SMC + bu do ro). Luu rieng khoa "x" trong NVS -> khong lam mat C1/C2/GEAR/MIX da luu.
struct Extra { uint32_t magic; float mode, lam, ksw, phi, li, km, bl1, bl2, kis, lats; };
static const uint32_t EXTRA_MAGIC = 0xB0B0E002;
static Extra X;
static void setExtraDefaults() {
  X.magic = EXTRA_MAGIC; X.mode = 0.0f;
  X.lam = 1.48f; X.ksw = 73.2f; X.phi = 25.3f; X.li = 0.0f; X.km = 9.5f;    // toi uu trong mo phong co tre/ro/ma sat/lech phang
  X.kis = 0.19f; X.lats = 0.004f;                                          // tich phan bu lech (ngoai mat truot) + thoi gian du doan rieng cua SMC
  X.bl1 = 0.0f; X.bl2 = 0.0f;
}

struct FloatParam { const char* name; float* ptr; float lo, hi; };
static FloatParam FP[] = {
  {"KP",    &P.kp,    0.0f, 5.0f},
  {"KI",    &P.ki,    0.0f, 1.0f},
  {"KD",    &P.kd,    0.0f, 3.0f},
  {"LAT",   &P.lat,   0.0f, 0.3f},
  {"GEAR",  &P.gear,  0.3f, 8.0f},
  {"LIM",   &P.lim,   0.0f, 12.0f},
  {"ILIM",  &P.ilim,  0.0f, 8.0f},
  {"DITH",  &g_dith,  0.0f, 1.5f},
  {"ALPHA", &P.alpha, 0.05f, 1.0f},
  {"BETA",  &P.beta,  0.0f, 1.0f},
  {"C1",    &P.c1,    1000.0f, 2000.0f},
  {"C2",    &P.c2,    1000.0f, 2000.0f},
  {"MODE",  &X.mode,  0.0f, 1.0f},
  {"LAM",   &X.lam,   0.3f, 8.0f},
  {"KSW",   &X.ksw,   1.0f, 150.0f},
  {"PHI",   &X.phi,   1.0f, 80.0f},
  {"LI",    &X.li,    0.0f, 10.0f},
  {"KM",    &X.km,    3.0f, 30.0f},
  {"KIS",   &X.kis,   0.0f, 0.5f},
  {"LATS",  &X.lats,  0.0f, 0.3f},
  {"BL1",   &X.bl1,   0.0f, 8.0f},
  {"BL2",   &X.bl2,   0.0f, 8.0f},
};
static const int NFP = sizeof(FP) / sizeof(FP[0]);

static Preferences prefs;
static void saveParams() { prefs.begin("bop", false); prefs.putBytes("p", &P, sizeof(P)); prefs.putBytes("x", &X, sizeof(X)); prefs.end(); }
static void loadParams() {
  setDefaults();
  setExtraDefaults();
  prefs.begin("bop", false);
  if (prefs.getBytesLength("p") == sizeof(Params)) {
    Params t;
    prefs.getBytes("p", &t, sizeof(Params));
    if (t.magic == PARAM_MAGIC) P = t;
  }
  if (prefs.getBytesLength("x") == sizeof(Extra)) {
    Extra t;
    prefs.getBytes("x", &t, sizeof(Extra));
    if (t.magic == EXTRA_MAGIC) X = t;
  }
  prefs.end();
}

// ============================== TRANG THAI ==============================
struct AB { float x, v; };
static AB fx, fy;                       // bo loc alpha-beta moi truc
static float spx = 8.25f, spy = 5.0f;   // diem dat (cm) - tam 16.5 x 10
static float svx = 0, svy = 0;          // van toc dat (cm/s)
static float ix = 0, iy = 0;            // thanh phan tich phan (do tam)
static float mx = 0, my = 0;            // nghieng thu cong khi DISARM (do)
static float gx = 0, gy = 0;            // lenh nghieng hien hanh (do)
static bool  svMode = false;            // che do SV: dat thang goc servo (chi khi DISARM)
static float sv1 = 0, sv2 = 0;          // goc servo (do) cua che do SV
static float lastX = 0, lastY = 0;      // do tho gan nhat (de telemetry)
static float stuckX = 0, stuckY = 0;    // thoi gian (s) bi dung yen khi con lech diem dat
static float qx = 0, qy = 0;            // tich phan sai so trong mat truot SMC (cm*s)
static const float KI_BOOST = 4.0f;     // he so tang KI khi bi 'ket'

static bool armed = false;
static bool ballOK = false;
static int  validStreak = 0;
static bool haveLastT = false;
static uint32_t lastT = 0;              // t_ms cua goi D truoc (do laptop gui)
static uint32_t lastValidMs = 0;
static float curUs1, curUs2;            // xung dang xuat (sau slew)

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline bool ctrlActive() { return armed && ballOK && validStreak >= 3; }

// ============================== PWM ==============================
static void writeUs(int pin, int ch, float us) {
  uint32_t duty = (uint32_t)(us * (float)(1u << SERVO_RES) / 20000.0f + 0.5f);
  PWM_WRITE(pin, ch, duty);
}

static void updateServos() {
  // chon lenh nghieng
  if (!armed)               { gx = clampf(mx, -P.lim, P.lim); gy = clampf(my, -P.lim, P.lim); }
  else if (!ctrlActive())   { gx = clampf(ix, -P.lim, P.lim); gy = clampf(iy, -P.lim, P.lim); }
  // tron truc -> goc servo -> xung
  // rung nhe (dither) chi khi dang dieu khien: vuot do ro / ma sat tinh cua bi
  static uint8_t ditCnt = 0; static bool ditPh = false;
  float dd = 0.0f;
  if (ctrlActive() && g_dith > 0.0f) {
    if (++ditCnt >= 3) { ditCnt = 0; ditPh = !ditPh; }   // doi pha moi 60 ms -> ~8 Hz
    dd = ditPh ? g_dith : -g_dith;
  }
  float d1, d2;
  if (!armed && svMode) {
    d1 = sv1 * US_PER_DEG;
    d2 = sv2 * US_PER_DEG;
  } else {
    d1 = P.gear * (P.m[0] * (gx + dd) + P.m[1] * (gy + dd)) * US_PER_DEG;
    d2 = P.gear * (P.m[2] * (gx + dd) + P.m[3] * (gy + dd)) * US_PER_DEG;
  }
  // BU DO RO (backlash inverse): theo doi chieu chuyen dong cua lenh tung servo, cong them nua do ro
  // theo chieu do -> banh rang chuyen phia ngay khi lenh doi huong, khong bi 'nuot' lenh nho.
  static float lastD1 = 0, lastD2 = 0, dir1 = 0, dir2 = 0;
  const float BL_EPS_US = 4.0f;                          // 0.4 do servo: nguong doi chieu
  if (ctrlActive() && (X.bl1 > 0.0f || X.bl2 > 0.0f)) {
    if (d1 - lastD1 > BL_EPS_US) { dir1 = 1.0f; lastD1 = d1; } else if (lastD1 - d1 > BL_EPS_US) { dir1 = -1.0f; lastD1 = d1; }
    if (d2 - lastD2 > BL_EPS_US) { dir2 = 1.0f; lastD2 = d2; } else if (lastD2 - d2 > BL_EPS_US) { dir2 = -1.0f; lastD2 = d2; }
    d1 += dir1 * X.bl1 * US_PER_DEG * 0.5f;
    d2 += dir2 * X.bl2 * US_PER_DEG * 0.5f;
  } else {
    lastD1 = d1; lastD2 = d2; dir1 = 0; dir2 = 0;
  }
  float t1 = P.c1 + clampf(d1, -MAX_DEV_US, MAX_DEV_US);
  float t2 = P.c2 + clampf(d2, -MAX_DEV_US, MAX_DEV_US);
  // gioi han toc do
  curUs1 += clampf(t1 - curUs1, -SLEW_US_PER_TICK, SLEW_US_PER_TICK);
  curUs2 += clampf(t2 - curUs2, -SLEW_US_PER_TICK, SLEW_US_PER_TICK);
  // ghi CA HAI kenh cung mot lan
  writeUs(PIN_SERVO1, 0, curUs1);
  writeUs(PIN_SERVO2, 1, curUs2);
}

// ============================== DIEU KHIEN ==============================
// Luu y: KHONG dung kieu struct tu dinh nghia lam tham so ham (Arduino IDE tu chen prototype len tren struct -> loi bien dich)
static void abInit(float& px, float& pv, float z) { px = z; pv = 0; }
static void abUpdate(float& px, float& pv, float z, float dt) {
  float xp = px + pv * dt;
  float r = z - xp;
  px = xp + P.alpha * r;
  pv = pv + (P.beta / dt) * r;
}

static void resetIntegrators() { ix = 0; iy = 0; stuckX = 0; stuckY = 0; qx = 0; qy = 0; }

static void markLost() {
  ballOK = false;
  validStreak = 0;
  haveLastT = false;
}

// SMC lop bien + tich phan BU LECH ngoai mat truot, cho 1 truc.  e = x - r (cm), de = v - rd (cm/s).
//   s = de + LAM*e + LI*q ;  u = -(LAM*de + LI*e + KSW*sat(s/PHI))/KM + I ;  I += -KIS*e*dt
// Tai sao I nam NGOAI: thanh phan chuyen mach bi chan o KSW/KM do, khong thang noi lech phang lon (>KSW/KM);
// tich phan ben trong mat truot khong bu duoc -> bi ket o bien (da gap trong thu nghiem). Bo tich phan ngoai thi khong bi chan.
// Chi dung tham so float (khong dung struct tu dinh nghia) de Arduino IDE khong bi loi prototype.
static void smcAxis(float xs, float v, float r, float rd, float dt, float stuck, float& q, float& I, float& u) {
  float e = xs - r, de = v - rd;
  float s = de + X.lam * e + X.li * q;
  float sat = clampf(s / X.phi, -1.0f, 1.0f);
  float un = -(X.lam * de + X.li * e + X.ksw * sat) / X.km;          // do tam; x'' = KM * u + nhieu
  bool satU = (un + I > P.lim && e < 0) || (un + I < -P.lim && e > 0);   // chong windup
  if (fabsf(de) < 8.0f && !satU) {
    float boost = (stuck > 0.4f) ? KI_BOOST : 1.0f;
    I = clampf(I - boost * X.kis * e * dt, -P.ilim, P.ilim);
    if (X.li > 0.0f) { float qmax = X.km * P.ilim / X.li; q = clampf(q + boost * e * dt, -qmax, qmax); }
  }
  u = clampf(un + I, -P.lim, P.lim);
}

static void controlStep(float dt) {
  // du doan bu tre camera
  float xs = fx.x + fx.v * P.lat;
  float ys = fy.x + fy.v * P.lat;
  float ex = spx - xs, ey = spy - ys;

  if (X.mode > 0.5f) {                                  // ---- SMC ----
    stuckX = (fabsf(ex) > 0.8f && fabsf(fx.v) < 1.5f) ? stuckX + dt : 0.0f;
    stuckY = (fabsf(ey) > 0.8f && fabsf(fy.v) < 1.5f) ? stuckY + dt : 0.0f;
    float xss = fx.x + fx.v * X.lats, yss = fy.x + fy.v * X.lats;       // du doan rieng cua SMC
    smcAxis(xss, fx.v, spx, svx, dt, stuckX, qx, ix, gx);
    smcAxis(yss, fy.v, spy, svy, dt, stuckY, qy, iy, gy);
    return;
  }

  // PD (dao ham theo van toc do - van toc dat)
  float ux = P.kp * ex + P.kd * (svx - fx.v);
  float uy = P.kp * ey + P.kd * (svy - fy.v);

  // I: tich phan o MOI khoang sai so (tam chi 16.5 x 10 cm, khong duoc tat I khi bi xa diem dat,
  // neu khong bi se ket o cho lech phang). Chi tich phan khi (a) bi dang cham (<8 cm/s) va
  // (b) lenh chua bao hoa theo cung chieu (chong windup).
  // 'KET': bi dung yen (<1.5 cm/s) ma van lech > 0.8 cm trong > 0.4 s -> I chay nhanh gap KI_BOOST lan.
  stuckX = (fabsf(ex) > 0.8f && fabsf(fx.v) < 1.5f) ? stuckX + dt : 0.0f;
  stuckY = (fabsf(ey) > 0.8f && fabsf(fy.v) < 1.5f) ? stuckY + dt : 0.0f;
  float kix = P.ki * (stuckX > 0.4f ? KI_BOOST : 1.0f);
  float kiy = P.ki * (stuckY > 0.4f ? KI_BOOST : 1.0f);
  if (fabsf(fx.v) < 8.0f && !((ux + ix > P.lim && ex > 0) || (ux + ix < -P.lim && ex < 0)))
    ix = clampf(ix + kix * ex * dt, -P.ilim, P.ilim);
  if (fabsf(fy.v) < 8.0f && !((uy + iy > P.lim && ey > 0) || (uy + iy < -P.lim && ey < 0)))
    iy = clampf(iy + kiy * ey * dt, -P.ilim, P.ilim);

  gx = clampf(ux + ix, -P.lim, P.lim);
  gy = clampf(uy + iy, -P.lim, P.lim);
}

// ============================== GOI DU LIEU ==============================
static void handleD(char** tok, int n) {
  if (n < 5) return;
  uint32_t tms = (uint32_t)strtoul(tok[2], NULL, 10);
  float x = atof(tok[3]), y = atof(tok[4]);
  if (!isfinite(x) || !isfinite(y) || fabsf(x) > 60.0f || fabsf(y) > 60.0f) return;

  lastValidMs = millis();
  lastX = x; lastY = y;
  float dt = haveLastT ? (float)((uint32_t)(tms - lastT)) / 1000.0f : 0.0f;

  if (!haveLastT || !ballOK || dt > 0.25f) {     // bat dau lai: khoi tao bo loc
    abInit(fx.x, fx.v, x); abInit(fy.x, fy.v, y);
    haveLastT = true; lastT = tms; ballOK = true; validStreak = 1;
    return;
  }
  if (dt < 0.004f) return;                        // goi trung / qua sat nhau
  lastT = tms;
  abUpdate(fx.x, fx.v, x, dt);
  abUpdate(fy.x, fy.v, y, dt);
  if (validStreak < 1000) validStreak++;
  if (ctrlActive()) controlStep(dt);
}

// ============================== LENH ==============================
static void reply(const char* s) { Serial.println(s); }

static void printParams() {
  char b[256];
  snprintf(b, sizeof(b), "P,KP=%.3f KI=%.3f KD=%.3f LAT=%.3f GEAR=%.2f LIM=%.1f ILIM=%.1f ALPHA=%.2f BETA=%.2f",
           P.kp, P.ki, P.kd, P.lat, P.gear, P.lim, P.ilim, P.alpha, P.beta);
  Serial.println(b);
  snprintf(b, sizeof(b), "P,C1=%.0f C2=%.0f MIX=%.2f %.2f %.2f %.2f SP=%.2f,%.2f armed=%d",
           P.c1, P.c2, P.m[0], P.m[1], P.m[2], P.m[3], spx, spy, armed ? 1 : 0);
  Serial.println(b);
  snprintf(b, sizeof(b), "P,MODE=%d LAM=%.2f KSW=%.1f PHI=%.1f LI=%.2f KM=%.1f KIS=%.3f LATS=%.3f BL1=%.1f BL2=%.1f",
           X.mode > 0.5f ? 1 : 0, X.lam, X.ksw, X.phi, X.li, X.km, X.kis, X.lats, X.bl1, X.bl2);
  Serial.println(b);
}

static int splitLine(char* s, char** tok, int maxTok) {
  int n = 0;
  char* p = strtok(s, " ,\t");
  while (p && n < maxTok) { tok[n++] = p; p = strtok(NULL, " ,\t"); }
  return n;
}

static void handleLine(char* line) {
  char* tok[8];
  int n = splitLine(line, tok, 8);
  if (n == 0) return;
  for (char* c = tok[0]; *c; ++c) *c = toupper((unsigned char)*c);
  const char* cmd = tok[0];

  if (!strcmp(cmd, "D"))  { handleD(tok, n); return; }
  if (!strcmp(cmd, "I"))  { markLost(); return; }

  if (!strcmp(cmd, "ARM")) {
    resetIntegrators(); markLost(); svMode = false; armed = true; reply("OK ARM"); return;
  }
  if (!strcmp(cmd, "DISARM")) { armed = false; svMode = false; resetIntegrators(); reply("OK DISARM"); return; }
  if (!strcmp(cmd, "SP")) {
    if (n < 3) { reply("ERR SP x y [vx vy]"); return; }
    float x = atof(tok[1]), y = atof(tok[2]);
    if (x < 0 || x > 16.5f || y < 0 || y > 10.0f) { reply("ERR SP ngoai tam"); return; }
    spx = x; spy = y;
    svx = (n >= 5) ? atof(tok[3]) : 0.0f;
    svy = (n >= 5) ? atof(tok[4]) : 0.0f;
    return;                                        // khong phan hoi (co the gui 30 Hz)
  }
  if (!strcmp(cmd, "TILT")) {
    if (armed) { reply("ERR DISARM truoc khi TILT"); return; }
    if (n < 3) { reply("ERR TILT x y"); return; }
    mx = atof(tok[1]); my = atof(tok[2]);
    svMode = false;
    reply("OK TILT"); return;
  }
  if (!strcmp(cmd, "SV")) {
    if (armed) { reply("ERR DISARM truoc khi SV"); return; }
    if (n < 3) { reply("ERR SV d1 d2 (do servo)"); return; }
    sv1 = clampf(atof(tok[1]), -30.0f, 30.0f);
    sv2 = clampf(atof(tok[2]), -30.0f, 30.0f);
    svMode = true;
    reply("OK SV"); return;
  }
  if (!strcmp(cmd, "MIX")) {
    if (n < 5) { reply("ERR MIX a b c d"); return; }
    for (int i = 0; i < 4; i++) P.m[i] = clampf(atof(tok[1 + i]), -3.0f, 3.0f);
    reply("OK MIX"); return;
  }
  if (!strcmp(cmd, "SAVE"))  { saveParams(); reply("OK SAVED"); return; }
  if (!strcmp(cmd, "RESET")) { setDefaults(); setExtraDefaults(); resetIntegrators(); reply("OK DEFAULTS (chua SAVE)"); return; }
  if (!strcmp(cmd, "?") || !strcmp(cmd, "PARAMS")) { printParams(); return; }

  for (int i = 0; i < NFP; i++) {
    if (!strcmp(cmd, FP[i].name)) {
      if (n < 2) { reply("ERR thieu gia tri"); return; }
      float v = atof(tok[1]);
      if (!isfinite(v) || v < FP[i].lo || v > FP[i].hi) { reply("ERR ngoai khoang cho phep"); return; }
      *FP[i].ptr = v;
      if (!strcmp(cmd, "MODE")) resetIntegrators();       // doi che do -> xoa tich phan
      reply("OK");
      return;
    }
  }
  reply("ERR lenh la");
}

static void pollSerial() {
  static char buf[96];
  static int len = 0;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) { buf[len] = 0; handleLine(buf); len = 0; }
    } else if (len < (int)sizeof(buf) - 1) {
      buf[len++] = c;
    } else {
      len = 0;                                     // dong qua dai -> bo
    }
  }
}

static void sendStatus() {
  int state = !armed ? 0 : (ctrlActive() ? 1 : 2);
  char b[160];
  int n = snprintf(b, sizeof(b), "S,%d,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                   state, (unsigned long)millis(), fx.x, fy.x, fx.v, fy.v, gx, gy, ix, iy);
  Serial.write((const uint8_t*)b, n);
}

// ============================== SETUP / LOOP ==============================
void setup() {
  Serial.setRxBufferSize(512);
  Serial.begin(SERIAL_BAUD);
  loadParams();

  curUs1 = P.c1; curUs2 = P.c2;
  PWM_ATTACH(PIN_SERVO1, 0);
  PWM_ATTACH(PIN_SERVO2, 1);
  writeUs(PIN_SERVO1, 0, curUs1);                  // ca hai ve trung tinh cung luc
  writeUs(PIN_SERVO2, 1, curUs2);
  delay(300);
  Serial.println("BOOT ball_plate_esp32s3 DISARMED");
  printParams();
}

void loop() {
  pollSerial();

  uint32_t now = millis();
  if (ballOK && (now - lastValidMs) > TIMEOUT_MS) markLost();   // watchdog du lieu

  static uint32_t tServo = 0, tStatus = 0;
  if (now - tServo >= 20)            { tServo = now;  updateServos(); }
  if (now - tStatus >= STATUS_PERIOD_MS) { tStatus = now; sendStatus(); }
}
