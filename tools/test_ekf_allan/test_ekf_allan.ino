#include <Wire.h>
#include <Arduino.h>

#define MPU_ADDR 0x68

// =========================
// MPU6050 registers
// =========================
#define REG_SMPLRT_DIV    0x19
#define REG_CONFIG        0x1A
#define REG_GYRO_CONFIG   0x1B
#define REG_ACCEL_CONFIG  0x1C
#define REG_ACCEL_XOUT_H  0x3B
#define REG_PWR_MGMT_1    0x6B

// =========================
// Units and scales
// =========================
static const float ACCEL_SENS = 16384.0f;   // +/-2g
static const float GYRO_SENS  = 131.0f;     // +/-250 deg/s
static const float G_TO_MS2   = 9.80665f;
static const float DEG2RAD_F  = 3.14159265358979323846f / 180.0f;
static const float RAD2DEG_F  = 180.0f / 3.14159265358979323846f;

// =========================
// Config from YAML
// =========================
static const float IMU_FREQ_HZ = 50.0f;

// Fixed bias
static const float BGX = -3.953998335645e-02f;
static const float BGY = -1.286214337937e-02f;
static const float BGZ = -4.484531785842e-02f;

static const float BAX =  2.926474101974e-02f;
static const float BAY = -7.168433031569e-02f;
static const float BAZ =  2.529648889327e-02f;

// Allan gyro [N, B, K]
static const float GX_N = 9.947163643841e-05f;
static const float GX_B = 6.866465271708e-05f;
static const float GX_K = 1.696116599080e-06f;

static const float GY_N = 1.241490345774e-04f;
static const float GY_B = 7.945412208038e-05f;
static const float GY_K = 1.985035213080e-06f;

// Allan accel [N only used]
static const float AX_N = 2.922703675203e-03f;
static const float AY_N = 2.877209498495e-03f;
static const float AZ_N = 3.963057979962e-03f;

// =========================
// Tuning
// =========================
static const float COMP_ALPHA = 0.98f;
static const float GATE_G_ERR_MS2 = 1.0f;
static const float R_ACCEL_SCALE = 5.0f;
static const uint32_t PRINT_EVERY = 5;

// "Simple" Kalman tuning without Allan model
static const float KF_SIMPLE_Q_ANGLE = 1e-4f;
static const float KF_SIMPLE_Q_BIAS  = 1e-5f;
static const float KF_SIMPLE_R_MEAS  = 1e-2f;

// =========================
// Simple structs
// =========================
struct Vec3f {
  float x;
  float y;
  float z;
};

// =========================
// Utility
// =========================
static inline float qAngleFromAllan(float N_g, float dt) {
  return N_g * N_g * dt;
}

static inline float qBiasFromAllan(float B_g, float K_g, float dt) {
  return (B_g * B_g + K_g * K_g * dt) * dt;
}

static inline float rAccelFromAllan(float N_a, float fs) {
  float sigma_sample = N_a * sqrtf(fs);
  float sigma_angle = sigma_sample / G_TO_MS2;
  return sigma_angle * sigma_angle;
}

// =========================
// Complementary filter (simple, no noise model)
// =========================
class ImuComplementarySimple {
public:
  void update(const Vec3f &gyroCal, const Vec3f &accelCal, float dt) {
    float an = sqrtf(accelCal.x*accelCal.x + accelCal.y*accelCal.y + accelCal.z*accelCal.z);
    if (an < 1e-6f) return;

    float rollAcc  = atan2f(accelCal.y, accelCal.z);
    float pitchAcc = atan2f(-accelCal.x, sqrtf(accelCal.y * accelCal.y + accelCal.z * accelCal.z));

    if (!initialized) {
      roll = rollAcc;
      pitch = pitchAcc;
      yaw = 0.0f;
      initialized = true;
      return;
    }

    float rollGyro  = roll  + gyroCal.x * dt;
    float pitchGyro = pitch + gyroCal.y * dt;
    float yawGyro   = yaw   + gyroCal.z * dt;

    roll  = COMP_ALPHA * rollGyro  + (1.0f - COMP_ALPHA) * rollAcc;
    pitch = COMP_ALPHA * pitchGyro + (1.0f - COMP_ALPHA) * pitchAcc;
    yaw   = yawGyro;
  }

  float roll = 0.0f;
  float pitch = 0.0f;
  float yaw = 0.0f;

private:
  bool initialized = false;
};

// =========================
// 1D Kalman for angle + bias
// =========================
class KalmanAngle1D {
public:
  void begin(float q_angle_in, float q_bias_in, float r_measure_in) {
    q_angle = q_angle_in;
    q_bias = q_bias_in;
    r_measure = r_measure_in;

    angle = 0.0f;
    bias = 0.0f;

    P00 = 1e-3f;
    P01 = 0.0f;
    P10 = 0.0f;
    P11 = 1e-3f;

    initialized = false;
  }

  void setAngle(float a) {
    angle = a;
    initialized = true;
  }

  float update(float gyroRate, float accelAngle, float dt) {
    if (!initialized) {
      setAngle(accelAngle);
    }

    float rate = gyroRate - bias;
    angle += dt * rate;

    P00 += dt * (dt * P11 - P01 - P10 + q_angle);
    P01 -= dt * P11;
    P10 -= dt * P11;
    P11 += q_bias * dt;

    float y = accelAngle - angle;
    float S = P00 + r_measure;
    float K0 = P00 / S;
    float K1 = P10 / S;

    angle += K0 * y;
    bias  += K1 * y;

    float P00_temp = P00;
    float P01_temp = P01;

    P00 -= K0 * P00_temp;
    P01 -= K0 * P01_temp;
    P10 -= K1 * P00_temp;
    P11 -= K1 * P01_temp;

    return angle;
  }

  float getAngle() const { return angle; }
  float getBias() const { return bias; }

private:
  bool initialized = false;

  float angle = 0.0f;
  float bias  = 0.0f;

  float q_angle = 0.0f;
  float q_bias = 0.0f;
  float r_measure = 0.0f;

  float P00 = 0.0f, P01 = 0.0f, P10 = 0.0f, P11 = 0.0f;
};

// =========================
// Roll/Pitch Kalman wrapper
// =========================
class ImuKalmanRP {
public:
  void beginAllan(float fs) {
    float dt = 1.0f / fs;

    float q_roll  = 5.0f * qAngleFromAllan(GX_N, dt);
    float q_pitch = 5.0f * qAngleFromAllan(GY_N, dt);

    float q_bias_roll  = 5.0f * qBiasFromAllan(GX_B, GX_K, dt);
    float q_bias_pitch = 5.0f * qBiasFromAllan(GY_B, GY_K, dt);

    float r_roll  = R_ACCEL_SCALE * rAccelFromAllan(AY_N, fs);
    float r_pitch = R_ACCEL_SCALE * rAccelFromAllan(AX_N, fs);

    kRoll.begin(q_roll, q_bias_roll, r_roll);
    kPitch.begin(q_pitch, q_bias_pitch, r_pitch);
  }

  void beginSimple() {
    kRoll.begin(KF_SIMPLE_Q_ANGLE, KF_SIMPLE_Q_BIAS, KF_SIMPLE_R_MEAS);
    kPitch.begin(KF_SIMPLE_Q_ANGLE, KF_SIMPLE_Q_BIAS, KF_SIMPLE_R_MEAS);
  }

  void update(const Vec3f &gyroRaw, const Vec3f &accelRaw, float dt) {
    gyroCal.x = gyroRaw.x - BGX;
    gyroCal.y = gyroRaw.y - BGY;
    gyroCal.z = gyroRaw.z - BGZ;

    accelCal.x = accelRaw.x - BAX;
    accelCal.y = accelRaw.y - BAY;
    accelCal.z = accelRaw.z - BAZ;

    float an = sqrtf(accelCal.x*accelCal.x + accelCal.y*accelCal.y + accelCal.z*accelCal.z);
    float g_err = fabsf(an - G_TO_MS2);

    float rollAcc = roll;
    float pitchAcc = pitch;

    if (an > 1e-6f) {
      rollAcc  = atan2f(accelCal.y, accelCal.z);
      pitchAcc = atan2f(-accelCal.x, sqrtf(accelCal.y * accelCal.y + accelCal.z * accelCal.z));
    }

    if (an > 1e-6f && g_err < GATE_G_ERR_MS2) {
      roll  = kRoll.update(gyroCal.x, rollAcc, dt);
      pitch = kPitch.update(gyroCal.y, pitchAcc, dt);
    } else {
      roll  += (gyroCal.x - kRoll.getBias()) * dt;
      pitch += (gyroCal.y - kPitch.getBias()) * dt;
    }

    yaw += gyroCal.z * dt;
  }

  Vec3f getGyroCal() const { return gyroCal; }
  Vec3f getAccelCal() const { return accelCal; }

  float roll = 0.0f;
  float pitch = 0.0f;
  float yaw = 0.0f;

private:
  KalmanAngle1D kRoll;
  KalmanAngle1D kPitch;

  Vec3f gyroCal {0,0,0};
  Vec3f accelCal {0,0,0};
};

// =========================
// MPU6050 low-level
// =========================
void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

bool readRegisters(uint8_t startReg, uint8_t count, uint8_t* dest) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) return false;

  uint8_t n = Wire.requestFrom((int)MPU_ADDR, (int)count, (int)true);
  if (n != count) return false;

  for (uint8_t i = 0; i < count; ++i) dest[i] = Wire.read();
  return true;
}

int16_t makeInt16(uint8_t hi, uint8_t lo) {
  return (int16_t)((hi << 8) | lo);
}

bool readMpuRaw(int16_t& axRaw, int16_t& ayRaw, int16_t& azRaw,
                int16_t& gxRaw, int16_t& gyRaw, int16_t& gzRaw) {
  uint8_t data[14];
  if (!readRegisters(REG_ACCEL_XOUT_H, 14, data)) return false;

  axRaw = makeInt16(data[0],  data[1]);
  ayRaw = makeInt16(data[2],  data[3]);
  azRaw = makeInt16(data[4],  data[5]);
  gxRaw = makeInt16(data[8],  data[9]);
  gyRaw = makeInt16(data[10], data[11]);
  gzRaw = makeInt16(data[12], data[13]);
  return true;
}

void setupMpu() {
  writeRegister(REG_PWR_MGMT_1, 0x00);
  delay(100);

  writeRegister(REG_SMPLRT_DIV, 9);
  writeRegister(REG_CONFIG, 0x04);
  writeRegister(REG_GYRO_CONFIG, 0x00);
  writeRegister(REG_ACCEL_CONFIG, 0x00);
  delay(50);
}

// =========================
// Global filters
// =========================
ImuKalmanRP kalmanAllan;
ImuKalmanRP kalmanSimple;
ImuComplementarySimple compSimple;

unsigned long lastMicrosLoop = 0;
uint32_t printCounter = 0;

// =========================
// Arduino
// =========================
void setup() {
  Serial.begin(115200);
  Wire.begin(9, 8, 100000);

  setupMpu();

  kalmanAllan.beginAllan(IMU_FREQ_HZ);
  kalmanSimple.beginSimple();

  lastMicrosLoop = micros();

  Serial.println("setup");
}

void loop() {
  int16_t axRaw, ayRaw, azRaw;
  int16_t gxRaw, gyRaw, gzRaw;

  if (!readMpuRaw(axRaw, ayRaw, azRaw, gxRaw, gyRaw, gzRaw)) {
    Serial.println("read_error");
    delay(10);
    return;
  }

  unsigned long now = micros();
  float dt = (now - lastMicrosLoop) * 1e-6f;
  lastMicrosLoop = now;

  if (dt <= 0.0f || dt > 0.2f) {
    dt = 1.0f / IMU_FREQ_HZ;
  }

  Vec3f accelRaw;
  accelRaw.x = ((float)axRaw / ACCEL_SENS) * G_TO_MS2;
  accelRaw.y = ((float)ayRaw / ACCEL_SENS) * G_TO_MS2;
  accelRaw.z = ((float)azRaw / ACCEL_SENS) * G_TO_MS2;

  Vec3f gyroRaw;
  gyroRaw.x = ((float)gxRaw / GYRO_SENS) * DEG2RAD_F;
  gyroRaw.y = ((float)gyRaw / GYRO_SENS) * DEG2RAD_F;
  gyroRaw.z = ((float)gzRaw / GYRO_SENS) * DEG2RAD_F;

  kalmanAllan.update(gyroRaw, accelRaw, dt);
  kalmanSimple.update(gyroRaw, accelRaw, dt);

  Vec3f gyroCal = kalmanSimple.getGyroCal();
  Vec3f accelCal = kalmanSimple.getAccelCal();
  compSimple.update(gyroCal, accelCal, dt);

  if ((printCounter++ % PRINT_EVERY) == 0) {
    // Serial.print(">ax_cal:");
    // Serial.println(accelCal.x, 6);

    // Serial.print(">ay_cal:");
    // Serial.println(accelCal.y, 6);

    // Serial.print(">az_cal:");
    // Serial.println(accelCal.z, 6);

    Serial.print(">roll_kf_allan:");
    Serial.println(kalmanAllan.roll * RAD2DEG_F, 6);

    Serial.print(">pitch_kf_allan:");
    Serial.println(kalmanAllan.pitch * RAD2DEG_F, 6);

    Serial.print(">yaw_kf_allan:");
    Serial.println(kalmanAllan.yaw * RAD2DEG_F, 6);

    Serial.print(">roll_kf_simple:");
    Serial.println(kalmanSimple.roll * RAD2DEG_F, 6);

    Serial.print(">pitch_kf_simple:");
    Serial.println(kalmanSimple.pitch * RAD2DEG_F, 6);

    Serial.print(">yaw_kf_simple:");
    Serial.println(kalmanSimple.yaw * RAD2DEG_F, 6);

    Serial.print(">roll_comp_simple:");
    Serial.println(compSimple.roll * RAD2DEG_F, 6);

    Serial.print(">pitch_comp_simple:");
    Serial.println(compSimple.pitch * RAD2DEG_F, 6);

    Serial.print(">yaw_comp_simple:");
    Serial.println(compSimple.yaw * RAD2DEG_F, 6);

    // Serial.print(">dt:");
    // Serial.println(dt, 6);
  }

  delay(5);
}