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

// =========================
// Send frequency
// =========================
static const float SEND_HZ = 100.0f;   // cambia aqui la frecuencia de envio
static const uint32_t SEND_PERIOD_US = (uint32_t)(1000000.0f / SEND_HZ);

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

struct Vec3f {
  float x;
  float y;
  float z;
};

uint32_t lastSendUs = 0;

// =========================
// Arduino
// =========================
void setup() {
  Serial.begin(115200);
  Wire.begin(9, 8, 100000);

  setupMpu();

  lastSendUs = micros();

  Serial.println("setup");
}

void loop() {
  const uint32_t nowUs = micros();

  if ((uint32_t)(nowUs - lastSendUs) < SEND_PERIOD_US) {
    return;
  }

  lastSendUs += SEND_PERIOD_US;

  int16_t axRaw, ayRaw, azRaw;
  int16_t gxRaw, gyRaw, gzRaw;

  if (!readMpuRaw(axRaw, ayRaw, azRaw, gxRaw, gyRaw, gzRaw)) {
    Serial.println("read_error");
    return;
  }

  Vec3f accelRaw;
  accelRaw.x = ((float)axRaw / ACCEL_SENS) * G_TO_MS2;
  accelRaw.y = ((float)ayRaw / ACCEL_SENS) * G_TO_MS2;
  accelRaw.z = ((float)azRaw / ACCEL_SENS) * G_TO_MS2;

  Vec3f gyroRaw;
  gyroRaw.x = ((float)gxRaw / GYRO_SENS) * DEG2RAD_F;
  gyroRaw.y = ((float)gyRaw / GYRO_SENS) * DEG2RAD_F;
  gyroRaw.z = ((float)gzRaw / GYRO_SENS) * DEG2RAD_F;

  Serial.print(micros() * 1e-3, 6); Serial.print(",");
  Serial.print(gyroRaw.x, 6); Serial.print(",");
  Serial.print(gyroRaw.y, 6); Serial.print(",");
  Serial.print(gyroRaw.z, 6); Serial.print(",");
  Serial.print(accelRaw.x, 6); Serial.print(",");
  Serial.print(accelRaw.y, 6); Serial.print(",");
  Serial.print(accelRaw.z, 6); Serial.print("\r\n");
}