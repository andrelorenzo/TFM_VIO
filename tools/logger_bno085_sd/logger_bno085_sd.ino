#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>

// =========================
// Pines
// =========================
#define I2C_SCL 8
#define I2C_SDA 9

#define SD_CS   10
#define SD_MOSI 11
#define SD_SCK  12
#define SD_MISO 13

#define RGB_LED_PIN 48

// =========================
// MPU6050 / GY-521
// =========================
#define MPU6050_ADDR 0x68

#define MPU6050_REG_PWR_MGMT_1   0x6B
#define MPU6050_REG_SMPLRT_DIV   0x19
#define MPU6050_REG_CONFIG       0x1A
#define MPU6050_REG_GYRO_CONFIG  0x1B
#define MPU6050_REG_ACCEL_CONFIG 0x1C
#define MPU6050_REG_ACCEL_XOUT_H 0x3B

// =========================
// Frecuencia
// =========================
const uint16_t SAMPLE_RATE_HZ = 100;
const uint32_t REPORT_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

// =========================
// Segmentación de archivos
// =========================
const uint32_t FILE_DURATION_MS = 3600000UL;  // 1 hora por CSV

// =========================
// Escalas MPU6050
// =========================
const float MPU_ACCEL_LSB_PER_G    = 4096.0f;
const float MPU_GYRO_LSB_PER_DEG_S = 65.5f;
const float GRAVITY_MS2            = 9.80665f;
const float DEG_TO_RAD_F           = 0.017453292519943295769f;

// =========================
// Umbral de nivelado en m/s²
// =========================
const float LEVEL_TOL_MS2 = 0.05f;

// =========================
// Variables globales
// =========================
File logFile;

float mpu_ax = 0.0f, mpu_ay = 0.0f, mpu_az = 0.0f;
float mpu_gx = 0.0f, mpu_gy = 0.0f, mpu_gz = 0.0f;
float mpu_temp_c = 0.0f;

uint32_t lastLogUs = 0;          // solo para temporización
uint64_t elapsedLogUs = 0;       // tiempo acumulado real para guardar en CSV
uint32_t linesSinceFlush = 0;
uint32_t fileStartMs = 0;
uint16_t currentFileIndex = 0;

// =========================
// LED helpers
// =========================
void setLedOff() {
  neopixelWrite(RGB_LED_PIN, 0, 0, 0);
}

void setLedRed() {
  neopixelWrite(RGB_LED_PIN, 32, 0, 0);
}

void setLedGreen() {
  neopixelWrite(RGB_LED_PIN, 0, 32, 0);
}

void setLedOrange() {
  neopixelWrite(RGB_LED_PIN, 32, 12, 0);
}

// =========================
// Helpers generales
// =========================
void fatalError(const char* msg) {
  Serial.println(msg);
  setLedRed();
  while (1) {
    delay(100);
  }
}

bool i2cWrite8(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool i2cReadBytes(uint8_t addr, uint8_t reg, uint8_t* buffer, size_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom((int)addr, (int)len) != (int)len) {
    return false;
  }

  for (size_t i = 0; i < len; i++) {
    buffer[i] = Wire.read();
  }
  return true;
}

int16_t makeInt16(uint8_t hi, uint8_t lo) {
  return (int16_t)((hi << 8) | lo);
}

float absf(float x) {
  return x >= 0.0f ? x : -x;
}

// =========================
// MPU6050
// =========================
bool initMPU6050() {
  if (!i2cWrite8(MPU6050_ADDR, MPU6050_REG_PWR_MGMT_1, 0x00)) return false;
  delay(100);

  if (!i2cWrite8(MPU6050_ADDR, MPU6050_REG_SMPLRT_DIV, 9)) return false;
  if (!i2cWrite8(MPU6050_ADDR, MPU6050_REG_CONFIG, 0x02)) return false;
  if (!i2cWrite8(MPU6050_ADDR, MPU6050_REG_GYRO_CONFIG, 0x08)) return false;
  if (!i2cWrite8(MPU6050_ADDR, MPU6050_REG_ACCEL_CONFIG, 0x10)) return false;

  return true;
}

bool readMPU6050() {
  uint8_t buf[14];

  if (!i2cReadBytes(MPU6050_ADDR, MPU6050_REG_ACCEL_XOUT_H, buf, 14)) {
    return false;
  }

  const int16_t raw_ax   = makeInt16(buf[0],  buf[1]);
  const int16_t raw_ay   = makeInt16(buf[2],  buf[3]);
  const int16_t raw_az   = makeInt16(buf[4],  buf[5]);
  const int16_t raw_temp = makeInt16(buf[6],  buf[7]);
  const int16_t raw_gx   = makeInt16(buf[8],  buf[9]);
  const int16_t raw_gy   = makeInt16(buf[10], buf[11]);
  const int16_t raw_gz   = makeInt16(buf[12], buf[13]);

  mpu_ax = (raw_ax / MPU_ACCEL_LSB_PER_G) * GRAVITY_MS2;
  mpu_ay = (raw_ay / MPU_ACCEL_LSB_PER_G) * GRAVITY_MS2;
  mpu_az = (raw_az / MPU_ACCEL_LSB_PER_G) * GRAVITY_MS2;

  mpu_gx = (raw_gx / MPU_GYRO_LSB_PER_DEG_S) * DEG_TO_RAD_F;
  mpu_gy = (raw_gy / MPU_GYRO_LSB_PER_DEG_S) * DEG_TO_RAD_F;
  mpu_gz = (raw_gz / MPU_GYRO_LSB_PER_DEG_S) * DEG_TO_RAD_F;

  mpu_temp_c = (raw_temp / 340.0f) + 36.53f;

  return true;
}

// =========================
// Nivelado por LED
// =========================
void updateLevelLed() {
  const bool x_ok = absf(mpu_ax) <= LEVEL_TOL_MS2;
  const bool y_ok = absf(mpu_ay) <= LEVEL_TOL_MS2;

  if (x_ok && y_ok) {
    setLedGreen();
  } else if (x_ok || y_ok) {
    setLedOrange();
  } else {
    setLedRed();
  }
}

// =========================
// CSV
// =========================
void deleteOldLogs() {
  char filename[16];

  for (int i = 0; i < 1000; i++) {
    snprintf(filename, sizeof(filename), "/LLOG%03d.csv", i);
    if (SD.exists(filename)) {
      SD.remove(filename);
    }
  }
}

bool openLogFile(uint16_t index) {
  char filename[16];
  snprintf(filename, sizeof(filename), "/LLOG%03d.csv", index);

  logFile = SD.open(filename, FILE_WRITE);
  if (!logFile) return false;

  logFile.println("ts_ms,gx,gy,gz,ax,ay,az,tmp");
  logFile.flush();

  linesSinceFlush = 0;
  fileStartMs = millis();

  Serial.print("Escribiendo en: ");
  Serial.println(filename);

  return true;
}

bool rotateLogFileIfNeeded() {
  if ((uint32_t)(millis() - fileStartMs) < FILE_DURATION_MS) {
    return true;
  }

  if (logFile) {
    logFile.flush();
    logFile.close();
  }

  currentFileIndex++;
  return openLogFile(currentFileIndex);
}

bool writeCSVLine(uint64_t t_us) {
  if (!logFile) return false;

  const double t_ms = (double)t_us / 1000.0;

  logFile.print(t_ms, 3);
  logFile.print(',');

  logFile.print(mpu_gx, 9); logFile.print(',');
  logFile.print(mpu_gy, 9); logFile.print(',');
  logFile.print(mpu_gz, 9); logFile.print(',');
  logFile.print(mpu_ax, 9); logFile.print(',');
  logFile.print(mpu_ay, 9); logFile.print(',');
  logFile.print(mpu_az, 9); logFile.print(',');
  logFile.println(mpu_temp_c, 3);

  linesSinceFlush++;
  if (linesSinceFlush >= 100) {
    logFile.flush();
    linesSinceFlush = 0;
  }

  return (bool)logFile;
}

// =========================
// Setup
// =========================
void setup() {
  Serial.begin(115200);
  delay(1000);

  setLedOff();

  Serial.println();
  Serial.println("GY-521 + microSD en ESP32-S3");

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);

  if (!SD.begin(SD_CS, SPI)) {
    fatalError("ERROR: no se pudo inicializar la microSD");
  }
  Serial.println("microSD OK");

  deleteOldLogs();
  currentFileIndex = 0;

  if (!openLogFile(currentFileIndex)) {
    fatalError("ERROR: no se pudo abrir archivo CSV");
  }

  if (!initMPU6050()) {
    fatalError("ERROR: no se encontro o no respondio el MPU6050");
  }
  Serial.println("MPU6050 OK");

  Serial.print("Frecuencia de log: ");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.println(" Hz");

  Serial.print("Duracion por archivo: ");
  Serial.print(FILE_DURATION_MS);
  Serial.println(" ms");

  lastLogUs = micros();
  elapsedLogUs = 0;
}

// =========================
// Loop
// =========================
void loop() {
  const uint32_t nowUs = micros();

  if ((int32_t)(nowUs - lastLogUs) >= (int32_t)REPORT_INTERVAL_US) {
    lastLogUs += REPORT_INTERVAL_US;
    elapsedLogUs += REPORT_INTERVAL_US;

    if (!readMPU6050()) {
      setLedRed();
      return;
    }

    updateLevelLed();

    if (!rotateLogFileIfNeeded()) {
      fatalError("ERROR: no se pudo rotar el archivo CSV");
    }

    if (!writeCSVLine(elapsedLogUs)) {
      fatalError("ERROR: fallo escribiendo en el archivo CSV");
    }
  }
}