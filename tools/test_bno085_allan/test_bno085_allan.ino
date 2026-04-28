#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <math.h>

// =========================
// Pines
// =========================
#define I2C_SCL 8
#define I2C_SDA 9
#define BNO08X_RESET 14
#define RGB_LED_PIN 48

// =========================
// Frecuencia
// =========================
const uint16_t SAMPLE_RATE_HZ = 50;
const float DT_S = 1.0f / SAMPLE_RATE_HZ;
const uint32_t REPORT_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;
const uint32_t PRINT_INTERVAL_MS  = 1000UL / SAMPLE_RATE_HZ;

// =========================
// Complementary filter
// =========================
const float COMP_ALPHA = 0.98f;   // típico 0.95-0.99

// =========================
// BNO085
// =========================
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// =========================
// Datos recibidos
// =========================
// Gyro uncalibrated [rad/s]
float gx = 0.0f, gy = 0.0f, gz = 0.0f;
float gbx = 0.0f, gby = 0.0f, gbz = 0.0f;   // bias reportado por el sensor

// Accel [m/s^2]
float ax = 0.0f, ay = 0.0f, az = 0.0f;

bool gyroValid = false;
bool accelValid = false;

// =========================
// Filtro por eje (tu corrección)
// =========================
struct AxisFilter {
  float sigma_n;
  float sigma_wb;
  float bias_init_std;
  float bias;
  float y;
  bool initialized;
};

// Gyro
AxisFilter f_gx = {
  0.0020695358490417435f,
  0,
  1.923000293256989e-05f,
  0.0f, 0.0f, false
};

AxisFilter f_gy = {
  0.001502461988929328f,
  0,
  2.2436317767359602e-05f,
  0.0f, 0.0f, false
};

AxisFilter f_gz = {
  0.0018050063307510388,
  0,
  3.0230176900075297e-05f,
  0.0f, 0.0f, false
};

// Accel
AxisFilter f_ax = {
  0.010158884790128435f,
  3.307764346296583e-06f,
  0.00026926817876089343f,
  0.0f, 0.0f, false
};

AxisFilter f_ay = {
  0.019644998888639006f,
  4.794056939632767e-05f,
  0.0011784629098705644f,
  0.0f, 0.0f, false
};

AxisFilter f_az = {
  0.01569906539869613,
  8.2550223772409582e-05f,
  0.000563665407533425,
  0.0f, 0.0f, false
};

// =========================
// Tuning de tu filtro Allan-based
// =========================
const float gyroBiasTau_s  = 120.0f;
const float accelBiasTau_s = 20.0f;

// =========================
// Estados de comparación
// =========================
// Ángulos por acelerómetro
float roll_acc = 0.0f;
float pitch_acc = 0.0f;

// Complementary filter "antes" (gyro uncalibrated)
float roll_cf_raw = 0.0f;
float pitch_cf_raw = 0.0f;

// Complementary filter "después" (gyro corregido por ti)
float roll_cf_corr = 0.0f;
float pitch_cf_corr = 0.0f;

bool attitudeInitialized = false;

// =========================
// LED helpers
// =========================
void setLedOff()   { neopixelWrite(RGB_LED_PIN, 0, 0, 0); }
void setLedRed()   { neopixelWrite(RGB_LED_PIN, 32, 0, 0); }
void setLedGreen() { neopixelWrite(RGB_LED_PIN, 0, 32, 0); }

// =========================
// Helpers
// =========================
void fatalError(const char* msg) {
  Serial.println(msg);
  setLedRed();
  while (1) {
    delay(100);
  }
}

void setReports() {
  Serial.println("Configurando reportes UNCALIBRATED...");

  if (!bno08x.enableReport(SH2_GYROSCOPE_UNCALIBRATED, REPORT_INTERVAL_US)) {
    fatalError("ERROR: no se pudo habilitar SH2_GYROSCOPE_UNCALIBRATED");
  }

  if (!bno08x.enableReport(SH2_ACCELEROMETER, REPORT_INTERVAL_US)) {
    fatalError("ERROR: no se pudo habilitar SH2_ACCELEROMETER");
  }
}

void pollIMU() {
  while (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {

      case SH2_GYROSCOPE_UNCALIBRATED:
        gx  = sensorValue.un.gyroscopeUncal.x;
        gy  = sensorValue.un.gyroscopeUncal.y;
        gz  = sensorValue.un.gyroscopeUncal.z;
        gbx = sensorValue.un.gyroscopeUncal.biasX;
        gby = sensorValue.un.gyroscopeUncal.biasY;
        gbz = sensorValue.un.gyroscopeUncal.biasZ;
        gyroValid = true;
        break;

      case SH2_ACCELEROMETER:
        ax = sensorValue.un.accelerometer.x;
        ay = sensorValue.un.accelerometer.y;
        az = sensorValue.un.accelerometer.z;
        accelValid = true;
        break;

      default:
        break;
    }
  }
}

float computeAlphaBias(float dt, float tau_s) {
  if (tau_s <= 0.0f) return 1.0f;
  return dt / (tau_s + dt);
}

float computeAlphaSmoothFromNoise(float sigma_n) {
  float alpha = 0.15f;

  if (sigma_n > 0.01f) {
    alpha = 0.08f;
  } else if (sigma_n > 0.001f) {
    alpha = 0.12f;
  } else {
    alpha = 0.20f;
  }

  return alpha;
}

float filterAxis(AxisFilter &f, float z, float dt, float biasTau_s) {
  if (!f.initialized) {
    f.bias = z;
    f.y = 0.0f;
    f.initialized = true;
    return f.y;
  }

  float alpha_bias = computeAlphaBias(dt, biasTau_s);
  f.bias += alpha_bias * (z - f.bias);

  float debiased = z - f.bias;

  float alpha_smooth = computeAlphaSmoothFromNoise(f.sigma_n);
  f.y += alpha_smooth * (debiased - f.y);

  return f.y;
}

void computeAccelAngles(float ax_mps2, float ay_mps2, float az_mps2,
                        float &roll_rad, float &pitch_rad) {
  // Convención típica:
  // roll  alrededor de X
  // pitch alrededor de Y
  roll_rad  = atan2f(ay_mps2, az_mps2);
  pitch_rad = atan2f(-ax_mps2, sqrtf(ay_mps2 * ay_mps2 + az_mps2 * az_mps2));
}

float rad2deg(float x) {
  return x * 180.0f / PI;
}

// =========================
// Setup
// =========================
void setup() {
  Serial.begin(115200);
  delay(1000);

  setLedOff();

  Serial.println();
  Serial.println("BNO085 complementary filter drift comparison");

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  if (!bno08x.begin_I2C(BNO08x_I2CADDR_DEFAULT, &Wire)) {
    fatalError("ERROR: no se encontró el BNO085");
  }

  Serial.println("BNO085 OK");
  setReports();
  setLedGreen();

  Serial.println("Configuracion OK");
  Serial.println("Formato salida:");
  Serial.println("roll_acc,roll_rawCF,roll_corrCF,pitch_acc,pitch_rawCF,pitch_corrCF,gx,gx_corr,gy,gy_corr");
}

// =========================
// Loop
// =========================
void loop() {
  if (bno08x.wasReset()) {
    Serial.println("El BNO085 se ha reiniciado, reconfigurando reportes...");
    setReports();
    gyroValid = false;
    accelValid = false;
    attitudeInitialized = false;
    setLedGreen();
  }

  pollIMU();

  static uint32_t lastPrintMs = 0;
  uint32_t nowMs = millis();

  if ((uint32_t)(nowMs - lastPrintMs) >= PRINT_INTERVAL_MS) {
    lastPrintMs += PRINT_INTERVAL_MS;

    if (gyroValid && accelValid) {
      // 1) Filtrado/corrección "tuya"
      float gx_corr = filterAxis(f_gx, gx, DT_S, gyroBiasTau_s);
      float gy_corr = filterAxis(f_gy, gy, DT_S, gyroBiasTau_s);
      float gz_corr = filterAxis(f_gz, gz, DT_S, gyroBiasTau_s);

      float ax_corr = filterAxis(f_ax, ax, DT_S, accelBiasTau_s);
      float ay_corr = filterAxis(f_ay, ay, DT_S, accelBiasTau_s);
      float az_corr = filterAxis(f_az, az, DT_S, accelBiasTau_s);

      // 2) Ángulos del acelerómetro
      computeAccelAngles(ax, ay, az, roll_acc, pitch_acc);

      // Si quieres comparar con accel también filtrado, cambia a:
      // computeAccelAngles(ax_corr, ay_corr, az_corr, roll_acc, pitch_acc);

      if (!attitudeInitialized) {
        roll_cf_raw = roll_acc;
        pitch_cf_raw = pitch_acc;

        roll_cf_corr = roll_acc;
        pitch_cf_corr = pitch_acc;

        attitudeInitialized = true;
      }

      // 3) Complementary filter ANTES
      roll_cf_raw  = COMP_ALPHA * (roll_cf_raw  + gx * DT_S) + (1.0f - COMP_ALPHA) * roll_acc;
      pitch_cf_raw = COMP_ALPHA * (pitch_cf_raw + gy * DT_S) + (1.0f - COMP_ALPHA) * pitch_acc;

      // 4) Complementary filter DESPUÉS
      roll_cf_corr  = COMP_ALPHA * (roll_cf_corr  + gx_corr * DT_S) + (1.0f - COMP_ALPHA) * roll_acc;
      pitch_cf_corr = COMP_ALPHA * (pitch_cf_corr + gy_corr * DT_S) + (1.0f - COMP_ALPHA) * pitch_acc;

      // 5) Imprimir para TelPlotter
      Serial.print(">roll_acc:"); Serial.print(rad2deg(roll_acc), 6);      // 0
      Serial.print(">roll_cf_raw:"); Serial.print(rad2deg(roll_cf_raw), 6);   // 1
      Serial.print(">roll_cf_corr:"); Serial.print(rad2deg(roll_cf_corr), 6);  // 2
      Serial.print(">pitch_acc:"); Serial.print(rad2deg(pitch_acc), 6);     // 3
      Serial.print(">pitch_cf_raw:"); Serial.print(rad2deg(pitch_cf_raw), 6);  // 4
      Serial.print(">pitch_cf_corr:"); Serial.print(rad2deg(pitch_cf_corr), 6); // 5
      Serial.print(">gx:"); Serial.print(gx, 6);                     // 6
      Serial.print(">gx_corr:"); Serial.print(gx_corr, 6);                // 7
      Serial.print(">gy:"); Serial.print(gy, 6);                     // 8
      Serial.print(">gy_corr:"); Serial.println(gy_corr, 6);
    }
  }
}