#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <memory>

#include "config.hpp"
namespace vio {

using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec9 = Eigen::Matrix<double, 9, 1>;
using Vec15 = Eigen::Matrix<double, 15, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat9 = Eigen::Matrix<double, 9, 9>;
using Mat15 = Eigen::Matrix<double, 15, 15>;

inline Mat3 Skew(const Vec3& v) {
  Mat3 S;
  S << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
      -v.y(), v.x(), 0.0;
  return S;
}

inline Eigen::Quaterniond NormalizeQ(const Eigen::Quaterniond& q) {
  Eigen::Quaterniond qn = q;
  qn.normalize();
  if (qn.w() < 0.0) {
    qn.coeffs() *= -1.0;
  }
  return qn;
}

inline Mat3 ExpSO3(const Vec3& w) {
  const double theta = w.norm();
  const Mat3 I = Mat3::Identity();
  if (theta < 1e-10) {
    return I + Skew(w);
  }
  const Vec3 a = w / theta;
  const Mat3 A = Skew(a);
  return I + std::sin(theta) * A + (1.0 - std::cos(theta)) * (A * A);
}

inline Vec3 LogSO3(const Mat3& R) {
  const double cos_theta = std::max(-1.0, std::min(1.0, 0.5 * (R.trace() - 1.0)));
  const double theta = std::acos(cos_theta);
  if (theta < 1e-10) {
    return Vec3::Zero();
  }
  Vec3 vee;
  vee << R(2, 1) - R(1, 2),
         R(0, 2) - R(2, 0),
         R(1, 0) - R(0, 1);
  return 0.5 * theta / std::sin(theta) * vee;
}

inline Eigen::Quaterniond ExpQuat(const Vec3& w) {
  return NormalizeQ(Eigen::Quaterniond(ExpSO3(w)));
}

inline Vec3 SafeNormalized(const Vec3& v, const Vec3& fallback = Vec3::Zero()) {
  const double n = v.norm();
  if (n <= 1e-12 || !std::isfinite(n)) {
    return fallback;
  }
  return v / n;
}

inline Mat3 RightJacobianSO3(const Vec3& w) {
  const double theta = w.norm();
  const Mat3 I = Mat3::Identity();
  if (theta < 1e-8) {
    return I - 0.5 * Skew(w) + (1.0 / 6.0) * Skew(w) * Skew(w);
  }
  const Mat3 W = Skew(w);
  const double theta2 = theta * theta;
  const double theta3 = theta2 * theta;
  return I - (1.0 - std::cos(theta)) / theta2 * W
           + (theta - std::sin(theta)) / theta3 * (W * W);
}

struct ImuNoiseParams {
  Mat3 sigma_g = Mat3::Identity();
  Mat3 sigma_a = Mat3::Identity();
  Mat3 sigma_wg = Mat3::Identity();
  Mat3 sigma_wa = Mat3::Identity();

  static ImuNoiseParams FromAllan(const Vec3& N_g,
                                  const Vec3& N_a,
                                  const Vec3& K_g,
                                  const Vec3& K_a) {
    ImuNoiseParams p;
    p.sigma_g = N_g.array().square().matrix().asDiagonal();
    p.sigma_a = N_a.array().square().matrix().asDiagonal();
    p.sigma_wg = K_g.array().square().matrix().asDiagonal();
    p.sigma_wa = K_a.array().square().matrix().asDiagonal();
    return p;
  }
};

struct ImuSample {
  double t = 0.0;
  Vec3 gyro = Vec3::Zero();
  Vec3 accel = Vec3::Zero();
};

struct NavState {
  double t = 0.0;
  Vec3 p = Vec3::Zero();
  Vec3 v = Vec3::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Vec3 bg = Vec3::Zero();
  Vec3 ba = Vec3::Zero();

  Mat3 Rwb() const { return q.toRotationMatrix(); }
};

struct ImuFactorResidual {
  Vec9 r = Vec9::Zero();
  Mat9 cov = Mat9::Identity();
};

struct BiasRandomWalkResidual {
  Vec6 r = Vec6::Zero();
  Mat6 cov = Mat6::Identity();
};

struct ImuIntervalSummary {
  bool valid = false;
  double delta_t = 0.0;
  Vec3 dtheta = Vec3::Zero();
  Vec3 dv = Vec3::Zero();
  Vec3 dp = Vec3::Zero();
  Mat9 cov = Mat9::Zero();
};

struct ImuEskfConfig {
  Vec3 gravity_world  = Vec3(0.0, 0.0, 0.0);
  bool enable_zupt = true;
  bool enable_gravity_update = true;
  double stationary_gyro_threshold = 0.03;
  double stationary_accel_abs_threshold = 0.20;
  double zupt_velocity_sigma = 0.05;
  double gravity_direction_sigma = 0.05;
};

struct ImuEskfOutput {
  bool initialized = false;
  bool stationary = false;
  double dt = 0.0;
  NavState state;
  Mat15 covariance = Mat15::Zero();
  Vec3 accel_world = Vec3::Zero();
  ImuIntervalSummary current_interval;
};

inline Mat15 DefaultInitialCovariance(double pos_sigma = 0.10,
                                      double vel_sigma = 0.10,
                                      double rot_sigma_rad = 5.0 * 3.14159265358979323846 / 180.0,
                                      double bg_sigma = 1e-3,
                                      double ba_sigma = 1e-2) {
  Mat15 P = Mat15::Zero();
  P.block<3, 3>(0, 0) = Mat3::Identity() * (pos_sigma * pos_sigma);
  P.block<3, 3>(3, 3) = Mat3::Identity() * (vel_sigma * vel_sigma);
  P.block<3, 3>(6, 6) = Mat3::Identity() * (rot_sigma_rad * rot_sigma_rad);
  P.block<3, 3>(9, 9) = Mat3::Identity() * (bg_sigma * bg_sigma);
  P.block<3, 3>(12, 12) = Mat3::Identity() * (ba_sigma * ba_sigma);
  return P;
}

class ImuPropagator {
 public:
  explicit ImuPropagator(ImuNoiseParams noise, Vec3 gravity_world)
      : noise_(std::move(noise)), g_(gravity_world) {}

  void Propagate(const ImuSample& s0, const ImuSample& s1, NavState& x, Mat15& P) const {
    const double dt = s1.t - s0.t;
    if (dt < 0.0) {
      Logger(ERROR, "[PROPAGATE] DT0: %f, DT1: %f", s0.t, s1.t);
      throw std::runtime_error("IMU timestamps must be strictly increasing.");
    }

    const Vec3 omega = 0.5 * (s0.gyro + s1.gyro) - x.bg;
    const Vec3 acc0 = s0.accel - x.ba;
    const Vec3 acc1 = s1.accel - x.ba;

    const Mat3 Rk = x.Rwb();
    const Eigen::Quaterniond dq = ExpQuat(omega * dt);
    const Eigen::Quaterniond q_next = NormalizeQ(x.q * dq);
    const Mat3 Rk1 = q_next.toRotationMatrix();

    const Vec3 a_world = 0.5 * (Rk * acc0 + Rk1 * acc1) + g_;

    x.p = x.p + x.v * dt + 0.5 * a_world * dt * dt;
    x.v = x.v + a_world * dt;
    x.q = q_next;
    x.t = s1.t;

    const Vec3 acc_mid = 0.5 * (acc0 + acc1);
    const Mat3 Rmid = Rk;
    const Mat3 W = Skew(omega);
    const Mat3 A = Skew(acc_mid);

    Mat15 F = Mat15::Zero();
    F.block<3, 3>(0, 3) = Mat3::Identity();
    F.block<3, 3>(3, 6) = -Rmid * A;
    F.block<3, 3>(3, 12) = -Rmid;
    F.block<3, 3>(6, 6) = -W;
    F.block<3, 3>(6, 9) = -Mat3::Identity();

    Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
    G.block<3, 3>(3, 3) = -Rmid;
    G.block<3, 3>(6, 0) = -Mat3::Identity();
    G.block<3, 3>(9, 6) = Mat3::Identity();
    G.block<3, 3>(12, 9) = Mat3::Identity();

    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    Qc.block<3, 3>(0, 0) = noise_.sigma_g;
    Qc.block<3, 3>(3, 3) = noise_.sigma_a;
    Qc.block<3, 3>(6, 6) = noise_.sigma_wg;
    Qc.block<3, 3>(9, 9) = noise_.sigma_wa;

    const Mat15 Phi = Mat15::Identity() + F * dt;
    const Mat15 Qd = G * Qc * G.transpose() * dt;
    P = Phi * P * Phi.transpose() + Qd;
  }

  static void InjectError(const Vec15& dx, NavState& x) {
    x.p += dx.segment<3>(0);
    x.v += dx.segment<3>(3);
    x.q = NormalizeQ(ExpQuat(dx.segment<3>(6)) * x.q);
    x.bg += dx.segment<3>(9);
    x.ba += dx.segment<3>(12);
  }

 private:
  ImuNoiseParams noise_;
  Vec3 g_;
};

class ImuPreintegrator {
 public:
  explicit ImuPreintegrator(ImuNoiseParams noise)
      : noise_(std::move(noise)) {
    Reset(Vec3::Zero(), Vec3::Zero());
  }

  void Reset(const Vec3& bg_lin, const Vec3& ba_lin) {
    bg_lin_ = bg_lin;
    ba_lin_ = ba_lin;
    delta_t_ = 0.0;
    dR_ = Mat3::Identity();
    dv_ = Vec3::Zero();
    dp_ = Vec3::Zero();
    cov_ = Mat9::Zero();
    J_r_bg_ = Mat3::Zero();
    J_v_bg_ = Mat3::Zero();
    J_v_ba_ = Mat3::Zero();
    J_p_bg_ = Mat3::Zero();
    J_p_ba_ = Mat3::Zero();
  }

  void Integrate(const ImuSample& s0, const ImuSample& s1) {
    const double dt = s1.t - s0.t;
    if (dt < 0.0) {
      Logger(ERROR, "[INTEGRATE] DT0: %f, DT1: %f", s0.t, s1.t);
      throw std::runtime_error("IMU timestamps must be strictly increasing.");
    }

    const Vec3 omega = 0.5 * (s0.gyro + s1.gyro) - bg_lin_;
    const Vec3 acc = 0.5 * (s0.accel + s1.accel) - ba_lin_;

    const Mat3 dR_prev = dR_;
    const Vec3 dv_prev = dv_;

    const Vec3 phi = omega * dt;
    const Mat3 dR_inc = ExpSO3(phi);
    const Mat3 Jr = RightJacobianSO3(phi);

    dR_ = dR_ * dR_inc;
    dv_ = dv_ + dR_prev * acc * dt;
    dp_ = dp_ + dv_prev * dt + 0.5 * dR_prev * acc * dt * dt;
    delta_t_ += dt;

    const Vec3 acc_mid = 0.5 * (s0.accel + s1.accel) - ba_lin_;
    const Mat3 A_acc = Skew(acc_mid);
    const Mat3 J_r_bg_old = J_r_bg_;
    const Mat3 J_v_bg_old = J_v_bg_;
    const Mat3 J_v_ba_old = J_v_ba_;
    const Mat3 J_p_bg_old = J_p_bg_;
    const Mat3 J_p_ba_old = J_p_ba_;

    J_r_bg_ = dR_inc.transpose() * J_r_bg_old - Jr * dt;
    J_v_bg_ = J_v_bg_old - dR_prev * A_acc * J_r_bg_old * dt;
    J_v_ba_ = J_v_ba_old - dR_prev * dt;
    J_p_bg_ = J_p_bg_old + J_v_bg_old * dt - 0.5 * dR_prev * A_acc * J_r_bg_old * dt * dt;
    J_p_ba_ = J_p_ba_old + J_v_ba_old * dt - 0.5 * dR_prev * dt * dt;

    Mat9 A = Mat9::Identity();
    A.block<3, 3>(0, 0) = dR_inc.transpose();
    A.block<3, 3>(3, 0) = -dR_prev * A_acc * dt;
    A.block<3, 3>(6, 0) = -0.5 * dR_prev * A_acc * dt * dt;
    A.block<3, 3>(6, 3) = Mat3::Identity() * dt;

    Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
    B.block<3, 3>(0, 0) = Jr * dt;
    B.block<3, 3>(3, 3) = dR_prev * dt;
    B.block<3, 3>(6, 3) = 0.5 * dR_prev * dt * dt;

    Eigen::Matrix<double, 6, 6> Qm = Eigen::Matrix<double, 6, 6>::Zero();
    Qm.block<3, 3>(0, 0) = noise_.sigma_g;
    Qm.block<3, 3>(3, 3) = noise_.sigma_a;

    cov_ = A * cov_ * A.transpose() + B * Qm * B.transpose();
  }

  double delta_t() const { return delta_t_; }
  const Mat3& delta_R() const { return dR_; }
  const Vec3& delta_v() const { return dv_; }
  const Vec3& delta_p() const { return dp_; }
  const Mat9& covariance() const { return cov_; }

  Mat3 CorrectedDeltaR(const Vec3& bg) const {
    const Vec3 dbg = bg - bg_lin_;
    return dR_ * ExpSO3(J_r_bg_ * dbg);
  }

  Vec3 CorrectedDeltaV(const Vec3& bg, const Vec3& ba) const {
    const Vec3 dbg = bg - bg_lin_;
    const Vec3 dba = ba - ba_lin_;
    return dv_ + J_v_bg_ * dbg + J_v_ba_ * dba;
  }

  Vec3 CorrectedDeltaP(const Vec3& bg, const Vec3& ba) const {
    const Vec3 dbg = bg - bg_lin_;
    const Vec3 dba = ba - ba_lin_;
    return dp_ + J_p_bg_ * dbg + J_p_ba_ * dba;
  }

  ImuFactorResidual EvaluateResidual(const NavState& xi,
                                     const NavState& xj,
                                     const Vec3& gravity_world) const {
    ImuFactorResidual out;

    const Mat3 Ri = xi.Rwb();
    const Mat3 Rj = xj.Rwb();

    const Mat3 dR_corr = CorrectedDeltaR(xi.bg);
    const Vec3 dv_corr = CorrectedDeltaV(xi.bg, xi.ba);
    const Vec3 dp_corr = CorrectedDeltaP(xi.bg, xi.ba);

    const Vec3 r_p = Ri.transpose() * (xj.p - xi.p - xi.v * delta_t_ - 0.5 * gravity_world * delta_t_ * delta_t_) - dp_corr;
    const Vec3 r_v = Ri.transpose() * (xj.v - xi.v - gravity_world * delta_t_) - dv_corr;
    const Vec3 r_r = LogSO3(dR_corr.transpose() * Ri.transpose() * Rj);

    out.r.segment<3>(0) = r_r;
    out.r.segment<3>(3) = r_v;
    out.r.segment<3>(6) = r_p;
    out.cov = cov_;
    return out;
  }

  BiasRandomWalkResidual EvaluateBiasResidual(const NavState& xi,
                                              const NavState& xj) const {
    BiasRandomWalkResidual out;
    out.r.segment<3>(0) = xj.bg - xi.bg;
    out.r.segment<3>(3) = xj.ba - xi.ba;
    out.cov.setZero();
    out.cov.block<3, 3>(0, 0) = noise_.sigma_wg * delta_t_;
    out.cov.block<3, 3>(3, 3) = noise_.sigma_wa * delta_t_;
    return out;
  }

  const Mat3& J_r_bg() const { return J_r_bg_; }
  const Mat3& J_v_bg() const { return J_v_bg_; }
  const Mat3& J_v_ba() const { return J_v_ba_; }
  const Mat3& J_p_bg() const { return J_p_bg_; }
  const Mat3& J_p_ba() const { return J_p_ba_; }
  const Vec3& bias_lin_g() const { return bg_lin_; }
  const Vec3& bias_lin_a() const { return ba_lin_; }

 private:
  ImuNoiseParams noise_;
  Vec3 bg_lin_ = Vec3::Zero();
  Vec3 ba_lin_ = Vec3::Zero();
  double delta_t_ = 0.0;
  Mat3 dR_ = Mat3::Identity();
  Vec3 dv_ = Vec3::Zero();
  Vec3 dp_ = Vec3::Zero();
  Mat9 cov_ = Mat9::Zero();
  Mat3 J_r_bg_ = Mat3::Zero();
  Mat3 J_v_bg_ = Mat3::Zero();
  Mat3 J_v_ba_ = Mat3::Zero();
  Mat3 J_p_bg_ = Mat3::Zero();
  Mat3 J_p_ba_ = Mat3::Zero();
};

class ImuEskf15 {
 public:
  explicit ImuEskf15(const ImuNoiseParams& noise, const ImuEskfConfig& cfg = ImuEskfConfig(), const Mat15& P0 = DefaultInitialCovariance())
      : noise_(noise), cfg_(cfg), propagator_(noise_, cfg_.gravity_world), preintegrator_(noise_) {
    Reset(NavState{}, P0);
  }

  void Reset(const NavState& x0, const Mat15& P0) {
    state_ = x0;
    P_ = P0;
    initialized_ = false;
    have_prev_sample_ = false;
    last_completed_interval_ = ImuIntervalSummary{};
    preintegrator_.Reset(state_.bg, state_.ba);
  }

  void InitializeFromAccel(const ImuSample& s, const Vec3& p0 = Vec3::Zero(), const Vec3& v0 = Vec3::Zero()) {
    if (s.accel.norm() < 1e-9) {
      throw std::runtime_error("Cannot initialize IMU attitude from zero acceleration vector.");
    }
    if (cfg_.gravity_world.norm() < 1e-9) {
      throw std::runtime_error("Gravity vector must be non-zero.");
    }

    state_.t = s.t;
    state_.p = p0;
    state_.v = v0;
    state_.q = NormalizeQ(Eigen::Quaterniond::FromTwoVectors(
        SafeNormalized(s.accel),
        -GravityDirectionWorld()));
    state_.bg.setZero();
    state_.ba.setZero();

    initialized_ = true;
    have_prev_sample_ = true;
    prev_sample_ = s;
    preintegrator_.Reset(state_.bg, state_.ba);
  }

  ImuEskfOutput Update(const ImuSample& s) {
    if (!initialized_) {
      InitializeFromAccel(s);
      return BuildOutput(s, 0.0, IsStationarySample(s));
    }

    if (!have_prev_sample_) {
      prev_sample_ = s;
      have_prev_sample_ = true;
      state_.t = s.t;
      return BuildOutput(s, 0.0, IsStationarySample(s));
    }

    const double dt = s.t - prev_sample_.t;
    if (dt < 0.0) {
      Logger(ERROR, "[UPDATE] DT0: %f, DT1: %f", prev_sample_.t, s.t);
      throw std::runtime_error("IMU timestamps must be strictly increasing.");
    }

    propagator_.Propagate(prev_sample_, s, state_, P_);
    preintegrator_.Integrate(prev_sample_, s);

    const bool stationary = IsStationarySample(s);
    if (stationary) {
      if (cfg_.enable_zupt) {
        ApplyZeroVelocityUpdate(cfg_.zupt_velocity_sigma);
      }
      if (cfg_.enable_gravity_update) {
        ApplyGravityDirectionUpdate(s.accel, cfg_.gravity_direction_sigma);
      }
    }

    prev_sample_ = s;
    return BuildOutput(s, dt, stationary);
  }

  void NotifyKeyframe() {
    last_completed_interval_ = CurrentIntervalSummary();
    preintegrator_.Reset(state_.bg, state_.ba);
  }

  bool IsInitialized() const { return initialized_; }
  const NavState& state() const { return state_; }
  const Mat15& covariance() const { return P_; }
  const ImuPreintegrator& preintegrator() const { return preintegrator_; }
  const ImuIntervalSummary& last_completed_interval() const { return last_completed_interval_; }

 private:
  bool IsStationarySample(const ImuSample& s) const {
    const double gyro_norm = s.gyro.norm();
    const double acc_norm = s.accel.norm();
    const double g_norm = std::abs(cfg_.gravity_world.norm());
    return gyro_norm < cfg_.stationary_gyro_threshold &&
           std::abs(acc_norm - g_norm) < cfg_.stationary_accel_abs_threshold;
  }

  void ApplyZeroVelocityUpdate(double sigma) {
    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block<3, 3>(0, 3) = Mat3::Identity();

    const Mat3 R = Mat3::Identity() * (sigma * sigma);
    const Vec3 r = -state_.v;
    const Mat3 S = H * P_ * H.transpose() + R;
    const Eigen::Matrix<double, 15, 3> K = P_ * H.transpose() * S.inverse();
    const Vec15 dx = K * r;

    ImuPropagator::InjectError(dx, state_);

    const Mat15 I = Mat15::Identity();
    P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R * K.transpose();
  }

  void ApplyGravityDirectionUpdate(const Vec3& acc_body, double sigma) {
    if (acc_body.norm() < 1e-9) {
      return;
    }

    const Vec3 z_meas = SafeNormalized(acc_body);
    const Vec3 z_hat = state_.Rwb().transpose() * (-GravityDirectionWorld());

    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block<3, 3>(0, 6) = -Skew(z_hat);

    const Mat3 R = Mat3::Identity() * (sigma * sigma);
    const Vec3 r = z_meas - z_hat;
    const Mat3 S = H * P_ * H.transpose() + R;
    const Eigen::Matrix<double, 15, 3> K = P_ * H.transpose() * S.inverse();
    const Vec15 dx = K * r;

    ImuPropagator::InjectError(dx, state_);

    const Mat15 I = Mat15::Identity();
    P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R * K.transpose();
  }

  ImuIntervalSummary CurrentIntervalSummary() const {
    ImuIntervalSummary out;
    out.valid = preintegrator_.delta_t() > 0.0;
    out.delta_t = preintegrator_.delta_t();
    out.dtheta = LogSO3(preintegrator_.CorrectedDeltaR(state_.bg));
    out.dv = preintegrator_.CorrectedDeltaV(state_.bg, state_.ba);
    out.dp = preintegrator_.CorrectedDeltaP(state_.bg, state_.ba);
    out.cov = preintegrator_.covariance();
    return out;
  }

  Vec3 GravityDirectionWorld() const {
    const double g_norm = cfg_.gravity_world.norm();
    if (g_norm < 1e-9) {
      throw std::runtime_error("Gravity vector must be non-zero.");
    }
    return cfg_.gravity_world / g_norm;
  }

  ImuEskfOutput BuildOutput(const ImuSample& s, double dt, bool stationary) const {
    ImuEskfOutput out;
    out.initialized = initialized_;
    out.stationary = stationary;
    out.dt = dt;
    out.state = state_;
    out.covariance = P_;
    out.accel_world = state_.Rwb() * (s.accel - state_.ba) + cfg_.gravity_world;
    out.current_interval = CurrentIntervalSummary();
    return out;
  }

  ImuNoiseParams noise_;
  ImuEskfConfig cfg_;
  NavState state_;
  Mat15 P_ = Mat15::Zero();
  ImuPropagator propagator_;
  ImuPreintegrator preintegrator_;
  bool initialized_ = false;
  bool have_prev_sample_ = false;
  ImuSample prev_sample_;
  ImuIntervalSummary last_completed_interval_;
};

}  // namespace vio





struct ImuFrontendOutput {
    bool valid = false;
    bool initialized = false;
    bool stationary = false;

    double timestamp_sec = 0.0;
    double dt = 0.0;

    cv::Vec3d gyro_raw = {0.0, 0.0, 0.0};
    cv::Vec3d accel_raw = {0.0, 0.0, 0.0};
    cv::Vec3d gyro_cal = {0.0, 0.0, 0.0};
    cv::Vec3d accel_cal = {0.0, 0.0, 0.0};

    cv::Vec3d gyro_bias_total = {0.0, 0.0, 0.0};
    cv::Vec3d accel_bias_total = {0.0, 0.0, 0.0};
    cv::Vec3d gyro_bias_dynamic = {0.0, 0.0, 0.0};
    cv::Vec3d accel_bias_dynamic = {0.0, 0.0, 0.0};

    cv::Vec3d accel_world = {0.0, 0.0, 0.0};
    cv::Vec3d vel_xyz = {0.0, 0.0, 0.0};
    cv::Vec3d pos_xyz = {0.0, 0.0, 0.0};
    cv::Vec3d rpy_rad = {0.0, 0.0, 0.0};
    cv::Vec3d rpy_deg = {0.0, 0.0, 0.0};
    cv::Vec4d quat_wxyz = {1.0, 0.0, 0.0, 0.0};

    double preint_dt = 0.0;
    cv::Vec3d preint_dtheta = {0.0, 0.0, 0.0};
    cv::Vec3d preint_dv = {0.0, 0.0, 0.0};
    cv::Vec3d preint_dp = {0.0, 0.0, 0.0};
};

class ImuFrontend {
public:
    ImuFrontend() = default;

    bool init(const Config& config) {
        fixed_bg_ = cvToEig(config.imu.gyro_bias);
        fixed_ba_ = cvToEig(config.imu.accel_bias);
        gyro_sm_ = cvMatxToEig(config.imu.gyro_scale_misalignment);
        accel_sm_ = cvMatxToEig(config.imu.accel_scale_misalignment);
        gravity_world_ = cvToEig(config.imu.gv);
        if (gravity_world_.norm() < 1e-6) {
            throw std::runtime_error("El vector imu.gv debe ser no nulo.");
        }

        vio::Vec3 N_g = cvToEig(config.imu.gyroAllanN());
        vio::Vec3 N_a = cvToEig(config.imu.accelAllanN());
        vio::Vec3 K_g = cvToEig(config.imu.gyroAllanK());
        vio::Vec3 K_a = cvToEig(config.imu.accelAllanK());

        const bool have_allan = config.imu.hasValidAllan();
        vio::ImuNoiseParams noise;
        if (config.gen.use_allan && have_allan) {
            noise = vio::ImuNoiseParams::FromAllan(N_g, N_a, K_g, K_a);
        } else {
            const vio::Vec3 default_Ng(1.0e-4, 1.0e-4, 1.0e-4);
            const vio::Vec3 default_Na(3.0e-3, 3.0e-3, 3.0e-3);
            const vio::Vec3 default_Kg(1.0e-6, 1.0e-6, 1.0e-6);
            const vio::Vec3 default_Ka(2.0e-5, 2.0e-5, 2.0e-5);
            noise = vio::ImuNoiseParams::FromAllan(default_Ng, default_Na, default_Kg, default_Ka);
        }

        vio::ImuEskfConfig eskf_cfg;
        eskf_cfg.gravity_world = gravity_world_;
        eskf_cfg.enable_zupt = true;
        eskf_cfg.enable_gravity_update = true;
        eskf_cfg.stationary_gyro_threshold = 0.03;
        eskf_cfg.stationary_accel_abs_threshold = 0.20;
        eskf_cfg.zupt_velocity_sigma = 0.05;
        eskf_cfg.gravity_direction_sigma = 0.05;

        const vio::Mat15 P0 = vio::DefaultInitialCovariance(0.10, 0.10, 5.0 * 3.14159265358979323846 / 180.0,
                                                            1.0e-3, 1.0e-2);
        eskf_.reset(new vio::ImuEskf15(noise, eskf_cfg, P0));
        last_output_ = ImuFrontendOutput{};
        return true;
    }

    ImuFrontendOutput update(const imuData& raw) {
        if (!eskf_) {
            throw std::runtime_error("ImuFrontend no inicializado.");
        }

        const vio::ImuSample sample = calibrate(raw);
        const vio::ImuEskfOutput out = eskf_->Update(sample);
        last_output_ = buildOutput(raw, sample, out);
        return last_output_;
    }

    void notifyFrameBoundary() {
        if (eskf_ && eskf_->IsInitialized()) {
            eskf_->NotifyKeyframe();
        }
    }

    const ImuFrontendOutput& lastOutput() const {
        return last_output_;
    }

private:
    static vio::Vec3 cvToEig(const cv::Vec3d& v) {
        return vio::Vec3(v[0], v[1], v[2]);
    }

    static cv::Vec3d eigToCv(const vio::Vec3& v) {
        return cv::Vec3d(v.x(), v.y(), v.z());
    }

    static vio::Mat3 cvMatxToEig(const cv::Matx33d& M) {
        vio::Mat3 out;
        out << M(0, 0), M(0, 1), M(0, 2),
               M(1, 0), M(1, 1), M(1, 2),
               M(2, 0), M(2, 1), M(2, 2);
        return out;
    }

    static cv::Vec4d quatToCv(const Eigen::Quaterniond& q) {
        return cv::Vec4d(q.w(), q.x(), q.y(), q.z());
    }

    static cv::Vec3d quatToRpyRad(const Eigen::Quaterniond& q) {
        const Eigen::Matrix3d R = q.toRotationMatrix();
        const double roll = std::atan2(R(2, 1), R(2, 2));
        const double pitch = std::asin(-std::max(-1.0, std::min(1.0, R(2, 0))));
        const double yaw = std::atan2(R(1, 0), R(0, 0));
        return cv::Vec3d(roll, pitch, yaw);
    }

    vio::ImuSample calibrate(const imuData& raw) const {
        const vio::Vec3 gyro_raw = cvToEig(raw.gyro);
        const vio::Vec3 accel_raw = cvToEig(raw.accel);

        vio::ImuSample s;
        s.t = raw.ts;
        s.gyro = gyro_sm_ * (gyro_raw - fixed_bg_);
        s.accel = accel_sm_ * (accel_raw - fixed_ba_);
        return s;
    }

    ImuFrontendOutput buildOutput(const imuData& raw,
                                  const vio::ImuSample& calibrated,
                                  const vio::ImuEskfOutput& out) const {
        ImuFrontendOutput state;
        state.valid = out.initialized;
        state.initialized = out.initialized;
        state.stationary = out.stationary;
        state.timestamp_sec = calibrated.t;
        state.dt = out.dt;

        state.gyro_raw = raw.gyro;
        state.accel_raw = raw.accel;
        state.gyro_cal = eigToCv(calibrated.gyro);
        state.accel_cal = eigToCv(calibrated.accel);

        state.gyro_bias_dynamic = eigToCv(out.state.bg);
        state.accel_bias_dynamic = eigToCv(out.state.ba);
        state.gyro_bias_total = eigToCv(fixed_bg_ + out.state.bg);
        state.accel_bias_total = eigToCv(fixed_ba_ + out.state.ba);

        state.accel_world = eigToCv(out.accel_world);
        state.vel_xyz = eigToCv(out.state.v);
        state.pos_xyz = eigToCv(out.state.p);
        state.quat_wxyz = quatToCv(out.state.q);
        state.rpy_rad = quatToRpyRad(out.state.q);
        state.rpy_deg = state.rpy_rad * (180.0 / 3.14159265358979323846);

        state.preint_dt = out.current_interval.delta_t;
        state.preint_dtheta = eigToCv(out.current_interval.dtheta);
        state.preint_dv = eigToCv(out.current_interval.dv);
        state.preint_dp = eigToCv(out.current_interval.dp);
        return state;
    }

    vio::Vec3 fixed_bg_ = vio::Vec3::Zero();
    vio::Vec3 fixed_ba_ = vio::Vec3::Zero();
    vio::Vec3 gravity_world_ = vio::Vec3(0.0, 0.0, 9.80665);
    vio::Mat3 gyro_sm_ = vio::Mat3::Identity();
    vio::Mat3 accel_sm_ = vio::Mat3::Identity();
    std::unique_ptr<vio::ImuEskf15> eskf_;
    ImuFrontendOutput last_output_;
};
