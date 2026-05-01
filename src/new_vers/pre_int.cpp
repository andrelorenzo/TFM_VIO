#include "pre_int.hpp"
#include "lie_math.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

Config gconf;

static bool g_attitude_initialized = false;
static bool g_gravity_convention_logged = false;
static size_t g_preint_img_count = 0;

static void SetCovMatrix(vec3 w, vec3 vg, vec3 gg, mat3 xR, Eigen::Matrix<double,18,18> * F, Eigen::Matrix<double,18,12> * G);
static vec3 MeanImuGyr(const std::deque<ImuSample>& imu);
static vec3 MeanImuAcc(const std::deque<ImuSample>& imu);
static vec3 StdImuGyr(const std::deque<ImuSample>& imu, const vec3& mean);
static vec3 StdImuAcc(const std::deque<ImuSample>& imu, const vec3& mean);

void imuPreInit(Config * config) {
    if (config == nullptr) return;
    gconf = *config;
    g_attitude_initialized = false;
    g_gravity_convention_logged = false;
    g_preint_img_count = 0;
}

void imuPreUpdate(SourceIn * source, StateOut * state) {
    if (source == nullptr || state == nullptr) return;
    if (!gconf.gen.imu_on) return;
    if (source->frame.empty() || source->frame.cols <= 0 || source->frame.rows <= 0) return;
    if (source->imu.empty()) return;

    const vec3 block_gyr_mean_raw = MeanImuGyr(source->imu);
    const vec3 block_acc_mean_raw = MeanImuAcc(source->imu);
    const vec3 block_gyr_std_raw = StdImuGyr(source->imu, block_gyr_mean_raw);
    const vec3 block_acc_std_raw = StdImuAcc(source->imu, block_acc_mean_raw);
    const vec3 block_gyr_mean_cal_before = block_gyr_mean_raw - gconf.imu.bg;
    const vec3 block_acc_mean_cal_before = block_acc_mean_raw - gconf.imu.ba;

    const double gmag = gconf.imu.gv.norm();
    if (gmag <= 1e-9) {
        Logger(WARN, "[PREINT_SKIP] invalid gravity vector gv=[%.6f %.6f %.6f]", gconf.imu.gv.x(), gconf.imu.gv.y(), gconf.imu.gv.z());
        return;
    }

    if (!g_gravity_convention_logged) {
        g_gravity_convention_logged = true;
        Logger(INFO, "[GRAV_CONVENTION] camera/world convention assumed: x=right y=down z=forward. imu.gv is accelerometer gravity-at-rest vector, not physical acceleration. gv=[%.6f %.6f %.6f] physical_g=[%.6f %.6f %.6f]", gconf.imu.gv.x(), gconf.imu.gv.y(), gconf.imu.gv.z(), -gconf.imu.gv.x(), -gconf.imu.gv.y(), -gconf.imu.gv.z());
    }

    if (!g_attitude_initialized) {
        const double acc_norm = block_acc_mean_cal_before.norm();
        Logger(INFO, "[IMU_INIT_CHECK] initialized=%d imu=%zu acc=[%.6f %.6f %.6f] acc_norm=%.6f gv=[%.6f %.6f %.6f] gmag=%.6f", g_attitude_initialized ? 1 : 0, source->imu.size(), block_acc_mean_cal_before.x(), block_acc_mean_cal_before.y(), block_acc_mean_cal_before.z(), acc_norm, gconf.imu.gv.x(), gconf.imu.gv.y(), gconf.imu.gv.z(), gmag);

        if (acc_norm > 1e-6) {
            state->quat_rad = quat::FromTwoVectors(block_acc_mean_cal_before.normalized(), gconf.imu.gv.normalized());
            state->quat_rad.normalize();
            state->rpy_rad = quatToCameraRpyRad(state->quat_rad);
            state->gravv = gconf.imu.gv.normalized();
            g_attitude_initialized = true;
            Logger(INFO, "[IMU_INIT_GRAVITY] acc=[%.6f %.6f %.6f] acc_norm=%.6f gv=[%.6f %.6f %.6f] rpy=[%.6f %.6f %.6f]", block_acc_mean_cal_before.x(), block_acc_mean_cal_before.y(), block_acc_mean_cal_before.z(), acc_norm, gconf.imu.gv.x(), gconf.imu.gv.y(), gconf.imu.gv.z(), state->rpy_rad.x(), state->rpy_rad.y(), state->rpy_rad.z());
        }
        else {
            state->gravv = gconf.imu.gv.normalized();
            Logger(WARN, "[IMU_INIT_FAIL] acc norm too small acc_norm=%.9f", acc_norm);
        }
    }

    const double acc_norm = block_acc_mean_cal_before.norm();
    const double acc_norm_err = std::abs(acc_norm - gmag);
    const double gravity_acc_norm_tol = std::max(0.35, gconf.imu.stationary_acc_mean_tol);
    const bool gravity_obs_ok = acc_norm > 1e-6 && acc_norm_err <= gravity_acc_norm_tol;

    mat3 R_pre_tilt = state->quat_rad.toRotationMatrix();
    const vec3 f_grav_body_pred_pre = R_pre_tilt.transpose() * gconf.imu.gv;
    double tilt_err_deg = 0.0;
    bool tilt_applied = false;

    if (g_attitude_initialized && gravity_obs_ok && f_grav_body_pred_pre.norm() > 1e-6) {
        quat q_body_err = quat::FromTwoVectors(f_grav_body_pred_pre.normalized(), block_acc_mean_cal_before.normalized());
        q_body_err.normalize();

        const double sin_half = q_body_err.vec().norm();
        const double cos_half = std::abs(q_body_err.w());
        tilt_err_deg = 2.0 * std::atan2(sin_half, cos_half) * 180.0 / M_PI;

        const double max_tilt_corr_deg = 5.0;
        const double base_alpha = 0.10;
        double alpha = base_alpha;

        if (tilt_err_deg > max_tilt_corr_deg && tilt_err_deg > 1e-9) {
            alpha *= max_tilt_corr_deg / tilt_err_deg;
        }

        alpha = std::clamp(alpha, 0.0, 1.0);

        if (alpha > 0.0) {
            quat q_part = quat::Identity().slerp(alpha, q_body_err);
            state->quat_rad = normalizeQ(state->quat_rad * q_part.conjugate());
            state->rpy_rad = quatToCameraRpyRad(state->quat_rad);
            state->gravv = gconf.imu.gv.normalized();
            tilt_applied = true;
        }
    }

    const mat3 R_state = state->quat_rad.toRotationMatrix();
    const vec3 f_grav_body_dbg = R_state.transpose() * gconf.imu.gv;
    const vec3 lin_acc_body_dbg = block_acc_mean_cal_before - f_grav_body_dbg;
    const vec3 lin_acc_world_dbg = R_state * lin_acc_body_dbg;

    const int block_samples = static_cast<int>(source->imu.size());
    const int low_dynamic_min_samples = std::max(3, std::min(gconf.imu.stationary_min_samples, 5));
    const bool low_dynamic_block = gconf.imu.stationary_enable && block_samples >= low_dynamic_min_samples && gravity_obs_ok && lin_acc_world_dbg.norm() <= std::max(0.75, gconf.imu.stationary_acc_mean_tol) && block_gyr_mean_cal_before.norm() <= std::max(0.20, gconf.imu.stationary_gyro_mean_max) && block_gyr_std_raw.maxCoeff() <= std::max(0.10, gconf.imu.stationary_gyro_std_max) && block_acc_std_raw.maxCoeff() <= std::max(0.35, gconf.imu.stationary_acc_std_max);

    if (low_dynamic_block && gconf.imu.bias_refine_enable) {
        const double alpha = std::clamp(gconf.imu.stationary_bias_alpha, 0.0, 1.0);
        const vec3 bg_target = block_gyr_mean_raw;
        const vec3 ba_target = block_acc_mean_raw - f_grav_body_dbg;
        gconf.imu.bg = (1.0 - alpha) * gconf.imu.bg + alpha * bg_target;
        gconf.imu.ba = (1.0 - alpha) * gconf.imu.ba + alpha * ba_target;
    }

    const ImuSample& last_imu = source->imu.back();
    state->ts_ms = source->frame_tsms;
    state->deb.acc_ms2 = last_imu.vacc;
    state->acc_cal_ms2 = last_imu.vacc - gconf.imu.ba;
    state->deb.gyr_rads = last_imu.vgyr;
    state->gyr_cal_rads = last_imu.vgyr - gconf.imu.bg;

    Eigen::Matrix<double,18,18> F, Phi, Psi;
    Eigen::Matrix<double,18,12> G;
    Eigen::Matrix<double,18,18> Q, P;

    Eigen::Matrix<double,12,12> S = Eigen::Matrix<double,12,12>::Zero();
    S.block<3,3>(0,0) = diagSquare(gconf.imu.gyroAllanN());
    S.block<3,3>(3,3) = diagSquare(gconf.imu.gyroAllanK());
    S.block<3,3>(6,6) = diagSquare(gconf.imu.accelAllanN());
    S.block<3,3>(9,9) = diagSquare(gconf.imu.accelAllanK());

    F.setZero();
    Phi.setZero();
    Psi.setIdentity();
    G.setZero();
    Q.setZero();
    P.setZero();

    const quat q_i = normalizeQ(state->quat_rad);
    const mat3 R_i = q_i.toRotationMatrix();
    const vec3 p_i = state->pos_m;
    const vec3 v_i = state->vel_ms;

    quat q_prop = q_i;
    vec3 p_prop = p_i;
    vec3 v_prop = v_i;
    mat3 dR_local = mat3::Identity();

    double Dt = 0.0;
    double sum_dt_ms = 0.0;
    double min_dt_ms = std::numeric_limits<double>::infinity();
    double max_dt_ms = 0.0;
    int valid_steps = 0;
    bool ts_ok = true;

    for (size_t i = 0; i < source->imu.size(); ++i) {
        const ImuSample& m = source->imu[i];
        const double dt_ms = m.dt;
        const double dt = 1e-3 * dt_ms;

        if (!std::isfinite(dt_ms) || dt_ms <= 0.0) {
            ts_ok = false;
            continue;
        }

        const vec3 w = m.vgyr - gconf.imu.bg;
        const vec3 f_meas_body = m.vacc - gconf.imu.ba;
        const mat3 R_wb = q_prop.toRotationMatrix();
        const vec3 f_grav_body = R_wb.transpose() * gconf.imu.gv;
        const vec3 acc_linear_body = f_meas_body - f_grav_body;
        const vec3 acc_linear_world = R_wb * acc_linear_body;

        Dt += dt;
        sum_dt_ms += dt_ms;
        min_dt_ms = std::min(min_dt_ms, dt_ms);
        max_dt_ms = std::max(max_dt_ms, dt_ms);
        ++valid_steps;

        const vec3 vg_cov = R_i.transpose() * v_prop;
        const vec3 gg_cov = R_i.transpose() * gconf.imu.gv.normalized();
        SetCovMatrix(w, vg_cov, gg_cov, dR_local, &F, &G);

        Phi = Eigen::Matrix<double,18,18>::Identity() + dt * F;
        Q = dt * G * S * G.transpose();
        P = Phi * P * Phi.transpose() + Q;
        P = 0.5 * (P + P.transpose()).eval();
        Psi.applyOnTheLeft(Phi);

        p_prop += v_prop * dt + 0.5 * acc_linear_world * dt * dt;
        v_prop += acc_linear_world * dt;

        const mat3 DR = expSO3(w * dt, gconf.imu.small_angle);
        q_prop = normalizeQ(q_prop * quat(DR));
        dR_local = dR_local * DR;
    }

    if (valid_steps == 0 || Dt <= 0.0) return;

    if (low_dynamic_block && gconf.imu.zupt_enable) {
        p_prop = p_i + v_i * Dt;
        v_prop = v_i;
    }

    const quat dq_ij = normalizeQ(q_i.conjugate() * q_prop);
    const vec3 dp_ij = R_i.transpose() * (p_prop - p_i - v_i * Dt);
    const vec3 dv_ij = R_i.transpose() * (v_prop - v_i);

    state->deb.imu_dp = dp_ij;
    state->deb.imu_dv = dv_ij;
    state->deb.imu_dq = dq_ij;
    state->deb.imu_drpy = quatToCameraRpyRad(dq_ij);
    state->deb.imu_stationary = low_dynamic_block;

    state->pos_m = p_prop;
    state->vel_ms = v_prop;
    state->gravv = gconf.imu.gv.normalized();
    state->quat_rad = normalizeQ(q_prop);
    state->rpy_rad = quatToCameraRpyRad(state->quat_rad);

    state->deb.imu_xyz = state->pos_m;
    state->deb.imu_rpy = state->rpy_rad;
    state->dt = Dt;

    const Eigen::Matrix<double,15,15> P15 = P.bottomRightCorner<15,15>().cast<double>();
    Eigen::Matrix<double,15,15> Info;

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(P15);
    if (qr.isInvertible()) Info = qr.inverse();
    else Info = pseudoInverse(P15);

    const Eigen::Matrix<double,15,15> P15_sym = 0.5 * (P15 + P15.transpose()).eval();
    Info = 0.5 * (Info + Info.transpose()).eval();

    Eigen::LLT<Eigen::Matrix<double,15,15>> llt(Info);
    if (llt.info() != Eigen::Success) {
        Info += 1e-9 * Eigen::Matrix<double,15,15>::Identity();
        llt.compute(Info);
    }

    Eigen::Matrix<double,15,15> Low = Eigen::Matrix<double,15,15>::Identity();
    if (llt.info() == Eigen::Success) Low = llt.matrixL();
    else Logger(WARN, "[PREINT_WARN] Info LLT failed img=%zu", g_preint_img_count);

    Eigen::Matrix<double,15,27> H_local;
    H_local << Psi.bottomLeftCorner<15,3>().cast<double>(), Psi.bottomRightCorner<15,9>().cast<double>(), -Eigen::Matrix<double,15,15>::Identity();

    state->H = H_local;
    state->H.applyOnTheLeft(Low.adjoint());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15>> eig_p15(P15_sym);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15>> eig_info(Info);

    const double imu_to_frame_ms = last_imu.ts - source->frame_tsms;
    const double frame_dt_err_ms = sum_dt_ms - source->frame_dtms;

    Logger(INFO, "[GRAV_TILT] img=%zu applied=%d obs_ok=%d acc_norm=%.6f acc_norm_err=%.6f tilt_err_deg=%.6f rpy=[%.6f %.6f %.6f]", g_preint_img_count, tilt_applied ? 1 : 0, gravity_obs_ok ? 1 : 0, acc_norm, acc_norm_err, tilt_err_deg, state->rpy_rad.x(), state->rpy_rad.y(), state->rpy_rad.z());

    Logger(INFO, "[GRAV_COMP] img=%zu f_meas_body=[%.6f %.6f %.6f] f_grav_body=[%.6f %.6f %.6f] lin_body=[%.6f %.6f %.6f] lin_world=[%.6f %.6f %.6f] lin_norm=%.6f low_dyn=%d", g_preint_img_count, block_acc_mean_cal_before.x(), block_acc_mean_cal_before.y(), block_acc_mean_cal_before.z(), f_grav_body_dbg.x(), f_grav_body_dbg.y(), f_grav_body_dbg.z(), lin_acc_body_dbg.x(), lin_acc_body_dbg.y(), lin_acc_body_dbg.z(), lin_acc_world_dbg.x(), lin_acc_world_dbg.y(), lin_acc_world_dbg.z(), lin_acc_world_dbg.norm(), low_dynamic_block ? 1 : 0);

    Logger(INFO, "[PREINT] img=%zu frame=%dx%d ts=%.3f frame_dt=%.3f imu=%zu valid_dt=%d Dt=%.6f sum_dt_ms=%.3f frame_dt_err=%.6f min_dt_ms=%.3f max_dt_ms=%.3f t0=%.3f t1=%.3f imu_to_frame=%.6f ts_ok=%d rank=%d", g_preint_img_count, source->frame.cols, source->frame.rows, source->frame_tsms, source->frame_dtms, source->imu.size(), valid_steps, Dt, sum_dt_ms, frame_dt_err_ms, min_dt_ms, max_dt_ms, source->imu.front().ts, source->imu.back().ts, imu_to_frame_ms, ts_ok ? 1 : 0, qr.rank());

    Logger(INFO, "[PREINT_STATE] pos=[%.6f %.6f %.6f] vel=[%.6f %.6f %.6f] rpy=[%.6f %.6f %.6f] acc_raw=[%.6f %.6f %.6f] acc_cal=[%.6f %.6f %.6f] gyr_raw=[%.6f %.6f %.6f] gyr_cal=[%.6f %.6f %.6f] dp=[%.6f %.6f %.6f] dv=[%.6f %.6f %.6f]", state->pos_m.x(), state->pos_m.y(), state->pos_m.z(), state->vel_ms.x(), state->vel_ms.y(), state->vel_ms.z(), state->rpy_rad.x(), state->rpy_rad.y(), state->rpy_rad.z(), state->deb.acc_ms2.x(), state->deb.acc_ms2.y(), state->deb.acc_ms2.z(), state->acc_cal_ms2.x(), state->acc_cal_ms2.y(), state->acc_cal_ms2.z(), state->deb.gyr_rads.x(), state->deb.gyr_rads.y(), state->deb.gyr_rads.z(), state->gyr_cal_rads.x(), state->gyr_cal_rads.y(), state->gyr_cal_rads.z(), dp_ij.x(), dp_ij.y(), dp_ij.z(), dv_ij.x(), dv_ij.y(), dv_ij.z());

    Logger(INFO, "[PREINT_COV] img=%zu p15_min=%.6e p15_max=%.6e info_min=%.6e info_max=%.6e allans_valid=%d", g_preint_img_count, eig_p15.eigenvalues().minCoeff(), eig_p15.eigenvalues().maxCoeff(), eig_info.eigenvalues().minCoeff(), eig_info.eigenvalues().maxCoeff(), gconf.imu.hasValidAllan() ? 1 : 0);

    ++g_preint_img_count;
    source->imu.clear();
}

static vec3 MeanImuGyr(const std::deque<ImuSample>& imu) {
    if (imu.empty()) return vec3::Zero();
    vec3 mean = vec3::Zero();
    for (const ImuSample& s : imu) mean += s.vgyr;
    return mean / static_cast<double>(imu.size());
}

static vec3 MeanImuAcc(const std::deque<ImuSample>& imu) {
    if (imu.empty()) return vec3::Zero();
    vec3 mean = vec3::Zero();
    for (const ImuSample& s : imu) mean += s.vacc;
    return mean / static_cast<double>(imu.size());
}

static vec3 StdImuGyr(const std::deque<ImuSample>& imu, const vec3& mean) {
    if (imu.size() < 2) return vec3::Zero();
    vec3 var = vec3::Zero();
    for (const ImuSample& s : imu) {
        const vec3 d = s.vgyr - mean;
        var += d.cwiseProduct(d);
    }
    var /= static_cast<double>(imu.size() - 1);
    return var.cwiseSqrt();
}

static vec3 StdImuAcc(const std::deque<ImuSample>& imu, const vec3& mean) {
    if (imu.size() < 2) return vec3::Zero();
    vec3 var = vec3::Zero();
    for (const ImuSample& s : imu) {
        const vec3 d = s.vacc - mean;
        var += d.cwiseProduct(d);
    }
    var /= static_cast<double>(imu.size() - 1);
    return var.cwiseSqrt();
}

static void SetCovMatrix(vec3 w, vec3 vg, vec3 gg, mat3 xR, Eigen::Matrix<double,18,18> * F, Eigen::Matrix<double,18,12> * G) {
    const mat3 I = mat3::Identity();

    F->block<3,3>(3,3) = -skewMat(w);
    F->block<3,3>(3,12) = -I;

    F->block<3,3>(6,3) = -xR * skewMat(vg);
    F->block<3,3>(6,9) = xR.transpose();

    F->block<3,3>(9,0) = -gconf.imu.gv.norm() * xR;
    F->block<3,3>(9,3) = -gconf.imu.gv.norm() * skewMat(gg);
    F->block<3,3>(9,9) = -skewMat(w);
    F->block<3,3>(9,12) = -skewMat(vg);
    F->block<3,3>(9,15) = -I;

    G->block<3,3>(3,0) = -I;
    G->block<3,3>(9,0) = -skewMat(vg);
    G->block<3,3>(9,6) = -I;
    G->block<3,3>(12,3) = I;
    G->block<3,3>(15,9) = I;
}
