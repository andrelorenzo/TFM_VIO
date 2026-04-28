#include "pre_int.hpp"
#include "lie_math.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <vector>
#include "numeric"



Config gconf;

void MakeSureTimestamps(std::deque<double>& ts, std::deque<vec3>& data) ;
bool ResampleAccToGyroInPlace(SourceIn* src) ;
static void SetCovMatrix(vec3 w, vec3 vg, vec3 gg, mat3 xR, Eigen::Matrix<double, 18, 18> * F, Eigen::Matrix<double,18,12> * G);
static vec3 MeanDequeVec3(const std::deque<vec3>& data);
static vec3 StdDequeVec3(const std::deque<vec3>& data, const vec3& mean);



void imuPreInit(Config * config){
    if (config == nullptr) return;
    gconf = *config;
}


void imuPreUpdate(SourceIn * source, StateOut * state){
    if (source == nullptr || state == nullptr) return;


    if(!gconf.gen.imu_on && !gconf.gen.color_on)return;                         // None => we cannot do anythin
    if(gconf.gen.color_on && source->frame.empty())return;                      // must calculate only when frame arrived
    if(gconf.gen.imu_on && (source->acc.empty() || source->gyr.empty()))return; // Only if we have new data


    static bool first = true;
    if(first){
        state->gravv = gconf.imu.gv.normalized();
        first = false;
    }

    if(!ResampleAccToGyroInPlace(source))return;   // Resample => same size for gyro and accel, if too few meas, return

    const vec3 block_gyr_mean_raw = MeanDequeVec3(source->gyr);
    const vec3 block_acc_mean_raw = MeanDequeVec3(source->acc);
    const vec3 block_gyr_std_raw = StdDequeVec3(source->gyr, block_gyr_mean_raw);
    const vec3 block_acc_std_raw = StdDequeVec3(source->acc, block_acc_mean_raw);
    const vec3 block_gyr_mean_cal_before = block_gyr_mean_raw - gconf.imu.bg;
    const vec3 block_acc_mean_cal_before = block_acc_mean_raw - gconf.imu.ba;
    const int block_samples = static_cast<int>(std::min(source->acc.size(), source->gyr.size()));
    const bool stationary_block =
        gconf.imu.stationary_enable &&
        block_samples >= gconf.imu.stationary_min_samples &&
        (block_gyr_mean_cal_before.norm() <= gconf.imu.stationary_gyro_mean_max) &&
        ((block_acc_mean_cal_before - gconf.imu.gv).norm() <= gconf.imu.stationary_acc_mean_tol) &&
        (block_gyr_std_raw.maxCoeff() <= gconf.imu.stationary_gyro_std_max) &&
        (block_acc_std_raw.maxCoeff() <= gconf.imu.stationary_acc_std_max);

    if (stationary_block && gconf.imu.bias_refine_enable) {
        // Adapt fixed biases only while we are confident the sensor is static.
        // This makes the next calibration step less sensitive to tiny vibrations.
        const double alpha = std::clamp(gconf.imu.stationary_bias_alpha, 0.0, 1.0);
        const vec3 bg_target = block_gyr_mean_raw;
        const vec3 ba_target = block_acc_mean_raw - gconf.imu.gv;
        gconf.imu.bg = (1.0 - alpha) * gconf.imu.bg + alpha * bg_target;
        gconf.imu.ba = (1.0 - alpha) * gconf.imu.ba + alpha * ba_target;
    }

    // source->print();
    state->ts_ms = source->frame_tsms > 0.0 ? source->frame_tsms : source->gyr_tsms.back();
    // Keep raw data for debug
    state->deb.acc_ms2  = source->acc.back();
    state->acc_cal_ms2  = source->acc.back() - gconf.imu.ba;
    state->deb.gyr_rads = source->gyr.back();
    state->gyr_cal_rads = source->gyr.back() - gconf.imu.bg;

    const double gmag = gconf.imu.gv.norm();
    vec3 vg = state->vel_ms;     // velocidad al inicio del bloque (global)
    vec3 gg = state->gravv.norm() > 1e-9 ? state->gravv.normalized() : gconf.imu.gv.normalized();
    vec3 vg2 = vg;
    vec3 gg2 = gg;
    vec3 pg = vec3::Zero();      // posicion al inicio del bloque (global)

    // Error := de' = F*de + G*n
    // as, de = [ dg, dtheta, dp, dv, dbg, dba ]
    // as, n = [Ng, Kg, Na, Ka]
    Eigen::Matrix<double,18,18> F, Phi, Psi;
    Eigen::Matrix<double,18,12> G;
    Eigen::Matrix<double,18,18> Q, P;

    // Allan variance noise (Cov = Allan^2)
    Eigen::Matrix<double, 12, 12> S = Eigen::Matrix<double, 12, 12>::Zero();
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

    mat3 xR = mat3::Identity();;
    vec3 xp = vec3::Zero();
    vec3 xv = vec3::Zero();

    double Dt = 0.0;
    for(int i = 1; i < source->acc.size(); ++i){

        // Get measures
        double dt = 1e-3 * source->gyr_dtms.at(i - 1);
        Dt += dt;
        vec3 wm = source->gyr.at(i);
        vec3 am = source->acc.at(i);

        // Calibrate measures
        vec3 w = wm - gconf.imu.bg;
        vec3 a = am - gconf.imu.ba;

        // Error model (Covariance)
        SetCovMatrix(w, vg, gg, xR, &F, &G);
        Phi = Eigen::Matrix<double,18,18>::Identity() + dt*F;

        Q = dt * G * S * (G.transpose());
        // Accumulated Covariance
        P = Phi*P*(Phi.transpose())+Q;
        P = .5*(P+P.transpose()).eval();
        // Accumulated transition
        Psi.applyOnTheLeft(Phi);


        // Integrate with rotation at the beginning of the interval.
        const mat3 xR_prev = xR;
        const vec3 j2a = xR_prev.transpose() * computeJ2SO3(w, dt, gconf.imu.small_angle) * a;
        const vec3 j1a = xR_prev.transpose() * computeJ1SO3(w, dt, gconf.imu.small_angle) * a;

        // Integrate position and velocity pre-deltas in the initial frame.
        xp = xp + xv*dt + j2a;
        xv = xv + j1a;

        // Integrate orientation
        mat3 DR = expSO3(w*dt, gconf.imu.small_angle);
        xR.applyOnTheLeft(DR);

        // These variables are re-fed into F/G in the next iteration.
        // Keep this propagation untouched: it is local to the covariance model.
        pg = vg2*Dt - 0.5 * gmag * gg2 * Dt*Dt + xp;
        vg = xR * (vg2 - gmag * gg2 * Dt + xv);
        gg = xR * gg2;
        gg.normalize();
    }


    mat3 dR_ij = xR.transpose();
    vec3 dv_ij = (-gmag * gg2 * Dt) + xv;
    vec3 dp_ij = (-0.5 * gmag * gg2 * Dt * Dt) + xp;
    quat dq_ij = normalizeQ(quat(dR_ij));
    const mat3 R_i = state->quat_rad.toRotationMatrix();
    const vec3 p_i = state->pos_m;
    const vec3 v_i = state->vel_ms;

    if (stationary_block && gconf.imu.zupt_enable) {
        // If the whole block looks stationary, suppress the local motion increment.
        // This prevents tiny vibrations from integrating into velocity/position drift.
        dR_ij = mat3::Identity();
        dv_ij = vec3::Zero();
        dp_ij = vec3::Zero();
        dq_ij = quat::Identity();
        gg = gconf.imu.gv.normalized();
    }

    // Local preintegrated solution for the current IMU block.
    // These deltas are consumed by the visual updater and must stay local.
    state->deb.imu_dp = dp_ij;
    state->deb.imu_dv = dv_ij;
    state->deb.imu_dq = dq_ij;
    state->deb.imu_drpy = quatToCameraRpyRad(dq_ij);
    state->deb.imu_stationary = stationary_block;

    // Persistent/global state update.
    // Important: compose the local deltas with the previous global state;
    // do not replace pos/vel/quat directly by the local IMU solution.
    state->pos_m = p_i + v_i * Dt + R_i * dp_ij;
    state->vel_ms = v_i + R_i * dv_ij;
    if (stationary_block && gconf.imu.zupt_enable) {
        state->pos_m = p_i;
        state->vel_ms = vec3::Zero();
    }
    state->gravv = gg;
    // Logger(DEBUG,
    //     "PREINT FINAL Dt=%.6f \n\tdR=[\t[%.9f %.9f %.9f] \n\t\t[%.9f %.9f %.9f] \n\t\t[%.9f %.9f %.9f]] \n\tdv=[%.9f %.9f %.9f] \n\tdp=[%.9f %.9f %.9f] \n\tp=[%.9f %.9f %.9f]\n",
    //     Dt,
    //     dR_ij(0,0), dR_ij(0,1), dR_ij(0,2),
    //     dR_ij(1,0), dR_ij(1,1), dR_ij(1,2),
    //     dR_ij(2,0), dR_ij(2,1), dR_ij(2,2),
    //     dv_ij.x(), dv_ij.y(), dv_ij.z(),
    //     dp_ij.x(), dp_ij.y(), dp_ij.z(),
    //     state->pos_m.x(), state->pos_m.y(), state->pos_m.z());
    // Logger(DEBUG,
    //     "IMU STAT stationary=%d gyr_mean=[%.9f %.9f %.9f] acc_mean=[%.9f %.9f %.9f] gyr_std=[%.9f %.9f %.9f] acc_std=[%.9f %.9f %.9f] bg=[%.9f %.9f %.9f] ba=[%.9f %.9f %.9f]",
    //     static_cast<int>(stationary_block),
    //     block_gyr_mean_cal_before.x(), block_gyr_mean_cal_before.y(), block_gyr_mean_cal_before.z(),
    //     block_acc_mean_cal_before.x(), block_acc_mean_cal_before.y(), block_acc_mean_cal_before.z(),
    //     block_gyr_std_raw.x(), block_gyr_std_raw.y(), block_gyr_std_raw.z(),
    //     block_acc_std_raw.x(), block_acc_std_raw.y(), block_acc_std_raw.z(),
    //     gconf.imu.bg.x(), gconf.imu.bg.y(), gconf.imu.bg.z(),
    //     gconf.imu.ba.x(), gconf.imu.ba.y(), gconf.imu.ba.z());
    state->quat_rad = normalizeQ(state->quat_rad * dq_ij);
    state->rpy_rad = quatToCameraRpyRad(state->quat_rad);

    
    // IMU-only global pose for debug/log/plots.
    // This is the propagated navigation state before any visual correction.
    state->deb.imu_xyz = state->pos_m;
    state->deb.imu_rpy = state->rpy_rad;
    state->dt = Dt;
    // state->ts_ms = source->frame_tsms;

    // Consume all data => wait for new only


    // Local IMU factor for the visual backend
    // This is stored only for later use by VIO and must not modify the
    // persistent state propagated above.
    const Eigen::Matrix<double,15,15> P15 = P.bottomRightCorner<15,15>().cast<double>();
    Eigen::Matrix<double,15,15> Info;

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(P15);
    const int rank_p15 = qr.rank();
    const bool p15_invertible = qr.isInvertible();
    if (qr.isInvertible()) {
        Info = qr.inverse();
    } else {
        Info = pseudoInverse(P15);
    }

    const Eigen::Matrix<double,15,15> P15_sym = 0.5 * (P15 + P15.transpose()).eval();
    Info = 0.5 * (Info + Info.transpose()).eval();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15>> eig_p15(P15_sym);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15>> eig_info(Info);
    const Eigen::Matrix<double,15,15> Low = Info.llt().matrixL();

    Eigen::Matrix<double,15,27> H_local;
    H_local << Psi.bottomLeftCorner<15,3>().cast<double>(), Psi.bottomRightCorner<15,9>().cast<double>(), -Eigen::Matrix<double,15,15>::Identity();

    state->H = H_local;
    state->H.applyOnTheLeft(Low.adjoint());

    // if (gconf.gen.debug) {
    //     // Whitening check: ||H_white dx||^2 must match (H_raw dx)^T Info (H_raw dx).
    //     // If this fails, the issue is in P15/Info/Low, not in the visual backend.
    //     Eigen::Matrix<double,27,1> dx = Eigen::Matrix<double,27,1>::Random();
    //     const Eigen::Matrix<double,15,1> r_raw = H_local * dx;
    //     const Eigen::Matrix<double,15,1> r_white = state->H * dx;
    //     const double e_info = r_raw.dot(Info * r_raw);
    //     const double e_white = r_white.squaredNorm();
    //     const double rel_err = std::abs(e_info - e_white) / std::max((double)1.0, std::abs(e_info));

    //     // Cholesky reconstruction check: Info should be close to Low * Low^T.
    //     const Eigen::Matrix<double,15,15> Info_rec = Low * Low.transpose();
    //     const double chol_err = (Info - Info_rec).norm() / std::max((double)1.0, Info.norm());
    //     const bool llt_ok = (Info.llt().info() == Eigen::Success);
    //     const double min_diag = Low.diagonal().minCoeff();
    //     const double p15_min_eig = eig_p15.info() == Eigen::Success ? eig_p15.eigenvalues().minCoeff() : std::numeric_limits<double>::quiet_NaN();
    //     const double p15_max_eig = eig_p15.info() == Eigen::Success ? eig_p15.eigenvalues().maxCoeff() : std::numeric_limits<double>::quiet_NaN();
    //     const double info_min_eig = eig_info.info() == Eigen::Success ? eig_info.eigenvalues().minCoeff() : std::numeric_limits<double>::quiet_NaN();
    //     const double info_max_eig = eig_info.info() == Eigen::Success ? eig_info.eigenvalues().maxCoeff() : std::numeric_limits<double>::quiet_NaN();
    //     const double p15_min_diag = P15.diagonal().minCoeff();
    //     const double p15_max_diag = P15.diagonal().maxCoeff();

        
    //     Logger(DEBUG,
    //         "(%zu/%zu) H CHECK rank=%d inv=%d p15_diag=[%.9g,%.9g] p15_eig=[%.9g,%.9g] info_eig=[%.9g,%.9g] llt_ok=%d min_diag=%.9g chol_err=%.9g e_info=%.9g e_white=%.9g rel_err=%.9g",
    //         source->acc.size(), source->gyr.size(),
    //         rank_p15, static_cast<int>(p15_invertible),
    //         p15_min_diag, p15_max_diag,
    //         p15_min_eig, p15_max_eig,
    //         info_min_eig, info_max_eig,
    //         static_cast<int>(llt_ok), min_diag, chol_err, e_info, e_white, rel_err);

    //     const mat3 R_final = state->quat_rad.toRotationMatrix();
    //     Logger(DEBUG,
    //         "PREINT FINAL STATE Dt=%.6f\n\tR=[\t[%.9f %.9f %.9f]\n\t\t[%.9f %.9f %.9f]\n\t\t[%.9f %.9f %.9f]]\n\tp=[%.9f %.9f %.9f]",
    //         Dt,
    //         R_final(0,0), R_final(0,1), R_final(0,2),
    //         R_final(1,0), R_final(1,1), R_final(1,2),
    //         R_final(2,0), R_final(2,1), R_final(2,2),
    //         state->pos_m.x(), state->pos_m.y(), state->pos_m.z());
    // }
    source->acc.clear();
    source->acc_dtms.clear();
    source->acc_tsms.clear();
    source->gyr.clear();
    source->gyr_dtms.clear();
    source->gyr_tsms.clear();
}

static vec3 MeanDequeVec3(const std::deque<vec3>& data) {
    if (data.empty()) {
        return vec3::Zero();
    }

    vec3 mean = vec3::Zero();
    for (const auto& v : data) {
        mean += v;
    }
    return mean / static_cast<double>(data.size());
}

static vec3 StdDequeVec3(const std::deque<vec3>& data, const vec3& mean) {
    if (data.size() < 2) {
        return vec3::Zero();
    }

    vec3 var = vec3::Zero();
    for (const auto& v : data) {
        const vec3 d = v - mean;
        var += d.cwiseProduct(d);
    }
    var /= static_cast<double>(data.size() - 1);
    return var.cwiseSqrt();
}

static void SetCovMatrix(vec3 w, vec3 vg, vec3 gg, mat3 xR, Eigen::Matrix<double, 18, 18> * F, Eigen::Matrix<double,18,12> * G){
/* F
    | [0]    [0]     [0] [0]   [0]  [0]  |
    | [0]    [-SW]   [0] [0]   [-I] [0]  |
    | [0]    [-Rt*v] [0] [Rt]  [0]  [0]  |
    | [-g*R] [-g*gv] [0] [-SW] [-v] [-I] |
    | [0]    [0]     [0] [0]   [0]  [0]  |
    | [0]    [0]     [0] [0]   [0]  [0]  |
*/


    const mat3 I = mat3::Identity();

    F->block<3,3>(3,3)   = -skewMat(w);
    F->block<3,3>(3,12)  = -I;

    F->block<3,3>(6,3)   = -xR * skewMat(vg);
    F->block<3,3>(6,9)   = xR.transpose();

    F->block<3,3>(9,0)   = -gconf.imu.gv.norm() * xR;
    F->block<3,3>(9,3)   = -gconf.imu.gv.norm() * skewMat(gg);
    F->block<3,3>(9,9)   = -skewMat(w);
    F->block<3,3>(9,12)  = -skewMat(vg);
    F->block<3,3>(9,15)  = -I;

/* G
    | [0]  [0] [0]  [0] |
    | [-I] [0] [0]  [0] |
    | [0]  [0] [0]  [0] |
    | [-v] [0] [-I] [0] |
    | [0]  [I] [0]  [0] |
    | [0]  [0] [0]  [I] |
*/
    G->block<3,3>(3,0)   = -I;
    G->block<3,3>(9,0)   = -skewMat(vg);
    G->block<3,3>(9,6)   = -I;
    G->block<3,3>(12,3)  = I;
    G->block<3,3>(15,9)  = I;

}

static inline void RecomputeDtms(const std::deque<double>& ts, std::deque<double>* dt) {
    dt->clear();
    if (!dt) return;
    if (ts.size() < 2) return;

    dt->resize(ts.size() - 1);
    for (size_t i = 0; i + 1 < ts.size(); ++i) {
        (*dt)[i] = ts[i + 1] - ts[i];
    }
}

static inline vec3 LerpVec3(const vec3& a, const vec3& b, double alpha) {
    return (1.0 - alpha) * a + alpha * b;
}

bool ResampleAccToGyroInPlace(SourceIn* src) {
    if (!src) return false;

    if (src->acc.size() < 2 || src->acc_tsms.size() < 2) return false;
    if (src->gyr.empty() || src->gyr_tsms.empty()) return false;

    if (src->acc.size() != src->acc_tsms.size()) return false;
    if (src->gyr.size() != src->gyr_tsms.size()) return false;

    const double tf = src->frame_tsms;
    if (tf > 0.0) {
        auto add_frame_sample = [tf](std::deque<double>& ts, std::deque<vec3>& data) {
            if (ts.size() < 2 || data.size() < 2) return;
            if (tf <= ts.back()) return;

            const size_t n = ts.size();
            const double t0 = ts[n - 2];
            const double t1 = ts[n - 1];
            if (t1 <= t0) return;

            data.push_back(LerpVec3(data[n - 2], data[n - 1], (tf - t0) / (t1 - t0)));
            ts.push_back(tf);
        };

        add_frame_sample(src->acc_tsms, src->acc);
        add_frame_sample(src->gyr_tsms, src->gyr);
    }

    const double acc_t_min = src->acc_tsms.front();
    const double acc_t_max = src->acc_tsms.back();

    std::deque<vec3> new_gyr;
    std::deque<double> new_gyr_tsms;

    std::deque<vec3> new_acc;
    std::deque<double> new_acc_tsms;

    size_t k = 0; // indice del tramo [acc[k], acc[k+1]]

    for (size_t i = 0; i < src->gyr_tsms.size(); ++i) {
        const double tg = src->gyr_tsms[i];

        // Solo aceptamos tiempos dentro del rango interpolable del accel
        if (tg < acc_t_min || tg > acc_t_max) {
            continue;
        }

        while (k + 1 < src->acc_tsms.size() && src->acc_tsms[k + 1] < tg) {
            ++k;
        }

        if (k + 1 >= src->acc_tsms.size()) {
            break;
        }

        const double t0 = src->acc_tsms[k];
        const double t1 = src->acc_tsms[k + 1];

        if (!(t0 <= tg && tg <= t1)) {
            continue;
        }

        const double dt = t1 - t0;
        if (dt <= 0.0) {
            continue;
        }

        const double alpha = (tg - t0) / dt;
        const vec3 a_interp = LerpVec3(src->acc[k], src->acc[k + 1], alpha);

        new_gyr.push_back(src->gyr[i]);
        new_gyr_tsms.push_back(tg);

        new_acc.push_back(a_interp);
        new_acc_tsms.push_back(tg);
    }

    if (new_gyr.size() < 2 || new_acc.size() < 2) {
        return false;
    }

    src->gyr = std::move(new_gyr);
    src->gyr_tsms = std::move(new_gyr_tsms);

    src->acc = std::move(new_acc);
    src->acc_tsms = std::move(new_acc_tsms);

    RecomputeDtms(src->gyr_tsms, &src->gyr_dtms);
    RecomputeDtms(src->acc_tsms, &src->acc_dtms);

    return true;
}

// TBD[170426-220843](40):I dont think i need this (make sure times are progresive)
void MakeSureTimestamps(std::deque<double>& ts, std::deque<vec3>& data) {
    if (ts.size() != data.size()) {
        throw std::runtime_error("ts y data deben tener el mismo tamano");
    }

    std::vector<size_t> idx(ts.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
        [&ts](size_t a, size_t b) {
            return ts[a] < ts[b];
        });

    std::deque<double> ts_sorted;
    std::deque<vec3> data_sorted;

    ts_sorted.resize(ts.size());
    data_sorted.resize(data.size());

    for (size_t i = 0; i < idx.size(); ++i) {
        ts_sorted[i] = ts[idx[i]];
        data_sorted[i] = data[idx[i]];
    }

    ts = std::move(ts_sorted);
    data = std::move(data_sorted);
}
