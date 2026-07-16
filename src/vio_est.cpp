#include "vio_est.hpp"

#include "lie_math.hpp"

#include "tracker.hpp"
#include "propagator.hpp"
#include "updater.hpp"

struct vioEstimator{
    Tracker * ptracker = nullptr;
    Updater * pupdater = nullptr;
    Propagator * pprogrator = nullptr;

    std::deque<vec3> mqLocalw;
    std::deque<vec3> mqLocalv;
    std::unordered_map<int,Feature*> mmFeatures;
    std::vector<int> mvActiveFeatureIDs;

    bool inited;
    mat3 Rci;
    vec3 tci;
    mat3 visual_RcG = mat3::Identity();
    vec3 visual_tcG = vec3::Zero();

    Pose last_pose;
    int last_pose_imid;
    double last_pose_tsms;

    int im_id;
    bool imu_only_mode = false;
    bool visual_only_mode = false;
};
vioEstimator vio_est;
Config vio_config;
cv::Mat imOutDebug;


void vioInit(Config config){
    vio_config = config;
    vio_est = vioEstimator{};
    vio_est.imu_only_mode = config.gen.type == SOURCE_CSV;
    vio_est.visual_only_mode = !config.gen.imu_on && config.gen.color_on;

    Logger(INFO,
           "vioInit: cam=[%d x %d] fps=%.3f is_rgb=%d feat_max=%d track=[%d,%d] imu_fps=%.3f",
           config.cam.width,
           config.cam.height,
           config.cam.fps,
           config.cam.is_rgb ? 1 : 0,
           config.vio.feat_max,
           config.vio.track_minlength,
           config.vio.track_maxlength,
           config.imu.fps);

    if (!vio_est.imu_only_mode) {
        Logger(INFO, "vioInit: creating Tracker");
        vio_est.ptracker = new Tracker(config);
        Logger(INFO, "vioInit: Tracker created");

        Logger(INFO, "vioInit: creating Updater");
        vio_est.pupdater = new Updater(config);
        Logger(INFO, "vioInit: Updater created");
    } else {
        Logger(INFO, "vioInit: SOURCE_CSV detected, visual frontend disabled");
    }

    Logger(INFO, "vioInit: creating Propagator");
    vio_est.pprogrator = new Propagator(config);
    Logger(INFO, "vioInit: Propagator created");
    

    cv::Mat T_ci = config.imu.T_ci;
    if (T_ci.empty()) {
        Logger(WARN, "vioInit: imu.tocolor is empty, using identity extrinsic");
        T_ci = cv::Mat::eye(4, 4, CV_64F);
    }

    cv::Mat T = InvertRigidTransform(T_ci);  // convert to cam to imu
    if (T.empty()) {
        Logger(WARN, "vioInit: invalid imu.tocolor transform, using identity extrinsic");
        T = cv::Mat::eye(4, 4, CV_64F);
    }
    T.convertTo(T, CV_64F);

    Eigen::Matrix4d Tic;
    cv::cv2eigen(T,Tic);
    mat3 Ric = Tic.topLeftCorner(3,3);
    vec3 tic = Tic.topRightCorner(3,1);
    vio_est.Rci = Ric.transpose();
    vio_est.tci = -vio_est.Rci*tic;


    vio_est.inited = false;
    vio_est.im_id = 0;
    vio_est.last_pose = Pose{};
    vio_est.last_pose_imid = 0;
    vio_est.last_pose_tsms = 0.0;

    Logger(INFO, "vioInit: completed successfully");
    

}

void vioClose(){
    for (auto& it : vio_est.mmFeatures)
        delete it.second;

    delete vio_est.ptracker;
    delete vio_est.pupdater;
    delete vio_est.pprogrator;

}

static void setLatestPose(int nImageId, double timestamp, StateOut * state) {
    vio_est.last_pose_imid = nImageId;
    vio_est.last_pose_tsms = timestamp;
    vec4 rota = state->Localx.head(4);
    vio_est.last_pose.rot = normalizeQ(quat(rota(3), rota(0), rota(1), rota(2)));
    vio_est.last_pose.pos = -QuatToRot(QuatInv(rota))*state->Localx.segment(4,3);
}

static Pose poseFromGlobalState(const vec4& qG, const vec3& tG) {
    Pose pose;
    pose.rot = normalizeQ(quat(qG(3), qG(0), qG(1), qG(2)));
    pose.pos = -QuatToRot(QuatInv(qG))*tG;
    return pose;
}

static void setupInitialLocalState(Config config, const vec3& g, const vec3& bg, const vec3& ba, double Dt, StateOut * state) {
    state->Localx.setZero(27);

    if (config.vio.en_align) {
        vec3 zv = g;

        vec3 ex = vec3(1,0,0);
        vec3 xv = ex-zv*zv.transpose()*ex;
        xv.normalize();

        vec3 yv = skewMat(zv)*xv;
        yv.normalize();

        mat3 R;
        R << xv, yv, zv;

        state->Localx.head(4) = RotToQuat(R);
    } else {
        state->Localx.head(4) << 0, 0, 0, 1;
    }

    state->Localx.segment(7,3) = g;
    state->Localx.segment(10,4) = RotToQuat(vio_est.Rci);
    state->Localx.segment(14,3) = vio_est.tci;
    state->Localx(17) = 0.0;
    state->Localx.tail(6) << bg, ba;

    const bool use_known_calib = !config.imu.T_ci.empty();
    const double safe_dt = std::max(Dt, 1.0 / std::max(1.0, config.imu.fps));

    state->LocalFactor.setZero(25,26);
    state->LocalFactor(0,0) = 1./1e-6;
    state->LocalFactor(1,1) = 1./1e-6;
    state->LocalFactor(2,2) = 1./1e-6;
    state->LocalFactor(3,3) = 1./1e-6;
    state->LocalFactor(4,4) = 1./1e-6;
    state->LocalFactor(5,5) = 1./1e-6;
    state->LocalFactor(6,6) = 1./sqrt(safe_dt)/config.imu.allanaN(0);
    state->LocalFactor(7,7) = 1./sqrt(safe_dt)/config.imu.allanaN(1);
    state->LocalFactor(8,8) = 1./sqrt(safe_dt)/config.imu.allanaN(2);

    state->LocalFactor(9,9) = use_known_calib ? 1./2e-2 : 1./2e-1;
    state->LocalFactor(10,10) = use_known_calib ? 1./2e-2 : 1./2e-1;
    state->LocalFactor(11,11) = use_known_calib ? 1./2e-2 : 1./2e-1;
    state->LocalFactor(12,12) = use_known_calib ? 1./1e-2 : 1./1e-1;
    state->LocalFactor(13,13) = use_known_calib ? 1./1e-2 : 1./1e-1;
    state->LocalFactor(14,14) = use_known_calib ? 1./1e-2 : 1./1e-1;
    state->LocalFactor(15,15) = use_known_calib ? 10.0 * config.imu.fps : std::max(config.cam.fps, config.imu.fps);

    state->LocalFactor(16,16) = 1./1e-3;
    state->LocalFactor(17,17) = 1./1e-3;
    state->LocalFactor(18,18) = 1./1e-3;
    state->LocalFactor(19,19) = 1./sqrt(safe_dt)/config.imu.allangK(0);
    state->LocalFactor(20,20) = 1./sqrt(safe_dt)/config.imu.allangK(1);
    state->LocalFactor(21,21) = 1./sqrt(safe_dt)/config.imu.allangK(2);
    state->LocalFactor(22,22) = 1./sqrt(safe_dt)/config.imu.allanaK(0);
    state->LocalFactor(23,23) = 1./sqrt(safe_dt)/config.imu.allanaK(1);
    state->LocalFactor(24,24) = 1./sqrt(safe_dt)/config.imu.allanaK(2);
}

static bool vioInitWhileSteady(Config config, SourceIn * source, StateOut * state){
    static int nImuCount = 0;

    static vec3 wm = vec3::Zero();
    static vec3 am = vec3::Zero();
    static double Dt = 0;

    static vec3 wm_last;

    static cv::Mat im_last;
    static double im_last_timestamp;

    vec3 ang = vec3::Zero();
    vec3 vel = vec3::Zero();
    vec3 len = vec3::Zero();

    for (const ImuSample& data : source->imu) {
        vec3 tempw = data.vgyr;
        vec3 tempa = data.vacc;
        double dt = data.dt;

        tempa -= config.imu.g*(tempa/tempa.norm());

        ang += dt*tempw;
        vel += dt*tempa;
        len += dt*vel+.5*pow(dt,2)*tempa;
    }

    // Not move yet
    if (ang.norm()*180./M_PI<config.vio.ang_ths && len.norm()<config.vio.dis_ths){
        for (const ImuSample& data : source->imu){
            wm += data.vgyr;
            am += data.vacc;
            Dt += data.dt;

            nImuCount++;
        }

        wm_last = source->imu.front().vgyr;

        im_last = source->frame;
        im_last_timestamp = source->frame_tsms;

        return false;
    }

    if (nImuCount==0){
        // Start in motion
        wm = source->imu.back().vgyr;
        am = source->imu.back().vacc;
        nImuCount = 1;

        im_last = source->frame;
        im_last_timestamp = source->frame_tsms;


        return false;
    }

    vec3 g, bg, ba;
    bg.setZero();
    ba.setZero();

    if (nImuCount==1){
        g = am;
        g.normalize();
    }else{
        wm /= nImuCount;
        am /= nImuCount;

        g = am;
        g.normalize();

        bg = wm;
        ba = am-config.imu.g*g;
    }

    setupInitialLocalState(config, g, bg, ba, Dt, state);
    const bool use_known_calib = !config.imu.T_ci.empty();

    vec3 v = state->Localx.segment(18,3);
    vec3 w = wm_last-state->Localx.segment(21,3);

    vio_est.mqLocalv.push_back(v);
    vio_est.mqLocalw.push_back(w);

    vio_est.inited = true;

    Logger(INFO,
           "vioInitWhileSteady: use_known_calib=%d sigma_px=%.9f sigma_py=%.9f td_prior=%.3f",
           use_known_calib ? 1 : 0,
           config.cam.spx,
           config.cam.spy,
           state->LocalFactor(15,15));



    // Start tracker
    mat3 RcG = vio_est.Rci*QuatToRot(state->Localx.head(4));
    vec3 tcG = vio_est.Rci*state->Localx.segment(4,3)+vio_est.tci;
    vio_est.ptracker->track(0, im_last, RcG, tcG, 0, vio_est.mmFeatures, imOutDebug);

    setLatestPose(0, im_last_timestamp, state);

    return true;
}

static bool vioInitImuOnly(Config config, SourceIn * source, StateOut * state) {
    if (source == nullptr || state == nullptr || source->imu.empty()) return false;

    vec3 am = vec3::Zero();
    vec3 wm = vec3::Zero();
    double Dt = 0.0;

    for (const ImuSample& data : source->imu) {
        am += data.vacc;
        wm += data.vgyr;
        Dt += data.dt;
    }

    const double n = static_cast<double>(source->imu.size());
    am /= std::max(1.0, n);
    wm /= std::max(1.0, n);

    vec3 g = am - config.imu.ba;
    if (g.norm() < 1e-9) g = am;
    if (g.norm() < 1e-9) g = vec3(0.0, -1.0, 0.0);
    else g.normalize();

    setupInitialLocalState(config, g, config.imu.bg, config.imu.ba, Dt, state);

    vio_est.mqLocalv.clear();
    vio_est.mqLocalw.clear();
    vio_est.mqLocalv.push_back(state->Localx.segment(18,3));
    vio_est.mqLocalw.push_back(wm - config.imu.bg);

    vio_est.last_pose = Pose{};
    vio_est.last_pose_imid = 0;
    vio_est.last_pose_tsms = source->frame_tsms;
    vio_est.im_id = 0;
    vio_est.inited = true;

    Logger(INFO,
           "vioInitImuOnly: g_dir=[%.6f %.6f %.6f] bg=[%.6f %.6f %.6f] ba=[%.6f %.6f %.6f]",
           g.x(), g.y(), g.z(),
           config.imu.bg.x(), config.imu.bg.y(), config.imu.bg.z(),
           config.imu.ba.x(), config.imu.ba.y(), config.imu.ba.z());
    return true;
}

static bool vioInitVisualOnly(Config config, SourceIn * source, StateOut * state) {
    (void)config;
    (void)state;
    if (source == nullptr || source->frame.empty() || vio_est.ptracker == nullptr) return false;

    vio_est.visual_RcG.setIdentity();
    vio_est.visual_tcG.setZero();
    vio_est.ptracker->track(0, source->frame, vio_est.visual_RcG, vio_est.visual_tcG, 0, vio_est.mmFeatures, imOutDebug);

    vio_est.last_pose = Pose{};
    vio_est.last_pose_imid = 0;
    vio_est.last_pose_tsms = source->frame_tsms;
    vio_est.im_id = 0;
    vio_est.inited = true;

    Logger(INFO, "vioInitVisualOnly: tracker primed for monocular visual-only mode");
    return true;
}

static void fillImuDebugCommon(SourceIn * source, StateOut * state) {
    const ImuSample& imu_last = source->imu.back();
    const vec3 bg = state->Localx.tail(6).head(3);
    const vec3 ba = state->Localx.tail(3);
    const vec3 w_body = imu_last.vgyr - bg;
    const vec3 a_body = imu_last.vacc - ba;
    state->state.dpose.ts = imu_last.ts;
    state->state.dpose.dt = imu_last.dt;
    state->state.dpose.vgyr = w_body;
    state->state.dpose.vacc = a_body;
    state->dt = source->frame_dtms;

    state->deb.imu_stat = true;
    state->deb.vio_valid = !vio_est.imu_only_mode;
    state->deb.vio_inl = 0;
    state->deb.rawimu = imu_last;
    state->deb.corimu.ts = imu_last.ts;
    state->deb.corimu.dt = imu_last.dt;
    state->deb.corimu.vgyr = w_body;
    state->deb.corimu.vacc = a_body;
}

static void fillVisualOnlyDebug(StateOut * state, double dt) {
    state->dt = dt;
    state->deb.imu_stat = false;
    state->deb.vio_valid = true;
    state->deb.vio_inl = 0;
    state->deb.imu = Pose{};
    state->deb.preimu.setZero();
    state->deb.rawimu = ImuSample{};
    state->deb.corimu = ImuSample{};
    state->state.dpose = ImuSample{};
}

bool vioUpdate(SourceIn * source, StateOut * state) {
    if(source == nullptr || state == nullptr) return false;
    if(!vio_est.visual_only_mode && source->imu.empty()) return false;

    for(int i = 0; i < source->imu.size(); ++i){
        source->imu[i].ts *= 1e-3;
        source->imu[i].dt *= 1e-3;
    }

    source->frame_tsms *= 1e-3;
    source->frame_dtms *= 1e-3;

    if(!vio_est.inited){
        if (vio_est.imu_only_mode) {
            if (!vioInitImuOnly(vio_config, source, state)) return false;
        } else if (vio_est.visual_only_mode) {
            if (!vioInitVisualOnly(vio_config, source, state)) return false;
            return false;
        } else {
            if(source->frame.empty()) return false;
            if(!vioInitWhileSteady(vio_config, source, state)) return false;
        }
    }

    if (vio_est.visual_only_mode) {
        if (source->frame.empty() || vio_est.ptracker == nullptr) return false;

        vio_est.im_id++;
        vio_est.ptracker->track(vio_est.im_id, source->frame, vio_est.visual_RcG, vio_est.visual_tcG, 0, vio_est.mmFeatures, imOutDebug);

        mat3 R_rel = mat3::Identity();
        vec3 t_rel = vec3::Zero();
        int inliers = 0;
        double mean_parallax = 0.0;
        if (vio_est.ptracker->getVisualOnlyMotion(R_rel, t_rel, inliers, mean_parallax)) {
            vio_est.visual_RcG = R_rel * vio_est.visual_RcG;
            vio_est.visual_tcG = R_rel * vio_est.visual_tcG + t_rel;
            vio_est.last_pose = poseFromGlobalState(RotToQuat(vio_est.visual_RcG), vio_est.visual_tcG);
        }

        vio_est.last_pose_imid = vio_est.im_id;
        vio_est.last_pose_tsms = source->frame_tsms;

        state->deb.vis = vio_est.last_pose;
        state->state.pose = vio_est.last_pose;
        state->state.vel = vec3::Zero();
        state->ts_ms = source->frame_tsms;
        fillVisualOnlyDebug(state, source->frame_dtms);
        state->deb.vio_inl = static_cast<uint64_t>(inliers);
        state->deb.vio_valid = true;
        return true;
    }

    vio_est.im_id++;

    vio_est.pprogrator->propagate(vio_est.im_id, source, state);

    if (vio_est.imu_only_mode) {
        const quat q_inc = normalizeQ(quat(state->x(3), state->x(0), state->x(1), state->x(2)));
        const vec3 p_inc = state->x.segment(4,3);
        const mat3 R_prev = vio_est.last_pose.rot.toRotationMatrix();

        Pose pose = vio_est.last_pose;
        pose.pos += R_prev * p_inc;
        pose.rot = normalizeQ(vio_est.last_pose.rot * q_inc);

        vio_est.last_pose = pose;
        vio_est.last_pose_imid = vio_est.im_id;
        vio_est.last_pose_tsms = source->frame_tsms;
        state->deb.imu = pose;
        state->state.pose = pose;
        state->ts_ms = source->frame_tsms;
        state->state.vel = pose.rot.toRotationMatrix() * state->x.segment(7,3);
        fillImuDebugCommon(source, state);
        return true;
    }

    // Predict camera pose
    Eigen::VectorXd xk = state->Localx.tail(16);
    mat3 Rk = QuatToRot(xk.head(4));
    vec3 tk = xk.segment(4,3);
    mat3 RkG = Rk*QuatToRot(state->Localx.head(4));
    vec3 tkG = Rk*(state->Localx.segment(4,3)-tk);

    mat3 RcG = vio_est.Rci*RkG;
    vec3 tcG = vio_est.Rci*tkG+vio_est.tci;

    int nMapPtsNeeded = vio_config.vio.slam_pts - vio_est.mvActiveFeatureIDs.size();

    vio_est.ptracker->track(vio_est.im_id, source->frame, RcG, tcG, nMapPtsNeeded, vio_est.mmFeatures, imOutDebug);

    // Save local velocities
    vec3 w = vec3::Zero();
    if (!source->imu.empty()) {
        w = source->imu.back().vgyr-state->Localx.tail(6).head(3);
    }
    vec3 v = state->Localx.tail(9).head(3);

    vio_est.mqLocalw.push_back(w);
    vio_est.mqLocalv.push_back(v);
    if (vio_est.im_id > (vio_config.vio.track_maxlength-1))
    {
        vio_est.mqLocalw.pop_front();
        vio_est.mqLocalv.pop_front();
    }

    vio_est.pupdater->update(vio_est.im_id, vio_est.mmFeatures, vio_est.ptracker->mvFeatMeasForExploration, vio_est.ptracker->mvFeatInfoForInitSlam, vio_est.ptracker->mvvFeatMeasForInitSlam, 
                    vio_est.ptracker->mvFeatInfoForPoseOnly, vio_est.ptracker->mvvFeatMeasForPoseOnly, vio_est.mvActiveFeatureIDs, vio_est.mqLocalw, vio_est.mqLocalv, state->Localx, state->LocalFactor);

    vec4 qkG = state->Localx.head(4);
    vec3 pGk = -QuatToRot(QuatInv(qkG))*state->Localx.segment(4,3);
    (void)pGk;

    state->deb.vis = poseFromGlobalState(qkG, state->Localx.segment(4,3));


    setLatestPose(vio_est.im_id, source->frame_tsms, state);

    vio_est.Rci = QuatToRot(state->Localx.segment(10,4));
    vio_est.tci = state->Localx.segment(14,3);


    

    state->state.pose.pos = vio_est.last_pose.pos;
    state->state.pose.rot = vio_est.last_pose.rot;
    state->ts_ms = vio_est.last_pose_tsms;

    if (vio_est.visual_only_mode) {
        state->state.vel = vec3::Zero();
        fillVisualOnlyDebug(state, source->frame_dtms);
    } else {
        const ImuSample& imu_last = source->imu.back();
        const vec3 bg = state->Localx.tail(6).head(3);
        const vec3 ba = state->Localx.tail(3);
        const vec3 w_body = imu_last.vgyr - bg;
        const vec3 a_body = imu_last.vacc - ba;
        const vec3 v_body = state->Localx.tail(9).head(3);
        const vec3 v_world = QuatToRot(QuatInv(qkG)) * v_body;

        state->state.dpose.ts = imu_last.ts;
        state->state.dpose.dt = imu_last.dt;
        state->state.dpose.vgyr = w_body;
        state->state.dpose.vacc = a_body;
        state->state.vel = v_world;

        state->dt = source->frame_dtms;

        state->deb.imu_stat = true;
        state->deb.vio_valid = true;
        state->deb.rawimu = imu_last;
        state->deb.corimu.ts = imu_last.ts;
        state->deb.corimu.dt = imu_last.dt;
        state->deb.corimu.vgyr = w_body;
        state->deb.corimu.vacc = a_body;
    }


    return true;

}


cv::Mat getDebugImage(){
    return imOutDebug.clone();
}
