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

    Pose last_pose;
    int last_pose_imid;
    double last_pose_tsms;

    int im_id;
};
vioEstimator vio_est;
Config vio_config;
cv::Mat imOutDebug;


void vioInit(Config config){
    vio_config = config;

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

    Logger(INFO, "vioInit: creating Tracker");
    vio_est.ptracker = new Tracker(config);
    Logger(INFO, "vioInit: Tracker created");

    Logger(INFO, "vioInit: creating Updater");
    vio_est.pupdater = new Updater(config);
    Logger(INFO, "vioInit: Updater created");

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

    state->Localx.setZero(27);

    if (config.vio.en_align){
        vec3 zv = g;

        vec3 ex = vec3(1,0,0);
        vec3 xv = ex-zv*zv.transpose()*ex;
        xv.normalize();

        vec3 yv = skewMat(zv)*xv;
        yv.normalize();

        mat3 R;
        R << xv, yv, zv;

        state->Localx.head(4) = RotToQuat(R);
    }else{
        state->Localx.head(4) << 0, 0, 0, 1;
    }

    state->Localx.segment(7,3) = g;
    state->Localx.segment(10,4) = RotToQuat(vio_est.Rci);
    state->Localx.segment(14,3) = vio_est.tci;
    state->Localx(17) = 0.0;
    state->Localx.tail(6) << bg, ba;

    

    const bool use_known_calib = !config.imu.T_ci.empty();

    state->LocalFactor.setZero(25,26);
    state->LocalFactor(0,0) = 1./1e-6; // qG
    state->LocalFactor(1,1) = 1./1e-6;
    state->LocalFactor(2,2) = 1./1e-6;
    state->LocalFactor(3,3) = 1./1e-6; // pG
    state->LocalFactor(4,4) = 1./1e-6;
    state->LocalFactor(5,5) = 1./1e-6;
    state->LocalFactor(6,6) = 1./sqrt(Dt)/config.imu.allanaN(0); // g
    state->LocalFactor(7,7) = 1./sqrt(Dt)/config.imu.allanaN(1);
    state->LocalFactor(8,8) = 1./sqrt(Dt)/config.imu.allanaN(2);

    state->LocalFactor(9,9) = use_known_calib ? 1./2e-2 : 1./2e-1; // qci
    state->LocalFactor(10,10) = use_known_calib ? 1./2e-2 : 1./2e-1;
    state->LocalFactor(11,11) = use_known_calib ? 1./2e-2 : 1./2e-1;
    state->LocalFactor(12,12) = use_known_calib ? 1./1e-2 : 1./1e-1; // pci
    state->LocalFactor(13,13) = use_known_calib ? 1./1e-2 : 1./1e-1;
    state->LocalFactor(14,14) = use_known_calib ? 1./1e-2 : 1./1e-1;
    state->LocalFactor(15,15) = use_known_calib ? 10.0 * config.imu.fps : config.cam.fps; // td

    state->LocalFactor(16,16) = 1./1e-3; // v
    state->LocalFactor(17,17) = 1./1e-3;
    state->LocalFactor(18,18) = 1./1e-3;
    state->LocalFactor(19,19) = 1./sqrt(Dt)/config.imu.allangK(0); // bg
    state->LocalFactor(20,20) = 1./sqrt(Dt)/config.imu.allangK(1);
    state->LocalFactor(21,21) = 1./sqrt(Dt)/config.imu.allangK(2);
    state->LocalFactor(22,22) = 1./sqrt(Dt)/config.imu.allanaK(0); // ba
    state->LocalFactor(23,23) = 1./sqrt(Dt)/config.imu.allanaK(1);
    state->LocalFactor(24,24) = 1./sqrt(Dt)/config.imu.allanaK(2);

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


bool vioUpdate(SourceIn * source, StateOut * state) {
    if(source->frame.empty() || source->imu.empty() || state == nullptr) return false;

    for(int i = 0; i < source->imu.size(); ++i){
        source->imu[i].ts *= 1e-3;
        source->imu[i].dt *= 1e-3;
    }

    source->frame_tsms *= 1e-3;
    source->frame_dtms *= 1e-3;

    if(!vio_est.inited){
        if(!vioInitWhileSteady(vio_config, source, state)) return false;
    }

    vio_est.im_id++;

    vio_est.pprogrator->propagate(vio_est.im_id, source, state);

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
    vec3 w = source->imu.back().vgyr-state->Localx.tail(6).head(3);
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


    

    const ImuSample& imu_last = source->imu.back();
    const vec3 bg = state->Localx.tail(6).head(3);
    const vec3 ba = state->Localx.tail(3);
    const vec3 w_body = imu_last.vgyr - bg;
    const vec3 a_body = imu_last.vacc - ba;
    const vec3 v_body = state->Localx.tail(9).head(3);
    const vec3 v_world = QuatToRot(QuatInv(qkG)) * v_body;

    state->state.pose.pos = vio_est.last_pose.pos;
    state->state.pose.rot = vio_est.last_pose.rot;
    state->ts_ms = vio_est.last_pose_tsms;
    state->state.dpose.ts = imu_last.ts;
    state->state.dpose.dt = imu_last.dt;
    state->state.dpose.vgyr = w_body;
    state->state.dpose.vacc = a_body;
    state->state.vel = v_world;

    state->dt = source->frame_dtms;

    state->deb.rawimu = imu_last;
    state->deb.corimu.ts = imu_last.ts;
    state->deb.corimu.dt = imu_last.dt;
    state->deb.corimu.vgyr = w_body;
    state->deb.corimu.vacc = a_body;


    return true;

}


cv::Mat getDebugImage(){
    return imOutDebug.clone();
}
