#include <cstdlib>
#include <iostream>
#include <numeric>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "tracker.hpp"

int featId = 0;

const cv::Scalar red(64,64,255);
const cv::Scalar blue(255,64,64);
const cv::Scalar green(64,255,64);


Tracker::Tracker(Config config)
{
    mbIsRGB             = config.cam.is_rgb;
    mbEnableFilter      = config.vio.track_filter_on;
    mbEnableEqualizer   = config.vio.track_en_equalizer;
    mnMaxFeatsPerImage  = config.vio.feat_max;
    mnMaxTrackingLength = config.vio.track_maxlength;
    mnMinTrackingLength = config.vio.track_minlength;
    mbEnableSlam        = config.vio.slam_pts > 0 ? true : false;
    mbVisualOnlyMode    = !config.gen.imu_on && config.gen.color_on;
    mnGoodParallax      = config.vio.good_para;
    mnVisualRansacThreshold = std::max(1e-3, 3.0 * std::max(config.cam.spx, config.cam.spy));
    mnVisualMinParallax = std::max(1e-4, 2.0 * std::max(config.cam.spx, config.cam.spy));

    mLastImage = cv::Mat();

    config.cam.K.copyTo(mK);
    config.cam.D.copyTo(mD);


    mbRestartVT = false;
    mbRefreshVT = false;

    mpRansac = new Ransac(config);
    mpFeatureDetector = new FeatureDetector(config);

    mbShowTrack = config.vio.track_show;
    mbShowNewer = config.vio.track_shownew;
}


Tracker::~Tracker()
{
    delete mpRansac;
    delete mpFeatureDetector;
}

bool Tracker::getVisualOnlyMotion(mat3& R_rel, vec3& t_rel, int& inliers, double& mean_parallax) const {
    R_rel = mLastVisualRelR;
    t_rel = mLastVisualRelT;
    inliers = mnLastVisualInliers;
    mean_parallax = mnLastVisualMeanParallax;
    return mbHasVisualMotion;
}


void Tracker::preprocess(const int nImageId, cv::Mat& image, const mat3& RcG, const vec3& tcG){
    // Convert to grayscale
    if (image.channels()==3)
    {
        if (mbIsRGB)
            cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        else
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels()==4)
    {
        if (mbIsRGB)
            cv::cvtColor(image, image, cv::COLOR_RGBA2GRAY);
        else
            cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
    }

    if (mbEnableFilter)
    {
        cv::GaussianBlur(image, image, cv::Size(5,5), 0);
        cv::adaptiveThreshold(image, image, 225, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 5, 0);
        cv::boxFilter(image, image, image.depth(), cv::Size(5,5));
    }

    if (mbEnableEqualizer)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(5,5));
        clahe->apply(image, image);
    }

    if ((int)mlCamOrientations.size()+1>mnMaxTrackingLength)
    {
        mlCamOrientations.pop_front();
        mlCamPositions.pop_front();
    }

    if (nImageId>0)
    {
        mRx = RcG*(mlCamOrientations.front().transpose());
        mRr = RcG*(mlCamOrientations.back().transpose());
    }

    mlCamOrientations.push_back(RcG);
    mlCamPositions.push_back(tcG);
}


void Tracker::undistort(const std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst){
    int N = src.size();

    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; ++i){
        mat.at<float>(i,0) = src.at(i).x;
        mat.at<float>(i,1) = src.at(i).y;
    }

    mat = mat.reshape(2);

    cv::undistortPoints(mat, mat, mK, mD);

    mat = mat.reshape(1);

    dst.resize(N);
    for(int i=0; i<N; ++i){
        dst.at(i).x = mat.at<float>(i,0);
        dst.at(i).y = mat.at<float>(i,1);
    }
}

bool Tracker::EstimateVisualOnlyMotion(const std::vector<cv::Point2f>& vCurrFeatUN, std::vector<unsigned char>& vInlierFlags, int& nInliers) {
    mbHasVisualMotion = false;
    mnLastVisualInliers = 0;
    mnLastVisualMeanParallax = 0.0;
    mLastVisualRelR.setIdentity();
    mLastVisualRelT.setZero();

    std::vector<cv::Point2f> vPrevNorm;
    std::vector<cv::Point2f> vCurrNorm;
    std::vector<int> vIndices;

    const int N = std::min({static_cast<int>(vInlierFlags.size()), static_cast<int>(vCurrFeatUN.size()), static_cast<int>(PointsForRansac.cols())});
    vPrevNorm.reserve(N);
    vCurrNorm.reserve(N);
    vIndices.reserve(N);

    for (int i = 0; i < N; ++i) {
        if (!vInlierFlags.at(i)) continue;

        const vec3 ray_prev = PointsForRansac.col(i);
        if (std::abs(ray_prev.z()) < 1e-9) continue;

        vPrevNorm.emplace_back(static_cast<float>(ray_prev.x() / ray_prev.z()),
                               static_cast<float>(ray_prev.y() / ray_prev.z()));
        vCurrNorm.emplace_back(vCurrFeatUN.at(i));
        vIndices.push_back(i);
    }

    if (vPrevNorm.size() < 8) {
        nInliers = static_cast<int>(vPrevNorm.size());
        return false;
    }

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(vPrevNorm, vCurrNorm, 1.0, cv::Point2d(0.0, 0.0), cv::RANSAC, 0.999, mnVisualRansacThreshold, mask);
    if (E.empty()) {
        nInliers = static_cast<int>(vPrevNorm.size());
        return false;
    }

    cv::Mat R_cv;
    cv::Mat t_cv;
    const int pose_inliers = cv::recoverPose(E, vPrevNorm, vCurrNorm, R_cv, t_cv, 1.0, cv::Point2d(0.0, 0.0), mask);
    if (pose_inliers <= 0) {
        nInliers = static_cast<int>(vPrevNorm.size());
        return false;
    }

    std::fill(vInlierFlags.begin(), vInlierFlags.end(), 0);

    double parallax_sum = 0.0;
    int accepted = 0;
    for (int j = 0; j < static_cast<int>(vIndices.size()); ++j) {
        const bool inlier = mask.at<unsigned char>(j) != 0;
        if (!inlier) continue;

        const int idx = vIndices.at(j);
        vInlierFlags.at(idx) = 1;
        parallax_sum += cv::norm(vCurrNorm.at(j) - vPrevNorm.at(j));
        accepted++;
    }

    nInliers = accepted;
    mnLastVisualInliers = accepted;
    mnLastVisualMeanParallax = accepted > 0 ? parallax_sum / static_cast<double>(accepted) : 0.0;

    if (accepted < 5) {
        return false;
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            mLastVisualRelR(r, c) = R_cv.at<double>(r, c);
        }
    }
    mLastVisualRelT = vec3(t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2));

    const double t_norm = mLastVisualRelT.norm();
    if (t_norm > 1e-9) {
        mLastVisualRelT /= t_norm;
    }

    if (mnLastVisualMeanParallax < mnVisualMinParallax) {
        mLastVisualRelT.setZero();
    }

    mbHasVisualMotion = true;
    return true;
}


void Tracker::DisplayTrack(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vInlierFlags, cv::Mat& imOut){
    cv::cvtColor(image, imOut, cv::COLOR_GRAY2BGR);

    for (int i=0; i<(int)vPrevFeatUVs.size(); ++i){
        if (vInlierFlags.at(i)){
            cv::circle(imOut, vPrevFeatUVs.at(i), 3, blue, -1);
            cv::line(imOut, vPrevFeatUVs.at(i), vCurrFeatUVs.at(i), blue);
        }else{
            cv::circle(imOut, vPrevFeatUVs.at(i), 3, red, 0);
        }
    }

    cv::putText(imOut, std::to_string(nImageId), cv::Point2f(15,30), cv::FONT_HERSHEY_PLAIN, 2, green, 2);
}


void Tracker::DisplayNewer(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vRefFeatUVs, const std::vector<cv::Point2f>& vNewFeatUVs, cv::Mat& imOut){
    cv::cvtColor(image, imOut, cv::COLOR_GRAY2BGR);

    for (const cv::Point2f& pt : vRefFeatUVs)
        cv::circle(imOut, pt, 3, blue, 0);

    for (const cv::Point2f& pt : vNewFeatUVs)
        cv::circle(imOut, pt, 3, green, -1);

    cv::putText(imOut, std::to_string(nImageId), cv::Point2f(15,30), cv::FONT_HERSHEY_PLAIN, 2, green, 2);
}


void Tracker::VisualTracking(const int nImageId, const cv::Mat& image, int nMapPtsNeeded, std::unordered_map<int,Feature*>& mFeatures, cv::Mat &imOut){
    mbHasVisualMotion = false;
    mnLastVisualInliers = 0;
    mnLastVisualMeanParallax = 0.0;
    mLastVisualRelR.setIdentity();
    mLastVisualRelT.setZero();

    std::vector<cv::Point2f> vFeatPts, vFeatPtsUN;
    std::vector<unsigned char> vInlierFlags;
    std::vector<float> vErrors;

    cv::Size winSize(15,15);
    cv::TermCriteria termCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-2);

    // LK method
    cv::calcOpticalFlowPyrLK(mLastImage, image, mvFeatPtsToTrack, vFeatPts, vInlierFlags, vErrors, 
                             winSize, 3, termCriteria, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 1e-4);

    int nFeats = vFeatPts.size();
    int nInliers = 0;

    const int lkInliers = std::accumulate(vInlierFlags.begin(), vInlierFlags.end(), 0);
    if (lkInliers > 0)
    {
        undistort(vFeatPts, vFeatPtsUN);
        Eigen::MatrixXd MatchesForRansac(3,nFeats);

        for (int i=0; i<nFeats; ++i)
        {
            cv::Point2f pt = vFeatPtsUN.at(i);
            MatchesForRansac.col(i) << pt.x, pt.y, 1;
            MatchesForRansac.col(i).normalize();
        }

        if (mbVisualOnlyMode) {
            if (!EstimateVisualOnlyMotion(vFeatPtsUN, vInlierFlags, nInliers)) {
                nInliers = lkInliers;
            }
        } else {
            mpRansac->FindInliers(PointsForRansac, MatchesForRansac, mRr, nInliers, vInlierFlags);
        }

        if (nInliers==0)
        {
            const double trace = std::max(-1.0, std::min(3.0, static_cast<double>(mRr.trace())));
            const double rel_angle_deg = std::acos(std::max(-1.0, std::min(1.0, 0.5 * (trace - 1.0)))) * 180.0 / M_PI;
            Logger(WARN,
                   "Tracker RANSAC rejected all tracks at image %d: tracked=%d lk_inliers=%d active_before=%zu rel_rot_deg=%.3f",
                   nImageId,
                   nFeats,
                   lkInliers,
                   mvFeatIDsToTrack.size(),
                   rel_angle_deg);
            std::cerr << "Visual Tracking: lost all features, refresh if the refill fails!" << "\n";
            mbRefreshVT = true;
        }
    }
    else
    {
        Logger(WARN,
               "Tracker LK lost all tracks at image %d: tracked=%d active_before=%zu",
               nImageId,
               nFeats,
               mvFeatIDsToTrack.size());
        std::cerr << "Visual Tracking: lost all features, refresh anyway!" << "\n";
        mbRefreshVT = true;
    }

    if (mbShowTrack)
    {

        DisplayTrack(nImageId, image, mvFeatPtsToTrack, vFeatPts, vInlierFlags, imOut);
    }

    std::vector<int> vFeatIDs(mvFeatIDsToTrack);
    mvFeatIDsToTrack.clear();
    mvFeatPtsToTrack.clear();

    PointsForRansac.resize(3,nInliers);

    int nFeatCnt = 0;

    for (int i=0; i<nFeats; ++i)
    {
        int id = vFeatIDs.at(i);
        Feature* pFeature = mFeatures.at(id);

        if (vInlierFlags.at(i))
        {
            cv::Point2f pt = vFeatPts.at(i);
            cv::Point2f ptUN = vFeatPtsUN.at(i);

            PointsForRansac.col(nFeatCnt) << ptUN.x, ptUN.y, 1;
            PointsForRansac.col(nFeatCnt).normalize();

            if (!pFeature->IsInited())
            {
                std::vector<cv::Point2f> vTrack;

                vTrack.swap(mmFeatTrackingHistory.at(id));
                vTrack.push_back(ptUN);

                int nTrackingLength = vTrack.size();

                if (nTrackingLength==mnMaxTrackingLength)
                {
                    if (mbEnableSlam)
                    {
                        double parallax = Parallax(vTrack.front(), vTrack.back());

                        if (nMapPtsNeeded>0)
                        {
                            if (parallax>=mnGoodParallax)
                            {
                                mvFeatInfoForInitSlam.emplace_back(id,INIT_SLAM);
                                mvvFeatMeasForInitSlam.emplace_back(vTrack);

                                nMapPtsNeeded--;
                            }
                            else
                            {
                                mvFeatInfoForPoseOnly.emplace_back(id,POSE_ONLY_M);
                                mvvFeatMeasForPoseOnly.emplace_back(vTrack);
                            }
                        }
                        else
                        {
                            if (parallax<mnGoodParallax)
                            {
                                auto vbeg = vTrack.begin()+1;
                                auto vend = vTrack.end();
                                std::vector<cv::Point2f>(vbeg,vend).swap(vTrack);
                                pFeature->reset(pFeature->RootImageId()+1);
                            }
                            else
                            {
                                mvFeatInfoForPoseOnly.emplace_back(id,POSE_ONLY_M);
                                mvvFeatMeasForPoseOnly.emplace_back(vTrack);
                            }
                        }
                    }
                    else
                    {
                        mvFeatInfoForPoseOnly.emplace_back(id,POSE_ONLY_M);
                        mvvFeatMeasForPoseOnly.emplace_back(vTrack);
                    }
                }

                mmFeatTrackingHistory.at(id).swap(vTrack);
            }
            else
            {
                if (!pFeature->IsMarginalized())
                    mvFeatMeasForExploration.emplace_back(id,ptUN);
                else
                    exit(-1);
            }

            mvFeatIDsToTrack.push_back(id);
            mvFeatPtsToTrack.push_back(pt);

            nFeatCnt++;
        }
        else
        {
            if (mmFeatTrackingHistory.count(id))
            {
                int nTrackingLength = mmFeatTrackingHistory.at(id).size();

                if (nTrackingLength>=mnMinTrackingLength)
                {
                    mvFeatInfoForPoseOnly.emplace_back(id,POSE_ONLY);
                    mvvFeatMeasForPoseOnly.emplace_back(mmFeatTrackingHistory.at(id));
                }

                mvFeatIDsLoseTrack.push_back(id);
            }
        }
    }

    if (!mvFeatCandidates.empty())
    {
        std::vector<cv::Point2f> vNewFeatPts;
        int nNewFeats = mpFeatureDetector->FindNewer(mvFeatCandidates, mvFeatPtsToTrack, vNewFeatPts);

        if (nNewFeats>0)
        {
            std::vector<cv::Point2f> vNewFeatPtsUN;
            undistort(vNewFeatPts, vNewFeatPtsUN);

            PointsForRansac.conservativeResize(3,nInliers+nNewFeats);

            for (int i=0; i<nNewFeats; ++i)
            {
                cv::Point2f pt = vNewFeatPts.at(i);
                cv::Point2f ptUN = vNewFeatPtsUN.at(i);

                PointsForRansac.col(nFeatCnt) << ptUN.x, ptUN.y, 1;
                PointsForRansac.col(nFeatCnt).normalize();

                int id = 0;

                if (!mvFeatIDsInactive.empty())
                {
                    id = mvFeatIDsInactive.back();
                    mFeatures.at(id)->reset(nImageId);
                    mvFeatIDsInactive.pop_back();
                }
                else
                {
                    id = featId++;
                    mFeatures[id] = new Feature(id, nImageId);
                }

                mmFeatTrackingHistory[id].reserve(mnMaxTrackingLength);
                mmFeatTrackingHistory[id].push_back(ptUN);

                mvFeatIDsToTrack.push_back(id);
                mvFeatPtsToTrack.push_back(pt);

                nFeatCnt++;
            }

            if (mbShowNewer)
            {
                
                DisplayNewer(nImageId, image, mvFeatPtsToTrack, vNewFeatPts, imOut);
                // mpFeatureDetector->DebugDrawGrid(image, imOut);
                // mpFeatureDetector->DebugDrawShiTomasi(image, imOut);
                // mpFeatureDetector->DebugDrawSubPix(image, 1.5*mnMaxFeatsPerImage, 1, imOut);
                // mpFeatureDetector->DebugDrawCandidates(image, mvFeatCandidates, mvFeatPtsToTrack, vNewFeatPts, imOut);
                // mpFeatureDetector->DebugDrawSelection(image, mvFeatCandidates, mvFeatPtsToTrack, vNewFeatPts, imOut);
                // mpFeatureDetector->DebugDrawCellOccupancy(image, mvFeatPtsToTrack, vNewFeatPts, imOut);
            }

            if (mbRefreshVT)
                mbRefreshVT = false;
        }
    }

    image.copyTo(mLastImage);
}


bool Tracker::start(const int nImageId, 
                    cv::Mat image, 
                    const mat3& RcG, 
                    const vec3& tcG, 
                    std::unordered_map<int,Feature*>& mFeatures)
{
    if (nImageId==0 || mbRestartVT)
    {
        featId = 0;
        for (auto& it : mFeatures)
            delete it.second;
        mFeatures.clear();
    }
    
    mvFeatIDsToTrack.clear();
    mvFeatPtsToTrack.clear();
    mmFeatTrackingHistory.clear();

    preprocess(nImageId, image, RcG, tcG);

    int nFeats = mpFeatureDetector->DetectWithSubPix(image, mnMaxFeatsPerImage, 1, mvFeatPtsToTrack);
    if (nFeats==0)
    {
        std::cerr << "No features available to track!" << "\n";

        if (!mbRestartVT && !mbRefreshVT)
            mbRestartVT = true;

        return false;
    }

    if (mbRestartVT)
        mbRestartVT = false;

    if (mbRefreshVT)
    {
        mlCamOrientations.clear();
        mlCamPositions.clear();

        mvFeatInfoForInitSlam.clear();
        mvvFeatMeasForInitSlam.clear();
        mvFeatInfoForPoseOnly.clear();
        mvvFeatMeasForPoseOnly.clear();
        mvFeatMeasForExploration.clear();

        mbRefreshVT = false;
    }

    std::vector<cv::Point2f> vFeatPtsToTrackUN;
    undistort(mvFeatPtsToTrack, vFeatPtsToTrackUN);

    PointsForRansac.resize(3,nFeats);

    for (int i=0; i<nFeats; ++i)
    {
        cv::Point2f ptUN = vFeatPtsToTrackUN.at(i);

        PointsForRansac.col(i) << ptUN.x, ptUN.y, 1;
        PointsForRansac.col(i).normalize();

        int id = featId++;
        mFeatures[id] = new Feature(id, nImageId);

        mmFeatTrackingHistory[id].reserve(mnMaxTrackingLength);
        mmFeatTrackingHistory[id].push_back(ptUN);

        mvFeatIDsToTrack.push_back(id);
    }

    image.copyTo(mLastImage);

    return true;
}


void Tracker::manage(const int nImageId, 
                     cv::Mat& image, 
                     const mat3& RcG, 
                     const vec3& tcG, 
                     const std::unordered_map<int,Feature*>& mFeatures)
{
    if (!mvFeatInfoForInitSlam.empty())
    {
        for (const std::pair<int,Type>& vit : mvFeatInfoForInitSlam)
        {
            int id = vit.first;
            int type = vit.second;

            Feature* pFeature = mFeatures.at(id);

            if (!pFeature->IsInited())
            {
                int N = 0;
                if (type==UNUSED)
                    N = 1;
                else if (type==BAD)
                    N = mnMinTrackingLength;
                else if (type==POSE_ONLY_M)
                    N = mnMaxTrackingLength-1;
                else
                    exit(-1);

                auto vbeg = mmFeatTrackingHistory.at(id).begin()+N;
                auto vend = mmFeatTrackingHistory.at(id).end();
                std::vector<cv::Point2f>(vbeg,vend).swap(mmFeatTrackingHistory.at(id));
                pFeature->reset(pFeature->RootImageId()+N);
            }
            else
                mmFeatTrackingHistory.erase(id);
        }
    }

    if (!mvFeatInfoForPoseOnly.empty())
    {
        for (const std::pair<int,Type>& vit : mvFeatInfoForPoseOnly)
        {
            int id = vit.first;
            int type = vit.second;

            Feature* pFeature = mFeatures.at(id);

            if (type!=POSE_ONLY)
            {
                int N = 0;
                if (type==UNUSED)
                    N = 1;
                else if (type==BAD)
                    N = mnMinTrackingLength;
                else if (type==POSE_ONLY_M)
                    N = mnMaxTrackingLength-1;
                else
                    exit(-1);

                auto vbeg = mmFeatTrackingHistory.at(id).begin()+N;
                auto vend = mmFeatTrackingHistory.at(id).end();
                std::vector<cv::Point2f>(vbeg,vend).swap(mmFeatTrackingHistory.at(id));
                pFeature->reset(pFeature->RootImageId()+N);
            }
        }
    }

    if (!mvFeatIDsLoseTrack.empty())
    {
        for (const int& id : mvFeatIDsLoseTrack)
        {
            mFeatures.at(id)->clear();
            mvFeatIDsInactive.push_back(id);
            mmFeatTrackingHistory.at(id).clear();
        }

        mvFeatIDsLoseTrack.clear();
    }

    mvFeatInfoForInitSlam.clear();
    mvvFeatMeasForInitSlam.clear();
    mvFeatInfoForPoseOnly.clear();
    mvvFeatMeasForPoseOnly.clear();
    mvFeatMeasForExploration.clear();

    preprocess(nImageId, image, RcG, tcG);

    mpFeatureDetector->DetectWithSubPix(image, 1.5*mnMaxFeatsPerImage, 1, mvFeatCandidates);
}


void Tracker::track(const int nImageId, const cv::Mat& image, const mat3& RcG, const vec3& tcG, int nMapPtsNeeded, std::unordered_map<int,Feature*>& mFeatures, cv::Mat &imOut)
{
    cv::Mat workingImage = image.clone();

    if (nImageId==0 || mbRestartVT || mbRefreshVT)
    {
        if (!start(nImageId, workingImage, RcG, tcG, mFeatures))
            return;
    }else{
        manage(nImageId, workingImage, RcG, tcG, mFeatures);

        VisualTracking(nImageId, workingImage, nMapPtsNeeded, mFeatures, imOut);
    }
}


static void MakeTrackDebugBgr(const cv::Mat& image, cv::Mat& imOut)
{
    if (image.empty()) {
        imOut = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(imOut, "empty image", cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0,0,255), 2, cv::LINE_AA);
        return;
    }

    if (image.channels()==1) cv::cvtColor(image, imOut, cv::COLOR_GRAY2BGR);
    else if (image.channels()==3) image.copyTo(imOut);
    else if (image.channels()==4) cv::cvtColor(image, imOut, cv::COLOR_BGRA2BGR);
    else imOut = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
}
void Tracker::DisplayKltTracks(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vTrackFlags, cv::Mat& imOut)
{
    MakeTrackDebugBgr(image, imOut);

    int nOk = 0;
    int nLost = 0;

    int N = std::min((int)vPrevFeatUVs.size(), (int)vCurrFeatUVs.size());

    for (int i=0; i<N; ++i) {
        bool ok = i<(int)vTrackFlags.size() && vTrackFlags.at(i);
        if (ok) {
            cv::circle(imOut, vPrevFeatUVs.at(i), 3, cv::Scalar(255,64,64), -1, cv::LINE_AA);
            cv::circle(imOut, vCurrFeatUVs.at(i), 3, cv::Scalar(64,255,64), -1, cv::LINE_AA);
            cv::line(imOut, vPrevFeatUVs.at(i), vCurrFeatUVs.at(i), cv::Scalar(255,64,64), 1, cv::LINE_AA);
            nOk++;
        } else {
            cv::drawMarker(imOut, vPrevFeatUVs.at(i), cv::Scalar(64,64,255), cv::MARKER_TILTED_CROSS, 10, 2, cv::LINE_AA);
            nLost++;
        }
    }

    cv::rectangle(imOut, cv::Point(8,8), cv::Point(520,72), cv::Scalar(0,0,0), -1);
    cv::putText(imOut, "KLT: seguimiento por flujo optico", cv::Point(18,30), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(64,255,64), 2, cv::LINE_AA);
    cv::putText(imOut, "azul: posicion previa | verde: posicion actual | rojo: perdido", cv::Point(18,50), cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    cv::putText(imOut, "frame=" + std::to_string(nImageId) + " ok=" + std::to_string(nOk) + " lost=" + std::to_string(nLost), cv::Point(18,68), cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}
void Tracker::DisplayRansacTracks(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vInlierFlags, cv::Mat& imOut)
{
    MakeTrackDebugBgr(image, imOut);

    int nInliers = 0;
    int nOutliers = 0;

    int N = std::min((int)vPrevFeatUVs.size(), (int)vCurrFeatUVs.size());

    for (int i=0; i<N; ++i) {
        bool inlier = i<(int)vInlierFlags.size() && vInlierFlags.at(i);
        if (inlier) {
            cv::circle(imOut, vPrevFeatUVs.at(i), 3, cv::Scalar(255,64,64), -1, cv::LINE_AA);
            cv::circle(imOut, vCurrFeatUVs.at(i), 3, cv::Scalar(64,255,64), -1, cv::LINE_AA);
            cv::line(imOut, vPrevFeatUVs.at(i), vCurrFeatUVs.at(i), cv::Scalar(255,64,64), 1, cv::LINE_AA);
            nInliers++;
        } else {
            cv::circle(imOut, vPrevFeatUVs.at(i), 4, cv::Scalar(64,64,255), 2, cv::LINE_AA);
            if (i<(int)vCurrFeatUVs.size()) cv::line(imOut, vPrevFeatUVs.at(i), vCurrFeatUVs.at(i), cv::Scalar(64,64,255), 1, cv::LINE_AA);
            nOutliers++;
        }
    }

    cv::rectangle(imOut, cv::Point(8,8), cv::Point(560,72), cv::Scalar(0,0,0), -1);
    cv::putText(imOut, "RANSAC: rechazo geometrico de correspondencias", cv::Point(18,30), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(64,255,64), 2, cv::LINE_AA);
    cv::putText(imOut, "azul/verde: inlier | rojo: outlier", cv::Point(18,50), cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    cv::putText(imOut, "frame=" + std::to_string(nImageId) + " inliers=" + std::to_string(nInliers) + " outliers=" + std::to_string(nOutliers), cv::Point(18,68), cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}
void Tracker::DisplayTrackErrors(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vInlierFlags, const std::vector<float>& vErrors, cv::Mat& imOut)
{
    MakeTrackDebugBgr(image, imOut);

    int N = std::min((int)vPrevFeatUVs.size(), (int)vCurrFeatUVs.size());

    for (int i=0; i<N; ++i) {
        bool inlier = i<(int)vInlierFlags.size() && vInlierFlags.at(i);
        float err = i<(int)vErrors.size() ? vErrors.at(i) : 0.0f;

        if (inlier) {
            cv::line(imOut, vPrevFeatUVs.at(i), vCurrFeatUVs.at(i), cv::Scalar(255,64,64), 1, cv::LINE_AA);
            cv::circle(imOut, vCurrFeatUVs.at(i), 3, cv::Scalar(64,255,64), -1, cv::LINE_AA);
        } else {
            cv::drawMarker(imOut, vPrevFeatUVs.at(i), cv::Scalar(64,64,255), cv::MARKER_TILTED_CROSS, 10, 2, cv::LINE_AA);
        }

        if (i%8==0) cv::putText(imOut, cv::format("%.2f", err), vCurrFeatUVs.at(i)+cv::Point2f(4,-4), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    }

    cv::rectangle(imOut, cv::Point(8,8), cv::Point(580,52), cv::Scalar(0,0,0), -1);
    cv::putText(imOut, "Error KLT asociado a cada correspondencia", cv::Point(18,30), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(64,255,64), 2, cv::LINE_AA);
    cv::putText(imOut, "frame=" + std::to_string(nImageId), cv::Point(18,50), cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}
void Tracker::DisplayTrackSummary(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vInlierFlags, cv::Mat& imOut)
{
    MakeTrackDebugBgr(image, imOut);

    int nInliers = 0;
    int nOutliers = 0;

    int N = std::min((int)vPrevFeatUVs.size(), (int)vCurrFeatUVs.size());

    for (int i=0; i<N; ++i) {
        bool inlier = i<(int)vInlierFlags.size() && vInlierFlags.at(i);
        if (inlier) nInliers++;
        else nOutliers++;
    }

    double ratio = N>0 ? 100.0*(double)nInliers/(double)N : 0.0;

    for (int i=0; i<N; ++i) {
        bool inlier = i<(int)vInlierFlags.size() && vInlierFlags.at(i);
        cv::Scalar color = inlier ? cv::Scalar(64,255,64) : cv::Scalar(64,64,255);
        cv::circle(imOut, vCurrFeatUVs.at(i), 3, color, -1, cv::LINE_AA);
    }

    cv::rectangle(imOut, cv::Point(8,8), cv::Point(460,120), cv::Scalar(0,0,0), -1);
    cv::putText(imOut, "Resumen seguimiento visual", cv::Point(18,32), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(64,255,64), 2, cv::LINE_AA);
    cv::putText(imOut, "frame: " + std::to_string(nImageId), cv::Point(18,56), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    cv::putText(imOut, "tracks: " + std::to_string(N), cv::Point(18,76), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    cv::putText(imOut, "inliers: " + std::to_string(nInliers) + " | outliers: " + std::to_string(nOutliers), cv::Point(18,96), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    cv::putText(imOut, "ratio inliers: " + cv::format("%.1f", ratio) + "%", cv::Point(18,116), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}
