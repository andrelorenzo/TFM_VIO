#include <cstdlib>
#include <iostream>
#include <numeric>

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
    mnGoodParallax      = config.vio.good_para;

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

        mpRansac->FindInliers(PointsForRansac, MatchesForRansac, mRr, nInliers, vInlierFlags);

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
                // Show the result in rviz
                DisplayNewer(nImageId, image, mvFeatPtsToTrack, vNewFeatPts, imOut);
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
