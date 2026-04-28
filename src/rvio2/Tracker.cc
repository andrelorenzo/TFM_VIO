/**
* This file is part of R-VIO2.
*
* Copyright (C) 2022 Zheng Huai <zhuai@udel.edu> and Guoquan Huang <ghuang@udel.edu>
* For more information see <http://github.com/rpng/R-VIO2> 
*
* R-VIO2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* R-VIO2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with R-VIO2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstdlib>
#include <iostream>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "Tracker.h"
#include "util/config_compat.h"
#include "util/numerics.h"


namespace RVIO2
{

int featId = 0;

const cv::Scalar red(64,64,255);
const cv::Scalar blue(255,64,64);
const cv::Scalar green(64,255,64);


Tracker::Tracker(const cv::FileStorage& fsSettings)
{
    const int bIsRGB = config_compat::ReadBool(fsSettings, {"cam.RGB", "Camera.RGB"}, false) ? 1 : 0;
    mbIsRGB = bIsRGB;

    const int bIsFisheye = config_compat::ReadBool(fsSettings, {"cam.fisheye", "Camera.Fisheye"}, false) ? 1 : 0;
    mbIsFisheye = bIsFisheye;

    const int bEnableFilter = config_compat::ReadBool(fsSettings, {"Tracker.EnableFilter", "rvio.enable_filter"}, false) ? 1 : 0;
    mbEnableFilter = bEnableFilter;

    const int bEnableEqualizer = config_compat::ReadBool(fsSettings, {"Tracker.EnableEqualizer", "rvio.enable_equalizer"}, true) ? 1 : 0;
    mbEnableEqualizer = bEnableEqualizer;

    mnMaxFeatsPerImage = config_compat::ReadInt(
        fsSettings, {"Tracker.nFeatures", "rvio.nFeatures", "orb.nFeatures", "orb.max_feat", "dep.max_feat"}, 200);

    mnMaxTrackingLength = config_compat::ReadInt(
        fsSettings, {"Tracker.nMaxTrackingLength", "rvio.max_track_length"}, 15);
    mnMinTrackingLength = config_compat::ReadInt(
        fsSettings, {"Tracker.nMinTrackingLength", "rvio.min_track_length", "vio.min_track_length"}, 3);

    const int nMaxSlamPoints = config_compat::ReadInt(fsSettings, {"Tracker.nMaxSlamPoints", "rvio.max_slam_points"}, 0);
    mbEnableSlam = nMaxSlamPoints>0 ? true : false;

    mnGoodParallax = config_compat::ReadFloat(
        fsSettings, {"Tracker.nGoodParallax", "rvio.good_parallax_deg", "vio.good_parallax_deg"}, 5.0f);

    mLastImage = cv::Mat();

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = config_compat::ReadFloat(fsSettings, {"cam.fx", "Camera.fx"});
    K.at<float>(1,1) = config_compat::ReadFloat(fsSettings, {"cam.fy", "Camera.fy"});
    K.at<float>(0,2) = config_compat::ReadFloat(fsSettings, {"cam.cx", "Camera.cx"});
    K.at<float>(1,2) = config_compat::ReadFloat(fsSettings, {"cam.cy", "Camera.cy"});
    K.copyTo(mK);

    const auto dist = config_compat::ReadDistCoeffs(fsSettings);
    cv::Mat D(4,1,CV_32F);
    D.at<float>(0) = static_cast<float>(dist[0]);
    D.at<float>(1) = static_cast<float>(dist[1]);
    D.at<float>(2) = static_cast<float>(dist[2]);
    D.at<float>(3) = static_cast<float>(dist[3]);
    const float k3 = static_cast<float>(dist[4]);
    if (k3!=0)
    {
        D.resize(5);
        D.at<float>(4) = k3;
    }
    D.copyTo(mD);

    mbRestartVT = false;
    mbRefreshVT = false;

    mpRansac = new Ransac(fsSettings);
    mpFeatureDetector = new FeatureDetector(fsSettings);

    mTrackPub = mTrackerNode.advertise<sensor_msgs::Image>("/rvio2/track", 1);
    mNewerPub = mTrackerNode.advertise<sensor_msgs::Image>("/rvio2/newer", 1);

    const int bShowTrack = config_compat::ReadBool(fsSettings, {"Displayer.ShowTrack", "rvio.show_track"}, true) ? 1 : 0;
    mbShowTrack = bShowTrack;

    const int bShowNewer = config_compat::ReadBool(fsSettings, {"Displayer.ShowNewer", "rvio.show_newer"}, false) ? 1 : 0;
    mbShowNewer = bShowNewer;
}


Tracker::~Tracker()
{
    delete mpRansac;
    delete mpFeatureDetector;
}


void Tracker::preprocess(const int nImageId, 
                         cv::Mat& image, 
                         const Eigen::Matrix3f& RcG, 
                         const Eigen::Vector3f& tcG)
{
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


void Tracker::undistort(const std::vector<cv::Point2f>& src, 
                        std::vector<cv::Point2f>& dst)
{
    int N = src.size();

    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; ++i)
    {
        mat.at<float>(i,0) = src.at(i).x;
        mat.at<float>(i,1) = src.at(i).y;
    }

    mat = mat.reshape(2);

    if (!mbIsFisheye)
        cv::undistortPoints(mat, mat, mK, mD);
    else
        cv::fisheye::undistortPoints(mat, mat, mK, mD);

    mat = mat.reshape(1);

    dst.resize(N);
    for(int i=0; i<N; ++i)
    {
        dst.at(i).x = mat.at<float>(i,0);
        dst.at(i).y = mat.at<float>(i,1);
    }
}


void Tracker::DisplayTrack(const int nImageId, 
                           const cv::Mat& image, 
                           const std::vector<cv::Point2f>& vPrevFeatUVs, 
                           const std::vector<cv::Point2f>& vCurrFeatUVs, 
                           const std::vector<unsigned char>& vInlierFlags, 
    cv_bridge::CvImage& imOut)
{
    imOut.encoding = "bgr8";
    cv::cvtColor(image, imOut.image, cv::COLOR_GRAY2BGR);

    for (int i=0; i<(int)vPrevFeatUVs.size(); ++i)
    {
        if (vInlierFlags.at(i))
        {
            cv::circle(imOut.image, vPrevFeatUVs.at(i), 3, blue, -1);
            cv::line(imOut.image, vPrevFeatUVs.at(i), vCurrFeatUVs.at(i), blue);
        }
        else
            cv::circle(imOut.image, vPrevFeatUVs.at(i), 3, red, 0);
    }

    cv::putText(imOut.image, std::to_string(nImageId), cv::Point2f(15,30), cv::FONT_HERSHEY_PLAIN, 2, green, 2);
}


void Tracker::DisplayNewer(const int nImageId, 
                           const cv::Mat& image, 
                           const std::vector<cv::Point2f>& vRefFeatUVs, 
                           const std::vector<cv::Point2f>& vNewFeatUVs,
    cv_bridge::CvImage& imOut)
{
    imOut.encoding = "bgr8";
    cv::cvtColor(image, imOut.image, cv::COLOR_GRAY2BGR);

    for (const cv::Point2f& pt : vRefFeatUVs)
        cv::circle(imOut.image, pt, 3, blue, 0);

    for (const cv::Point2f& pt : vNewFeatUVs)
        cv::circle(imOut.image, pt, 3, green, -1);

    cv::putText(imOut.image, std::to_string(nImageId), cv::Point2f(15,30), cv::FONT_HERSHEY_PLAIN, 2, green, 2);
}


void Tracker::VisualTracking(const int nImageId, 
                             const cv::Mat& image, 
                             int nMapPtsNeeded, 
                             std::unordered_map<int,Feature*>& mFeatures)
{
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

    if (std::accumulate(vInlierFlags.begin(), vInlierFlags.end(), 0)>0)
    {
        undistort(vFeatPts, vFeatPtsUN);
        Eigen::MatrixXf MatchesForRansac(3,nFeats);

        for (int i=0; i<nFeats; ++i)
        {
            cv::Point2f pt = vFeatPtsUN.at(i);
            MatchesForRansac.col(i) << pt.x, pt.y, 1;
            MatchesForRansac.col(i).normalize();
        }

        mpRansac->FindInliers(PointsForRansac, MatchesForRansac, mRr, nInliers, vInlierFlags);

        if (nInliers==0)
        {
            std::cerr << "Visual Tracking: lost all features, refresh if the refill fails!" << "\n";
            mbRefreshVT = true;
        }
    }
    else
    {
        std::cerr << "Visual Tracking: lost all features, refresh anyway!" << "\n";
        mbRefreshVT = true;
    }

    if (mbShowTrack)
    {
        // Show the result in rviz
        cv_bridge::CvImage imTrack;
        DisplayTrack(nImageId, image, mvFeatPtsToTrack, vFeatPts, vInlierFlags, imTrack);
        mTrackPub.publish(imTrack.toImageMsg());
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
                        float parallax = Parallax(vTrack.front(), vTrack.back());

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
                cv_bridge::CvImage imNewer;
                DisplayNewer(nImageId, image, mvFeatPtsToTrack, vNewFeatPts, imNewer);
                mNewerPub.publish(imNewer.toImageMsg());
            }

            if (mbRefreshVT)
                mbRefreshVT = false;
        }
    }

    image.copyTo(mLastImage);
}


bool Tracker::start(const int nImageId, 
                    cv::Mat image, 
                    const Eigen::Matrix3f& RcG, 
                    const Eigen::Vector3f& tcG, 
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
                     const Eigen::Matrix3f& RcG, 
                     const Eigen::Vector3f& tcG, 
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


void Tracker::track(const int nImageId, 
                    const cv::Mat& image, 
                    const Eigen::Matrix3f& RcG, 
                    const Eigen::Vector3f& tcG, 
                    int nMapPtsNeeded, 
                    std::unordered_map<int,Feature*>& mFeatures)
{
    cv::Mat workingImage = image.clone();

    if (nImageId==0 || mbRestartVT || mbRefreshVT)
    {
        if (!start(nImageId, workingImage, RcG, tcG, mFeatures))
            return;
    }
    else
    {
        manage(nImageId, workingImage, RcG, tcG, mFeatures);

        VisualTracking(nImageId, workingImage, nMapPtsNeeded, mFeatures);
    }
}

} // namespace RVIO2
