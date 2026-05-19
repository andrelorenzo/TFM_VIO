#pragma once

#include <list>
#include <vector>
#include <utility>
#include <unordered_map>
#include <cmath>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "ransac.hpp"
#include "feature.hpp"
#include "feature_detector.hpp"
#include "config.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


enum Type
{
    // Original values
    // 0: Init slam - reach the max. tracking length
    // 1: Pose only - reach the max. tracking length
    // 2: Pose only - lose track
    // 3: Exploration - local SLAM feature
    INIT_SLAM, POSE_ONLY_M, POSE_ONLY, EXPLO, 
    // Return values
    // 5: Unused measurement
    // 6: Bad measurement
    UNUSED, BAD
};


class Tracker
{
public:

    Tracker(Config config);

    ~Tracker();

    void track(const int nImageId, const cv::Mat& image, const mat3& RcG, const vec3& tcG, int nMapPtsNeeded, std::unordered_map<int,Feature*>& mFeatures, cv::Mat &imOut);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    bool start(const int nImageId, cv::Mat image, const mat3& RcG, const vec3& tcG, std::unordered_map<int,Feature*>& mFeatures);
    void manage(const int nImageId, cv::Mat& image, const mat3& RcG, const vec3& tcG, const std::unordered_map<int,Feature*>& mFeatures);
    void preprocess(const int nImageId, cv::Mat& image, const mat3& RcG, const vec3& tcG);
    void undistort(const std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst);
    void VisualTracking(const int nImageId, const cv::Mat& image, int nMapPtsNeeded, std::unordered_map<int,Feature*>& mFeatures, cv::Mat &imOut);
    void DisplayTrack(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vInlierFlags, cv::Mat& imOut);
    void DisplayNewer(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vRefFeatUVs, const std::vector<cv::Point2f>& vNewFeatUVs, cv::Mat& imOut);

    inline void OrienVec(const cv::Point2f& pt, vec3& e)
    {
        double phi = atan2(pt.y, sqrt(pow(pt.x,2)+1));
        double psi = atan2(pt.x, 1);
        e << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);
    }

    inline double Parallax(const cv::Point2f& pt0, const cv::Point2f& ptk)
    {
        vec3 e0, ek;
        OrienVec(pt0, e0);
        OrienVec(ptk, ek);
        double theta = fabs(acos(ek.dot(mRx*e0)));
        return 40*sin(theta)>1 ? theta*180/M_PI : 0;
    }
void DisplayKltTracks(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vTrackFlags, cv::Mat& imOut);

void DisplayRansacTracks(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vInlierFlags, cv::Mat& imOut);

void DisplayTrackErrors(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vInlierFlags, const std::vector<float>& vErrors, cv::Mat& imOut);

void DisplayTrackSummary(const int nImageId, const cv::Mat& image, const std::vector<cv::Point2f>& vPrevFeatUVs, const std::vector<cv::Point2f>& vCurrFeatUVs, const std::vector<unsigned char>& vInlierFlags, cv::Mat& imOut);

public:

    std::vector<std::pair<int,Type> > mvFeatInfoForInitSlam;
    std::vector<std::vector<cv::Point2f> > mvvFeatMeasForInitSlam;

    std::vector<std::pair<int,Type> > mvFeatInfoForPoseOnly;
    std::vector<std::vector<cv::Point2f> > mvvFeatMeasForPoseOnly;

    std::vector<std::pair<int,cv::Point2f> > mvFeatMeasForExploration;

private:

    std::vector<cv::Point2f> mvFeatCandidatesRaw;

    bool mbIsRGB;

    bool mbRestartVT;
    bool mbRefreshVT;

    bool mbEnableSlam;
    bool mbEnableFilter;
    bool mbEnableEqualizer;

    int mnMaxFeatsPerImage;

    int mnMinTrackingLength;
    int mnMaxTrackingLength;

    double mnGoodParallax;

    cv::Mat mLastImage;

    cv::Mat mK;
    cv::Mat mD;

    mat3 mRx;
    mat3 mRr;

    std::list<mat3> mlCamOrientations;
    std::list<vec3> mlCamPositions;

    std::unordered_map<int,std::vector<cv::Point2f> > mmFeatTrackingHistory;
    std::vector<int> mvFeatIDsToTrack;
    std::vector<cv::Point2f> mvFeatPtsToTrack;

    std::vector<int> mvFeatIDsInactive;
    std::vector<int> mvFeatIDsLoseTrack;

    Eigen::MatrixXd PointsForRansac;
    std::vector<cv::Point2f> mvFeatCandidates;

    Ransac* mpRansac;
    FeatureDetector* mpFeatureDetector;

    bool mbShowTrack;
    bool mbShowNewer;
};
