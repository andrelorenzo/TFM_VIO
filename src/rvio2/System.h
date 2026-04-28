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

#ifndef SYSTEM_H
#define SYSTEM_H

#include <deque>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <opencv2/core/core.hpp>

#include "Tracker.h"
#include "Updater.h"
#include "Propagator.h"
#include "Feature.h"
#include "InputBuffer.h"
#include "util/ros_compat.h"


namespace RVIO2
{

struct PoseEstimate
{
    int ImageId;
    double Timestamp;
    Eigen::Vector4f Quaternion;
    Eigen::Vector3f Position;

    PoseEstimate()
        : ImageId(-1), Timestamp(0.0)
    {
        Quaternion << 0.f, 0.f, 0.f, 1.f;
        Position.setZero();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class System
{
public:

    System(const std::string& strSettingsFile);

    ~System();

    bool run(PoseEstimate* pOutput = nullptr);

    void PushImuData(ImuData* pData) {mpInputBuffer->PushImuData(pData);}
    void PushImageData(ImageData* pData) {mpInputBuffer->PushImageData(pData);}

    inline const PoseEstimate& LatestPose() const {return mLatestPose;}
    inline bool IsInitialized() const {return mbIsInitialized;}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    bool initialize(const ImageData& Image, const std::vector<ImuData>& vImuData);
    void setLatestPose(int nImageId, double timestamp);

protected:

    // State vector
    Eigen::VectorXf Localx;

    // Information factor
    Eigen::MatrixXf LocalFactor;

    std::deque<Eigen::Vector3f> mqLocalw;
    std::deque<Eigen::Vector3f> mqLocalv;

    std::unordered_map<int,Feature*> mmFeatures;
    std::vector<int> mvActiveFeatureIDs;

private:

    bool mbRecordOutputs;

    bool mbEnableAlignment;
    bool mbUseGroundTruthCalib;

    bool mbIsInitialized;

    int mnImageId;
    int mnMaxSlamPoints;
    int mnLocalWindowSize;

    float mnGravity;
    float mnImuRate;

    float mnCamRate;
    float mnCamTimeOffset;

    Eigen::Vector3f mvSigmaGyroNoise;
    Eigen::Vector3f mvSigmaGyroBias;
    Eigen::Vector3f mvSigmaAccelNoise;
    Eigen::Vector3f mvSigmaAccelBias;

    float mnAngleThrd;
    float mnLengthThrd;

    Eigen::Matrix3f mRci;
    Eigen::Vector3f mtci;

    InputBuffer* mpInputBuffer;
    Propagator* mpPropagator;
    Updater* mpUpdater;
    Tracker* mpTracker;

    PoseEstimate mLatestPose;

    // Interact with rviz
    ros::NodeHandle mSystemNode;
    ros::Publisher mMapPub;
    ros::Publisher mPathPub;
    tf::TransformBroadcaster mTfPub;
};

} // namespace RVIO2

#endif
