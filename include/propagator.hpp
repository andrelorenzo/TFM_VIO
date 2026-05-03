#pragma once

#include <vector>
#include <unordered_map>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>

#include "config.hpp"


class Propagator
{
public:

    Propagator(Config config);

    void propagate(const int nImageId, SourceIn * source, StateOut * state);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    void CreateNewFactor(SourceIn * source, StateOut * state);

    void LocalQR(const int nImageId, StateOut * state);

    inline void FlipToHead(Eigen::Ref<Eigen::MatrixXd> Mat, const int dim)
    {
        int cols = Mat.cols();
        Eigen::MatrixXd tempM1 = Mat.rightCols(dim);
        Eigen::MatrixXd tempM2 = Mat.leftCols(cols-dim);
        Mat.leftCols(dim).swap(tempM1);
        Mat.rightCols(cols-dim).swap(tempM2);
    }



private:

    int mnLocalWindowSize;

    double mnImuRate;
    double mnGravity;
    double mnSmallAngle;

    vec3 mvGyroNoiseSigma;
    vec3 mvGyroRandomWalkSigma;
    vec3 mvAccelNoiseSigma;
    vec3 mvAccelRandomWalkSigma;

    Eigen::Matrix<double,12,12> mSigma;
};

