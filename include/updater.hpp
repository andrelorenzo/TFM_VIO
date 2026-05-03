#pragma once

#include <deque>
#include <vector>
#include <utility>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "feature.hpp"
#include "tracker.hpp"
#include "config.hpp"

class Updater
{
public:

    Updater(Config config);

    void update(const int nImageId, const std::unordered_map<int,Feature*>& mFeatures, const std::vector<std::pair<int,cv::Point2f> >& vFeatMeasForExploration, 
        std::vector<std::pair<int,Type> >& vFeatInfoForInitSlam, const std::vector<std::vector<cv::Point2f> >& vvFeatMeasForInitSlam, std::vector<std::pair<int,Type> >& vFeatInfoForPoseOnly, 
        const std::vector<std::vector<cv::Point2f> >& vvFeatMeasForPoseOnly, std::vector<int>& vActiveFeatureIDs, const std::deque<vec3>& qLocalw, 
        const std::deque<vec3>& qLocalv, Eigen::VectorXd& Localx, Eigen::MatrixXd& LocalFactor);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    int triangulate(const int nTrackLength, const std::vector<cv::Point2f>& vRevFeatMeas, const std::vector<Eigen::Matrix<double,7,1> >& vRevRelCamPoses,
        double& phi, double& psi, double& rho);

    void composition(const int nImageId, const std::unordered_map<int,Feature*>& mFeatures, std::vector<int>& vActiveFeatureIDs, const int nDimOfWinx, 
        const int nDimOfWinSR, Eigen::VectorXd& Localx, Eigen::MatrixXd& LocalFactor);

    void ComposeQR(const int nIdx, const int nDim, Eigen::MatrixXd& LocalFactor);
    void LocalQR(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, Eigen::MatrixXd& LocalFactor);
    void ReorderQR(const std::vector<int>& vFeatureStatuses, std::vector<int>& vFeatureIDs, Eigen::MatrixXd& LocalFactor);

    void CreateNewFactor(const cv::Point2f& z, const vec3& pfG, const vec3& pfG_fej, const Eigen::Matrix<double,7,1>& xG, 
        const Eigen::Matrix<double,7,1>& xk, const vec3& wk, const vec3& vk, Eigen::MatrixXd& Hf, Eigen::MatrixXd& HG, Eigen::MatrixXd& HP, 
        Eigen::MatrixXd& Hk, Eigen::VectorXd& r);

    bool CreateNewFactor(std::pair<int,Type>& pFeatInfo, const std::vector<cv::Point2f>& vRevFeatMeas, const std::vector<Eigen::Matrix<double,7,1> >& vRevRelImuPoses, 
        const std::vector<Eigen::Matrix<double,7,1> >& vRevRelCamPoses, const std::deque<vec3>& qRevLocalw, const std::deque<vec3>& qRevLocalv, 
        Eigen::MatrixXd& Hf, Eigen::MatrixXd& HP, Eigen::MatrixXd& HW, Eigen::VectorXd& r, vec3& xf);

    void GetRevRelPoses(const int type, const int nTrackLength, const Eigen::VectorXd& Winx, std::vector<Eigen::Matrix<double,7,1> >& vRevRelImuPoses, 
        std::vector<Eigen::Matrix<double,7,1> >& vRevRelCamPoses);

    void InitLM(const double phi, const double psi, const double rho, const int nTrackLength, const std::vector<cv::Point2f>& vRevFeatMeas, const std::vector<Eigen::Matrix<double,7,1> >& vRevRelCamPoses, 
        mat3& HTRinvH, vec3& HTRinve, double& cost);

    inline double chi2(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, const Eigen::Ref<const Eigen::MatrixXd> U) const{
        Eigen::MatrixXd V = H*U;
        Eigen::MatrixXd S = V*(V.transpose());
        S.diagonal() += pow(mnImageNoiseSigma,2)*Eigen::VectorXd::Ones(H.rows());
        return r.dot(S.llt().solve(r));
    }

    inline void FlipToTail(Eigen::Ref<Eigen::MatrixXd> Mat, const int dim){
        int cols = Mat.cols();
        Eigen::MatrixXd tempM1 = Mat.leftCols(dim);
        Eigen::MatrixXd tempM2 = Mat.rightCols(cols-dim);
        Mat.rightCols(dim).swap(tempM1);
        Mat.leftCols(cols-dim).swap(tempM2);
    }

    inline void FlipToHead(Eigen::Ref<Eigen::MatrixXd> Mat, const int dim){
        int cols = Mat.cols();
        Eigen::MatrixXd tempM1 = Mat.rightCols(dim);
        Eigen::MatrixXd tempM2 = Mat.leftCols(cols-dim);
        Mat.leftCols(dim).swap(tempM1);
        Mat.rightCols(cols-dim).swap(tempM2);
    }

private:

    int mnLocalWindowSize;

    double mnImageNoiseSigma;
    double mnImageNoiseSigmaInv;

    Eigen::Matrix2d mSigmaInv;

    mat3 mRci;
    vec3 mtci;
    mat3 mRic;
    vec3 mtic;

    std::vector<int> mvNewActiveFeatureIDs;
    std::vector<int> mvLostActiveFeatureIDs;
};
