#pragma once

#include <vector>
#include <utility>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include "config.hpp"

class Ransac
{
public:

    Ransac(Config config);
    void FindInliers(const Eigen::MatrixXd& Points1, const Eigen::MatrixXd& Points2, const mat3& R, int& nInliers, std::vector<unsigned char>& vInlierFlags);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    void PairTwoPoints(const int nInlierCandidates, const std::vector<int>& vInlierCandidateIndexes, std::vector<std::pair<int,int> >& vTwoPointIndexes);

    void ComputeE(const int nIterNum, const Eigen::MatrixXd& Points1, const Eigen::MatrixXd& Points2, const std::vector<std::pair<int,int> >& vTwoPointIndexes, const mat3& R, mat3& E);
    int CountVotes(const Eigen::MatrixXd& Points1, const Eigen::MatrixXd& Points2, const std::vector<int>& vInlierCandidateIndexes, const mat3& E);

    inline double SampsonError(const vec3& pt1, const vec3& pt2, const mat3& E){
        vec3 Fx1 = E*pt1;
        vec3 Fx2 = E.transpose()*pt2;
        return (pow(pt2.dot(E*pt1),2))/(pow(Fx1(0),2)+pow(Fx1(1),2)+pow(Fx2(0),2)+pow(Fx2(1),2));
    }

    inline double AlgebraicError(const vec3& pt1, const vec3& pt2, const mat3& E){
        return fabs(pt2.dot(E*pt1));
    }

private:

    int mnIterations;

    bool mbUseSampson;

    double mnSampsonThrd;
    double mnAlgebraicThrd;
};
