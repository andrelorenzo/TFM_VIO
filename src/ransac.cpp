#include <cmath>
#include <cstdlib>

#include "ransac.hpp"



Ransac::Ransac(Config config){
    
    mnIterations = config.vio.rsac_it<16 ? 16 : config.vio.rsac_it;
    mbUseSampson = config.vio.sampson_on;

    mnSampsonThrd = config.vio.sampson_threserr;
    mnAlgebraicThrd = config.vio.sampson_algerr;
}


void Ransac::PairTwoPoints(const int nInlierCandidates, const std::vector<int>& vInlierCandidateIndexes, std::vector<std::pair<int,int> >& vTwoPointIndexes){
    for (int i=0; i<mnIterations; ++i) {
        int idx1 = i;
        int idx2;
        do
            idx2 = rand()%nInlierCandidates;
        while (idx1==idx2);

        int idxA = vInlierCandidateIndexes.at(idx1);
        int idxB = vInlierCandidateIndexes.at(idx2);
        vTwoPointIndexes.emplace_back(idxA,idxB);
    }
}


void Ransac::ComputeE(const int nIterNum, const Eigen::MatrixXd& Points1, const Eigen::MatrixXd& Points2, const std::vector<std::pair<int,int> >& vTwoPointIndexes, const mat3& R, mat3& E){
    // point(i)(j): i=(A,B) in reference frame j=(1,2)
    vec3 pointA1 = Points1.col(vTwoPointIndexes.at(nIterNum).first);
    vec3 pointA2 = Points2.col(vTwoPointIndexes.at(nIterNum).first);
    vec3 pointB1 = Points1.col(vTwoPointIndexes.at(nIterNum).second);
    vec3 pointB2 = Points2.col(vTwoPointIndexes.at(nIterNum).second);

    // p0=R*p1
    vec3 pointA0 = R*pointA1;
    vec3 pointB0 = R*pointB1;

    // The solution of (p2^T)[tx]p0=0, where t is in terms of sinusoidals of
    // two directional angles: cos(alpha), sin(alpha), cos(beta) and sin(beta).
    // We need two correspondences (4 constraints) to solve t: {A0,A2}, {B0,B2}.
    double c1 = pointA2(0)*pointA0(1)-pointA0(0)*pointA2(1);
    double c2 = pointA0(1)*pointA2(2)-pointA2(1)*pointA0(2);
    double c3 = pointA2(0)*pointA0(2)-pointA0(0)*pointA2(2);
    double c4 = pointB2(0)*pointB0(1)-pointB0(0)*pointB2(1);
    double c5 = pointB0(1)*pointB2(2)-pointB2(1)*pointB0(2);
    double c6 = pointB2(0)*pointB0(2)-pointB0(0)*pointB2(2);

    double alpha = atan2(c3*c5-c2*c6, c1*c6-c3*c4);
    double beta = atan2(-c3, c1*sin(alpha)+c2*cos(alpha));

    double t0 = sin(beta)*cos(alpha);
    double t1 = cos(beta);
    double t2 = -sin(beta)*sin(alpha);

    mat3 tx;
    tx <<  0, -t2,  t1,  t2,   0, -t0, -t1,  t0,   0;
    E = tx*R;
}


int Ransac::CountVotes(const Eigen::MatrixXd& Points1, const Eigen::MatrixXd& Points2, const std::vector<int>& vInlierCandidateIndexes, const mat3& E){
    int nVotes = 0;
    for (const int &idx : vInlierCandidateIndexes){
        if (mbUseSampson){
            if (SampsonError(Points1.col(idx), Points2.col(idx), E)<mnSampsonThrd) nVotes++;
        }else{
            if (AlgebraicError(Points1.col(idx), Points2.col(idx), E)<mnAlgebraicThrd) nVotes++;
        }
    }

    return nVotes;
}


void Ransac::FindInliers(const Eigen::MatrixXd& Points1, const Eigen::MatrixXd& Points2, const mat3& R, int& nInliers, std::vector<unsigned char>& vInlierFlags){
    int nInlierCandidates = 0;
    std::vector<int> vInlierCandidateIndexes;
    vInlierCandidateIndexes.reserve((int)vInlierFlags.size());

    for (int i=0; i<(int)vInlierFlags.size(); ++i){
        if (vInlierFlags.at(i)){
            vInlierCandidateIndexes.push_back(i);
            nInlierCandidates++;
        }
    }

    std::vector<std::pair<int,int> > vTwoPointIndexes;
    if (nInlierCandidates>mnIterations){
        vTwoPointIndexes.reserve(nInlierCandidates);
        PairTwoPoints(nInlierCandidates, vInlierCandidateIndexes, vTwoPointIndexes);
    }else{
        // Not enough points to process
        Logger(WARN,
               "RANSAC skipped: only %d LK inlier candidates available, need > %d iterations",
               nInlierCandidates,
               mnIterations);
        std::fill(vInlierFlags.begin(), vInlierFlags.end(), 0);
        nInliers = 0;
        return;
    }

    int nWinnerVotes = 0;
    mat3 WinnerE;
    for (int iter=0; iter<mnIterations; ++iter){
        mat3 E;
        ComputeE(iter, Points1, Points2, vTwoPointIndexes, R, E);
        int nVotes = CountVotes(Points1, Points2, vInlierCandidateIndexes, E);
        if (nVotes>nWinnerVotes){
            WinnerE.swap(E);
            nWinnerVotes = nVotes;
        }
    }

    int nOutliers = 0;
    for (int i=0; i<nInlierCandidates; ++i){
        int idx = vInlierCandidateIndexes.at(i);

        if (mbUseSampson){
            double nError = SampsonError(Points1.col(idx), Points2.col(idx), WinnerE);
            if (nError>mnSampsonThrd || std::isinf(nError) || std::isnan(nError)){
                // Mark as outlier
                vInlierFlags.at(idx) = 0;
                nOutliers++;
            }
        }else{
            double nError = AlgebraicError(Points1.col(idx), Points2.col(idx), WinnerE);
            if (nError>mnAlgebraicThrd || std::isinf(nError) || std::isnan(nError)){
                // Mark as outlier
                vInlierFlags.at(idx) = 0;
                nOutliers++;
            }
        }
    }

    nInliers = nInlierCandidates-nOutliers;
}

