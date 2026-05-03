#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "propagator.hpp"
#include "lie_math.hpp"

mat3 I = mat3::Identity();
Eigen::Matrix<double,15,15> I15 = Eigen::Matrix<double,15,15>::Identity();
Eigen::Matrix<double,18,18> I18 = Eigen::Matrix<double,18,18>::Identity();


Propagator::Propagator(Config config)
{
    mnImuRate = config.imu.fps;

    mnGravity = config.imu.g;
    mnSmallAngle = 1.745329e-05;

    mvGyroNoiseSigma = config.imu.allangN;
    mvGyroRandomWalkSigma = config.imu.allangK;
    mvAccelNoiseSigma = config.imu.allanaN;
    mvAccelRandomWalkSigma = config.imu.allanaK;

    mSigma.setZero();
    mSigma.block<3,3>(0,0) = mvGyroNoiseSigma.cwiseProduct(mvGyroNoiseSigma).asDiagonal();
    mSigma.block<3,3>(3,3) = mvGyroRandomWalkSigma.cwiseProduct(mvGyroRandomWalkSigma).asDiagonal();
    mSigma.block<3,3>(6,6) = mvAccelNoiseSigma.cwiseProduct(mvAccelNoiseSigma).asDiagonal();
    mSigma.block<3,3>(9,9) = mvAccelRandomWalkSigma.cwiseProduct(mvAccelRandomWalkSigma).asDiagonal();

    mnLocalWindowSize = config.vio.track_maxlength - 1;
}


void Propagator::CreateNewFactor(SourceIn * source, StateOut * state){
    Eigen::VectorXd vb = state->Localx.tail(9);
    
    vec3 bg = vb.segment(3,3);
    vec3 ba = vb.segment(6,3);
    
    vec3 vk = vb.segment(0,3);
    vec3 gk = state->Localx.segment(7,3);
    
    // Local velocity/gravity in {Rk}
    vec3 vR = vk;
    vec3 gR = gk;

    mat3 Rk, RkT;
    vec3 pk;
    
    Rk.setIdentity();
    RkT.setIdentity();
    pk.setZero();

    vec3 dp, dv;
    dp.setZero();
    dv.setZero();

    Eigen::Matrix<double,18,18> F, Phi, Psi;
    Eigen::Matrix<double,18,12> G;
    Eigen::Matrix<double,18,18> Q, P;
    F.setZero();
    Phi.setZero();
    Psi.setIdentity();
    G.setZero();
    Q.setZero();
    P.setZero();

    double Dt = 0;

    for (const ImuSample& data : source->imu)
    {
        vec3 wm = data.vgyr;
        vec3 am = data.vacc;
        double dt = data.dt;

        vec3 w = wm-bg;
        vec3 a = am-ba;
        Dt += dt;

        double wn = w.norm();
        mat3 wx = skewMat(w);
        mat3 wx2 = wx*wx;
        mat3 vx = skewMat(vk);
        mat3 gx = skewMat(gk);

        // Covariance
        F.block<3,3>(3,3) = -wx;
        F.block<3,3>(3,12) = -I;
        F.block<3,3>(6,3) = -RkT*vx;
        F.block<3,3>(6,9) = RkT;
        F.block<3,3>(9,0) = -mnGravity*Rk;
        F.block<3,3>(9,3) = -mnGravity*gx;
        F.block<3,3>(9,9) = -wx;
        F.block<3,3>(9,12) = -vx;
        F.block<3,3>(9,15) = -I;

        G.block<3,3>(3,0) = -I;
        G.block<3,3>(9,0) = -vx;
        G.block<3,3>(9,6) = -I;
        G.block<3,3>(12,3) = I;
        G.block<3,3>(15,9) = I;

        Phi = I18+dt*F;
        Q = dt*G*mSigma*(G.transpose());

        P = Phi*P*(Phi.transpose())+Q;
        P = .5*(P+P.transpose()).eval();

        Psi.applyOnTheLeft(Phi);

        // State 
        mat3 dR;
        double f1, f2, f3, f4;
        if (wn<mnSmallAngle)
        {
            dR = I-dt*wx+.5*pow(dt,2)*wx2;

            f1 = -pow(dt,3)/3;
            f2 = pow(dt,4)/8;
            f3 = -pow(dt,2)/2;
            f4 = pow(dt,3)/6;
        }
        else
        {
            dR = I-sin(wn*dt)/wn*wx+(1.-cos(wn*dt))/pow(wn,2)*wx2;

            f1 = (wn*dt*cos(wn*dt)-sin(wn*dt))/pow(wn,3);
            f2 = .5*(pow(wn*dt,2)-2*cos(wn*dt)-2*wn*dt*sin(wn*dt)+2)/pow(wn,4);
            f3 = (cos(wn*dt)-1.)/pow(wn,2);
            f4 = (wn*dt-sin(wn*dt))/pow(wn,3);
        }

        Rk.applyOnTheLeft(dR);
        RkT = Rk.transpose();

        dp += dv*dt;
        dp += RkT*(0.5*pow(dt,2)*I+f1*wx+f2*wx2)*a;
        dv += RkT*(dt*I+f3*wx+f4*wx2)*a;

        // aqui
        pk = vR*Dt-0.5*mnGravity*gR*pow(Dt,2)+dp;
        vk = Rk*(vR-mnGravity*gR*Dt+dv);

        gk = Rk*gR;
        gk.normalize();
    }

    
    Eigen::Matrix<double,15,15> Info;

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(P.bottomRightCorner(15,15));
    if (qr.isInvertible())
        Info = qr.inverse();
    else
        Info = pseudoInverse(P.bottomRightCorner(15,15));

    Info = .5*(Info+Info.transpose()).eval();
    Eigen::Matrix<double,15,15> Low = Info.llt().matrixL();

    Eigen::Matrix<double,15,27> Hf;
    Hf << Psi.bottomLeftCorner(15,3), Psi.bottomRightCorner(15,9), -I15;
    Hf.applyOnTheLeft(Low.adjoint());
    state->H = Hf;

    state->x << RotToQuat(Rk),
                pk,
                vk,
                bg,
                ba;
}


void Propagator::LocalQR(const int nImageId, StateOut * state){
    int L = state->LocalFactor.rows();
    int W = nImageId>mnLocalWindowSize ? 6*mnLocalWindowSize+9 : 6*(nImageId-1)+9;

    int Lg = L-W-7-3;
    int Lv = L-9;

    Eigen::MatrixXd tempLocalSR;
    tempLocalSR.setZero(L+15,L+15);
    tempLocalSR.topLeftCorner(L,L) = state->LocalFactor.leftCols(L);
    tempLocalSR.block(L,Lg,15,3) = state->H.leftCols(3);
    tempLocalSR.block(L,Lv,15,9) = state->H.block(0,3,15,9);
    tempLocalSR.bottomRightCorner(15,15) = state->H.rightCols(15);

    FlipToHead(tempLocalSR.leftCols(L), 9);

    if (nImageId>mnLocalWindowSize)
        FlipToHead(tempLocalSR.leftCols(9+L-W+6), 6);

    Eigen::JacobiRotation<double> GR;

    for (int n=0; n<L+15; ++n)
    {
        for (int m=L+14; m>n; --m)
        {
            GR.makeGivens(tempLocalSR(m-1,n),tempLocalSR(m,n));
            tempLocalSR.applyOnTheLeft(m-1,m,GR.adjoint());
        }
    }

    if (nImageId>mnLocalWindowSize)
    {
        state->LocalFactor.leftCols(L) = tempLocalSR.bottomRightCorner(L,L).triangularView<Eigen::Upper>();
    }
    else
    {
        state->LocalFactor.setZero(L+6,L+7);
        state->LocalFactor.leftCols(L+6) = tempLocalSR.bottomRightCorner(L+6,L+6).triangularView<Eigen::Upper>();
    }

    if (nImageId>mnLocalWindowSize)
    {
        int w = 7*mnLocalWindowSize+9;
        Eigen::VectorXd tempx = state->Localx.segment(10+8+7,w-16);
        state->Localx.segment(10+8,w-16).swap(tempx);
        state->Localx.tail(16) = state->x;
    }
    else
    {
        int l = state->Localx.rows();
        state->Localx.conservativeResize(l+7);
        state->Localx.tail(16) = state->x;
    }
}


void Propagator::propagate(const int nImageId, SourceIn * source, StateOut * state){
    CreateNewFactor(source, state);
    LocalQR(nImageId, state);
}
