#include <cmath>
#include <cstdlib>
#include <numeric>

#include "updater.hpp"
#include "lie_math.hpp"


int cloud_id = 0;


Updater::Updater(Config config){

    mnLocalWindowSize = config.vio.track_maxlength - 1;
    mnImageNoiseSigma = std::max(config.cam.spx, config.cam.spy);
    mnImageNoiseSigmaInv = 1./mnImageNoiseSigma;

    mSigmaInv << pow(mnImageNoiseSigmaInv,2), 0, 0, pow(mnImageNoiseSigmaInv,2);
}


int Updater::triangulate(const int nTrackLength, const std::vector<cv::Point2f>& vRevFeatMeas, const std::vector<Eigen::Matrix<double,7,1> >& vRevRelCamPoses, 
    double& phi, double& psi, double& rho){
    phi = atan2(vRevFeatMeas.front().y, sqrt(pow(vRevFeatMeas.front().x,2)+1));
    psi = atan2(vRevFeatMeas.front().x, 1);
    rho = 0.;
    if (fabs(phi)>.5*M_PI || fabs(psi)>.5*M_PI)
        return -1;

    int nIter = 0;
    int maxIter = 20;
    double lambda = 1e-2;

    mat3 HTRinvH, new_HTRinvH;
    vec3 HTRinve, new_HTRinve;
    double cost, new_cost;

    InitLM(phi, psi, rho, nTrackLength, vRevFeatMeas, vRevRelCamPoses, 
           HTRinvH, HTRinve, cost);

    while (nIter<maxIter)
    {
        HTRinvH.diagonal() *= 1.+lambda;
        vec3 dpfinv = HTRinvH.llt().solve(HTRinve);

        double new_phi = phi+dpfinv(0);
        double new_psi = psi+dpfinv(1);
        double new_rho = rho+dpfinv(2);

        if (dpfinv.norm()<1e-6)
            break;

        InitLM(new_phi, new_psi, new_rho, nTrackLength, vRevFeatMeas, vRevRelCamPoses, 
               new_HTRinvH, new_HTRinve, new_cost);

        if (new_cost<cost)
        {
            phi = new_phi;
            psi = new_psi;
            rho = new_rho;

            if (fabs(cost-new_cost)<1e-6 || fabs(dpfinv(2))<1e-6)
                break;

            HTRinvH.swap(new_HTRinvH);
            HTRinve.swap(new_HTRinve);
            cost = new_cost;

            lambda /= 2;
        }
        else
        {
            if (fabs(cost-new_cost)<1e-6 || fabs(dpfinv(2))<1e-6)
                break;

            lambda *= 1.5;
        }

        nIter++;
    }

    if (fabs(phi)>.5*M_PI || fabs(psi)>.5*M_PI || std::isnan(rho) || std::isinf(rho) || rho<0)
        return -1;

    if (nIter==maxIter)
        return 0;

    return 1;
}


void Updater::InitLM(const double phi, const double psi, const double rho, const int nTrackLength, const std::vector<cv::Point2f>& vRevFeatMeas, 
    const std::vector<Eigen::Matrix<double,7,1> >& vRevRelCamPoses, mat3& HTRinvH, vec3& HTRinve, double& cost){
        
    vec3 epfinv;
    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);

    Eigen::Matrix<double,3,2> Jang;
    Jang << -sin(phi)*sin(psi), cos(phi)*cos(psi),
             cos(phi), 0,
            -sin(phi)*cos(psi), -cos(phi)*sin(psi);

    HTRinvH.setZero();
    HTRinve.setZero();
    cost = 0;

    for (int i=0; i<nTrackLength; ++i)
    {
        cv::Point2f z = vRevFeatMeas.at(i);

        mat3 Rc = QuatToRot(vRevRelCamPoses.at(i).head(4));
        vec3 tc = vRevRelCamPoses.at(i).tail(3);
        vec3 h = Rc*epfinv+rho*tc;

        Eigen::Matrix<double,2,3> Hproj;
        Hproj << 1./h(2), 0, -h(0)/pow(h(2),2),
                 0, 1./h(2), -h(1)/pow(h(2),2);

        Eigen::Matrix<double,2,3> H;
        H << Hproj*Rc*Jang, Hproj*tc;

        Eigen::Vector2d e;
        e << z.x-h(0)/h(2), z.y-h(1)/h(2);

        HTRinvH += H.transpose()*mSigmaInv*H;
        HTRinve += H.transpose()*mSigmaInv*e;
        cost += e.dot(mSigmaInv*e);
    }
}


void Updater::GetRevRelPoses(const int type, 
                             const int nTrackLength, 
                             const Eigen::VectorXd& Winx, 
                             std::vector<Eigen::Matrix<double,7,1> >& vRevRelImuPoses, 
                             std::vector<Eigen::Matrix<double,7,1> >& vRevRelCamPoses)
{
    int k = 0;
    if (type==POSE_ONLY)
        k = Winx.rows()/7-1;
    else
        k = mnLocalWindowSize;

    vRevRelImuPoses.resize(nTrackLength);
    vRevRelCamPoses.resize(nTrackLength);
    vRevRelImuPoses.at(0) << 0, 0, 0, 1, 0, 0, 0;
    vRevRelCamPoses.at(0) << 0, 0, 0, 1, 0, 0, 0;

    for (int i=1; i<nTrackLength; ++i)
    {
        Eigen::Matrix<double,7,1> x1 = vRevRelImuPoses.at(i-1);
        Eigen::Matrix<double,7,1> x2 = Winx.segment(7*(k-i),7);

        mat3 R = QuatToRot(QuatInv(x2.head(4)))*QuatToRot(x1.head(4));
        vec3 t = x2.tail(3)+QuatToRot(QuatInv(x2.head(4)))*x1.tail(3);

        vRevRelImuPoses.at(i) << RotToQuat(R), t;
        vRevRelCamPoses.at(i) << RotToQuat(mRci*R*mRic), mRci*R*mtic+mRci*t+mtci;
    }
}


bool Updater::CreateNewFactor(std::pair<int,Type>& pFeatInfo, 
                              const std::vector<cv::Point2f>& vRevFeatMeas, 
                              const std::vector<Eigen::Matrix<double,7,1> >& vRevRelImuPoses, 
                              const std::vector<Eigen::Matrix<double,7,1> >& vRevRelCamPoses, 
                              const std::deque<vec3>& qRevLocalw, 
                              const std::deque<vec3>& qRevLocalv, 
                              Eigen::MatrixXd& Hf, 
                              Eigen::MatrixXd& HP, 
                              Eigen::MatrixXd& HW, 
                              Eigen::VectorXd& r, 
                              vec3& xf)
{
    int type = pFeatInfo.second;
    int nTrackLength = vRevFeatMeas.size();

    double phi, psi, rho;
    int rval = triangulate(nTrackLength, vRevFeatMeas, vRevRelCamPoses, phi, psi, rho);
    if (rval==-1)
    {
        if (type!=POSE_ONLY)
            pFeatInfo.second = BAD;

        return false;
    }
    else if (rval==0)
    {
        if (type!=POSE_ONLY)
            pFeatInfo.second = UNUSED;

        return false;
    }
    else
    {
        if (type==INIT_SLAM)
        {
            double bl = vRevRelCamPoses.back().tail(3).norm();
            double thrd = 40*bl*rho;
            if (thrd<1.)
            {
                type = POSE_ONLY_M;
                pFeatInfo.second = POSE_ONLY_M;
            }
        }
    }

    xf << phi, psi, rho;

    vec3 epfinv;
    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);

    Eigen::Matrix<double,3,2> Jang;
    Jang << -sin(phi)*sin(psi),  cos(phi)*cos(psi),
             cos(phi), 0,
            -sin(phi)*cos(psi), -cos(phi)*sin(psi);

    Eigen::MatrixXd tempHf, tempHP, tempHW;
    Eigen::VectorXd tempr;

    tempr.setZero(2*nTrackLength);
    tempHf.setZero(2*nTrackLength,3);
    tempHP.setZero(2*nTrackLength,7);

    if (type!=POSE_ONLY)
        tempHW.setZero(2*nTrackLength,6*(nTrackLength-1));
    else
        tempHW.setZero(2*nTrackLength,6*(nTrackLength-1)+15);

    Eigen::Matrix<double,2,3> subHqci, subHtci;
    Eigen::Vector2d subHt;

    mat3 I;
    I.setIdentity();

    for (int i=0; i<nTrackLength; ++i)
    {
        cv::Point2f z = vRevFeatMeas.at(i);

        mat3 R = QuatToRot(vRevRelImuPoses.at(i).head(4));
        vec3 t = vRevRelImuPoses.at(i).tail(3);

        mat3 Rc = QuatToRot(vRevRelCamPoses.at(i).head(4));
        vec3 tc = vRevRelCamPoses.at(i).tail(3);
        vec3 h = Rc*epfinv+rho*tc;

        Eigen::Matrix<double,2,3> Hproj;
        Hproj << 1./h(2), 0, -h(0)/pow(h(2),2),
                 0, 1./h(2), -h(1)/pow(h(2),2);

        tempr.segment(2*i,2) << z.x-h(0)/h(2), z.y-h(1)/h(2);

        tempHf.block(2*i,0,2,3) << Hproj*Rc*Jang, Hproj*tc;

        subHt.setZero();
        subHqci.setZero();
        subHtci.setZero();

        if (i==0)
        {
            vec3 wc, vc;

            if (type!=POSE_ONLY)
            {
                wc = mRci*qRevLocalw.at(0);
                vc = mRci*qRevLocalv.at(0);
            }
            else
            {
                wc = mRci*qRevLocalw.at(1);
                vc = mRci*qRevLocalv.at(1);
            }

            subHt += Hproj*(Rc*(wc.cross(epfinv))+(vc.dot(epfinv))*tc);
        }

        if (i>0)
        {
            Eigen::Matrix<double,2,3> HRR = Hproj*mRci*R;

            for (int j=0; j<i; ++j)
            {
                mat3 R1T = QuatToRot(vRevRelImuPoses.at(j).head(4)).transpose();
                vec3 t1 = -R1T*vRevRelImuPoses.at(j).tail(3);
                mat3 R2T = QuatToRot(vRevRelImuPoses.at(j+1).head(4)).transpose();

                const vec3 vx_arg = (mRic*epfinv + rho*mtic - rho*t1).eval();
                mat3 Vx = skewMat(vx_arg);

                subHt += HRR*(-Vx*R1T*qRevLocalw.at(j)+rho*R1T*qRevLocalv.at(j));

                tempHW.block(2*i,6*(nTrackLength-2-j),2,6) << -HRR*Vx*R1T, rho*HRR*R2T;
            }

            const vec3 rc_epfinv = (Rc*epfinv).eval();
            const vec3 rc_mtci = (Rc*mtci).eval();
            const vec3 mrci_t = (mRci*t).eval();
            subHqci = Hproj*((skewMat(rc_epfinv)-rho*skewMat(rc_mtci))*(I-Rc)+rho*skewMat(mrci_t));
            subHtci = Hproj*rho*(I-Rc);
        }

        tempHP.block(2*i,0,2,7) << subHqci, subHtci, subHt;
    }

    int M = 2*nTrackLength;
    int N = 3;

    if (tempHf.col(2).norm()/M<1e-6)
    {
        N = 2;

        if (type==INIT_SLAM)
        {
            type = POSE_ONLY_M;
            pFeatInfo.second = POSE_ONLY_M;
        }
    }

    Eigen::JacobiRotation<double> GR;

    for (int n=0; n<N; ++n)
    {
        for (int m=M-1; m>n; --m)
        {
            GR.makeGivens(tempHf(m-1,n), tempHf(m,n));
            tempHf.applyOnTheLeft(m-1,m,GR.adjoint());
            tempHP.applyOnTheLeft(m-1,m,GR.adjoint());
            tempHW.applyOnTheLeft(m-1,m,GR.adjoint());
            tempr.applyOnTheLeft(m-1,m,GR.adjoint());
            tempHf(m,n) = 0;
        }
    }

    if (type==INIT_SLAM)
    {
        Hf = tempHf;
        HP = tempHP;
        HW = tempHW;
        r = tempr;
    }
    else
    {
        HP = tempHP.bottomRows(M-N);
        HW = tempHW.bottomRows(M-N);
        r = tempr.tail(M-N);
    }

    return true;
}


void Updater::CreateNewFactor(const cv::Point2f& z, 
                              const vec3& pfG, 
                              const vec3& pfG_fej, 
                              const Eigen::Matrix<double,7,1>& xG, 
                              const Eigen::Matrix<double,7,1>& xk, 
                              const vec3& wk, 
                              const vec3& vk, 
                              Eigen::MatrixXd& Hf, 
                              Eigen::MatrixXd& HG, 
                              Eigen::MatrixXd& HP, 
                              Eigen::MatrixXd& Hk, 
                              Eigen::VectorXd& r)
{
    r.resize(2);
    Hf.resize(2,3);
    HG.resize(2,9);
    HP.resize(2,7);
    Hk.resize(2,15);

    mat3 RG = QuatToRot(xG.head(4));
    vec3 pG = xG.tail(3);

    mat3 Rk = QuatToRot(xk.head(4));
    vec3 tk = xk.tail(3);

    mat3 RkG = Rk*RG;
    vec3 pkG = Rk*(pG-tk);
    vec3 pfk = RkG*pfG+pkG;
    vec3 pfk_fej = RkG*pfG_fej+pkG;

    vec3 h = mRci*pfk+mtci;
    vec3 h_fej = mRci*pfk_fej+mtci;

    Eigen::Matrix<double,2,3> Hproj;
    Hproj << 1./h_fej(2), 0, -h_fej(0)/pow(h_fej(2),2),
             0, 1./h_fej(2), -h_fej(1)/pow(h_fej(2),2);

    Eigen::Matrix<double,2,3> HR = Hproj*mRci;

    r << z.x-h(0)/h(2), z.y-h(1)/h(2);

    Hf = HR*RkG;
    const vec3 rg_pfg_fej = (RG*pfG_fej).eval();
    const vec3 mrci_pfk_fej = (mRci*pfk_fej).eval();
    HG << HR*Rk*skewMat(rg_pfg_fej), HR*Rk, Eigen::MatrixXd::Zero(2,3);
    HP << Hproj*skewMat(mrci_pfk_fej), Hproj, HR*skewMat(pfk_fej)*wk-HR*vk;
    Hk << HR*skewMat(pfk_fej), -HR*Rk, Eigen::MatrixXd::Zero(2,9);
}


void Updater::LocalQR(const Eigen::MatrixXd& H, 
                      const Eigen::VectorXd& r, 
                      Eigen::MatrixXd& LocalFactor)
{
    int M = H.rows();
    int N = H.cols();

    Eigen::MatrixXd tempLocalFactor;
    tempLocalFactor.resize(N+M,N+1);
    tempLocalFactor << LocalFactor.bottomRightCorner(N,N+1), H, r;

    Eigen::JacobiRotation<double> GR;

    for (int n=0; n<N; ++n)
    {
        for (int m=N+M-1; m>=N; --m)
        {
            if (tempLocalFactor(m,n)!=0)
            {
                GR.makeGivens(tempLocalFactor(n,n),tempLocalFactor(m,n));
                tempLocalFactor.applyOnTheLeft(n,m,GR.adjoint());
                tempLocalFactor(m,n) = 0;
            }
        }
    }

    LocalFactor.bottomRightCorner(N,N+1) = tempLocalFactor.topRows(N);
}


void Updater::ReorderQR(const std::vector<int>& vFeatureStatuses, 
                        std::vector<int>& vFeatureIDs, 
                        Eigen::MatrixXd& LocalFactor)
{
    int L = LocalFactor.rows();
    int l1 = 0;
    int l2 = 3*vFeatureIDs.size();

    std::vector<int> vActiveIDs;

    for (int n=0; n<(int)vFeatureIDs.size(); ++n)
    {
        int id = vFeatureIDs.at(n);

        if (vFeatureStatuses.at(n)==0)
        {
            if (n>0)
            {
                int N = 3*(n+1);

                FlipToHead(LocalFactor.block(0,l1,l1+l2,N), 3);
            }

            mvLostActiveFeatureIDs.push_back(id);
        }
        else
            vActiveIDs.push_back(id);
    }

    vFeatureIDs.swap(vActiveIDs);

    Eigen::MatrixXd tempLocalFactor;
    tempLocalFactor = LocalFactor.block(l1,l1,l2,L+1-l1);

    Eigen::JacobiRotation<double> GR;

    for (int n=0; n<l2; ++n)
    {
        for (int m=l2-1; m>n; --m)
        {
            if (tempLocalFactor(m,n)!=0)
            {
                GR.makeGivens(tempLocalFactor(m-1,n),tempLocalFactor(m,n));
                tempLocalFactor.applyOnTheLeft(m-1,m,GR.adjoint());
                tempLocalFactor(m,n) = 0;
            }
        }
    }

    LocalFactor.block(l1,l1,l2,L+1-l1).swap(tempLocalFactor);
}


void Updater::update(const int nImageId, const std::unordered_map<int,Feature*>& mFeatures, const std::vector<std::pair<int,cv::Point2f> >& vFeatMeasForExploration, 
    std::vector<std::pair<int,Type> >& vFeatInfoForInitSlam, const std::vector<std::vector<cv::Point2f> >& vvFeatMeasForInitSlam, std::vector<std::pair<int,Type> >& vFeatInfoForPoseOnly, 
    const std::vector<std::vector<cv::Point2f> >& vvFeatMeasForPoseOnly, std::vector<int>& vActiveFeatureIDs, const std::deque<vec3>& qLocalw, const std::deque<vec3>& qLocalv, 
    Eigen::VectorXd& Localx, Eigen::MatrixXd& LocalFactor){

    int nWinSize = nImageId>mnLocalWindowSize ? mnLocalWindowSize : nImageId;
    int nDimOfWinx = 7*nWinSize;
    int nDimOfWinSR = 6*nWinSize+9;

    if (vFeatMeasForExploration.empty() && vFeatInfoForInitSlam.empty() && vFeatInfoForPoseOnly.empty())
    {
        mvLostActiveFeatureIDs.swap(vActiveFeatureIDs);

        composition(nImageId, mFeatures, vActiveFeatureIDs, nDimOfWinx, nDimOfWinSR, Localx, LocalFactor);

        return;
    }

    mRci = QuatToRot(Localx.segment(10,4));
    mtci = Localx.segment(14,3);
    mRic = mRci.transpose();
    mtic = -mRic*mtci;

    Eigen::VectorXd Winx = Localx.segment(10+8,nDimOfWinx);

    Eigen::MatrixXd NavSRinv;
    NavSRinv.setIdentity(9+7+nDimOfWinSR,9+7+nDimOfWinSR);
    LocalFactor.bottomRightCorner(9+7+nDimOfWinSR,9+7+nDimOfWinSR+1).leftCols(9+7+nDimOfWinSR)
               .triangularView<Eigen::Upper>().solveInPlace(NavSRinv);

    if (!vFeatMeasForExploration.empty())
    {


        Eigen::MatrixXd H;
        Eigen::VectorXd r;
        int nNewRows = 0;

        vec3 wk = qLocalw.back();
        vec3 vk = qLocalv.back();

        Eigen::Matrix<double,7,1> xG = Localx.head(7);
        Eigen::Matrix<double,7,1> xk = Localx.tail(16).head(7);

        int nActiveFeatures = vActiveFeatureIDs.size();
        std::vector<int> vFeatureStatuses(nActiveFeatures,0);

        for (std::vector<std::pair<int,cv::Point2f> >::const_iterator vitMeas=vFeatMeasForExploration.begin(); vitMeas!=vFeatMeasForExploration.end(); ++vitMeas){
            int id = vitMeas->first;
            Feature* pFeature = mFeatures.at(id);

            // Get feature index
            auto vit = std::find_if(vActiveFeatureIDs.begin(), vActiveFeatureIDs.end(), [id](const int& val){return val==id;});
            int idx = vit-vActiveFeatureIDs.begin();
            vFeatureStatuses.at(idx) = 1;

            vec3 pfG = pFeature->Position();
            vec3 pfG_fej = pFeature->FejPosition();

            Eigen::MatrixXd tempHf, tempHG, tempHP, tempHk;
            Eigen::VectorXd tempr;
            CreateNewFactor((*vitMeas).second, pfG, pfG_fej, xG, xk, wk, vk, 
                            tempHf, tempHG, tempHP, tempHk, tempr);

            Eigen::MatrixXd tempH;
            tempH.setZero(2,9+7+nDimOfWinSR);
            tempH.leftCols(16) << tempHG, tempHP;
            tempH.rightCols(15).swap(tempHk);

            double val = chi2(tempH, tempr, NavSRinv);
            if (val>CHI2_THRESHOLD[2-1])continue;

            nNewRows += 2;

            H.conservativeResizeLike(Eigen::MatrixXd::Zero(nNewRows,3*nActiveFeatures+9+7+nDimOfWinSR));
            H.block(nNewRows-2,3*idx,2,3).swap(tempHf);
            H.bottomRightCorner(2,9+7+nDimOfWinSR).swap(tempH);

            r.conservativeResize(nNewRows);
            r.tail(2).swap(tempr);
        }

        if (nNewRows>0)
        {
            H *= mnImageNoiseSigmaInv;
            r *= mnImageNoiseSigmaInv;
            LocalQR(H, r, LocalFactor);
        }

        int n = std::accumulate(vFeatureStatuses.begin(), vFeatureStatuses.end(), 0);
        if (n<nActiveFeatures)
        {
            if (n>0)
                ReorderQR(vFeatureStatuses, vActiveFeatureIDs, LocalFactor);
            else
                mvLostActiveFeatureIDs.swap(vActiveFeatureIDs);
        }
    }
    else
        mvLostActiveFeatureIDs.swap(vActiveFeatureIDs);

    if (!vFeatInfoForInitSlam.empty())
    {

        Eigen::MatrixXd Hf;
        Eigen::VectorXd rf;
        int nNewPoints = 0;

        Eigen::MatrixXd Hx;
        Eigen::VectorXd rx;
        int nNewRows = 0;

        std::deque<vec3> qRevLocalw(qLocalw.rbegin(), qLocalw.rend());
        std::deque<vec3> qRevLocalv(qLocalv.rbegin(), qLocalv.rend());

        std::vector<Eigen::Matrix<double,7,1> > vRevRelImuPoses, vRevRelCamPoses;
        GetRevRelPoses(INIT_SLAM, mnLocalWindowSize+1, Winx, vRevRelImuPoses, vRevRelCamPoses);

        std::vector<std::pair<int,Type> >::iterator vitInfo = vFeatInfoForInitSlam.begin();
        std::vector<std::vector<cv::Point2f> >::const_iterator vitMeas = vvFeatMeasForInitSlam.begin();

        for (; vitInfo!=vFeatInfoForInitSlam.end(); ++vitInfo, ++vitMeas)
        {
            int id = (*vitInfo).first;
            Feature* pFeature = mFeatures.at(id);

            std::vector<cv::Point2f> vRevFeatMeas((*vitMeas).rbegin(), (*vitMeas).rend());

            Eigen::MatrixXd tempHf, tempHP, tempHW;
            Eigen::VectorXd tempr;
            vec3 xf;
            if (!CreateNewFactor(*vitInfo, vRevFeatMeas, vRevRelImuPoses, vRevRelCamPoses, qRevLocalw, qRevLocalv, 
                                 tempHf, tempHP, tempHW, tempr, xf))
                continue;

            int M = tempHW.rows();
            int N = tempHW.cols();

            int type = (*vitInfo).second;

            if (type==INIT_SLAM)
            {
                Eigen::MatrixXd tempHx;
                Eigen::VectorXd temprx;
                tempHx.setZero(M-3,7+nDimOfWinSR);
                tempHx.leftCols(7+N) << tempHP.bottomRows(M-3), tempHW.bottomRows(M-3);
                temprx = tempr.tail(M-3);

                double val = chi2(tempHx, temprx, NavSRinv.bottomRightCorner(7+nDimOfWinSR,7+nDimOfWinSR));
                if (val<CHI2_THRESHOLD[M-3-1])
                {
                    double phi = xf(0);
                    double psi = xf(1);
                    double rho = xf(2);

                    vec3 epfinv;
                    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);
                    vec3 pf = mRic*(1./rho*epfinv)+mtic;

                }
                else
                {
                    (*vitInfo).second = BAD;
                    continue;
                }

                nNewRows += M-3;
                Hx.conservativeResize(nNewRows,7+nDimOfWinSR);
                Hx.bottomRows(M-3).swap(tempHx);

                rx.conservativeResize(nNewRows);
                rx.tail(M-3).swap(temprx);

                nNewPoints++;
                Hf.conservativeResizeLike(Eigen::MatrixXd::Zero(3*nNewPoints,7+nDimOfWinSR+3*nNewPoints));
                Hf.bottomLeftCorner(3,7+N) << tempHP.topRows(3), tempHW.topRows(3);
                Hf.bottomRightCorner(3,3) = tempHf.topRows(3);

                rf.conservativeResize(3*nNewPoints);
                rf.tail(3) = tempr.head(3);

                pFeature->SetPosition(xf);
                pFeature->SetFejPosition(xf);

                mvNewActiveFeatureIDs.push_back(id);
            }
            else
            {
                Eigen::MatrixXd tempH;
                tempH.setZero(M,7+nDimOfWinSR);
                tempH.leftCols(7+N) << tempHP, tempHW;

                double val = chi2(tempH, tempr, NavSRinv.bottomRightCorner(7+nDimOfWinSR,7+nDimOfWinSR));
                if (val<CHI2_THRESHOLD[M-1])
                {
                    double phi = xf(0);
                    double psi = xf(1);
                    double rho = xf(2);

                    vec3 epfinv;
                    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);
                    vec3 pf = mRic*(1./rho*epfinv)+mtic;

                }
                else
                {
                    (*vitInfo).second = BAD;
                    continue;
                }

                nNewRows += M;

                Hx.conservativeResize(nNewRows,7+nDimOfWinSR);
                Hx.bottomRows(M).swap(tempH);

                rx.conservativeResize(nNewRows);
                rx.tail(M).swap(tempr);
            }
        }

        if (nNewRows>0)
        {
            Hx *= mnImageNoiseSigmaInv;
            rx *= mnImageNoiseSigmaInv;
            LocalQR(Hx, rx, LocalFactor);
        }

        if (nNewPoints>0)
        {
            Hf *= mnImageNoiseSigmaInv;
            rf *= mnImageNoiseSigmaInv;

            int L = LocalFactor.rows();
            int l1 = L-9-7-nDimOfWinSR;
            int l2 = 3*nNewPoints;
            int l3 = 9+7+nDimOfWinSR;

            Eigen::MatrixXd tempM;
            tempM.setZero(L+l2,L+l2+1);
            tempM.topLeftCorner(l1,l1) = LocalFactor.topLeftCorner(l1,l1);
            tempM.topRightCorner(l1,l3+1) = LocalFactor.topRightCorner(l1,l3+1);
            tempM.bottomRightCorner(l3,l3+1) = LocalFactor.bottomRightCorner(l3,l3+1);
            tempM.block(l1,l1,l2,l2) = Hf.rightCols(l2);
            tempM.block(l1,l1+l2+9,l2,7+nDimOfWinSR+1) << Hf.leftCols(7+nDimOfWinSR), rf;
            LocalFactor.swap(tempM);
        }
    }

    if (!vFeatInfoForPoseOnly.empty())
    {

        Eigen::MatrixXd H;
        Eigen::VectorXd r;
        int nNewRows = 0;

        std::vector<std::pair<int,Type> >::iterator vitInfo = vFeatInfoForPoseOnly.begin();
        std::vector<std::vector<cv::Point2f> >::const_iterator vitMeas = vvFeatMeasForPoseOnly.begin();

        for (; vitInfo!=vFeatInfoForPoseOnly.end(); ++vitInfo, ++vitMeas)
        {
            int id = (*vitInfo).first;
            Feature* pFeature = mFeatures.at(id);

            int type = (*vitInfo).second;

            int nTrackLength = (int)(*vitMeas).size();

            std::deque<vec3> qRevLocalw(qLocalw.rbegin(), qLocalw.rend());
            std::deque<vec3> qRevLocalv(qLocalv.rbegin(), qLocalv.rend());
            if (type==POSE_ONLY)
            {
                qRevLocalw.pop_front();
                qRevLocalv.pop_front();
            }

            std::vector<Eigen::Matrix<double,7,1> > vRevRelImuPoses, vRevRelCamPoses;
            GetRevRelPoses(type, nTrackLength, Winx, vRevRelImuPoses, vRevRelCamPoses);

            std::vector<cv::Point2f> vRevFeatMeas((*vitMeas).rbegin(), (*vitMeas).rend());

            Eigen::MatrixXd tempHf, tempHP, tempHW;
            Eigen::VectorXd tempr;
            vec3 xf;
            if (!CreateNewFactor(*vitInfo, vRevFeatMeas, vRevRelImuPoses, vRevRelCamPoses, qRevLocalw, qRevLocalv, 
                                 tempHf, tempHP, tempHW, tempr, xf))
                continue;

            int M = tempHW.rows();
            int N = tempHW.cols();

            Eigen::MatrixXd tempH;
            tempH.setZero(M,7+nDimOfWinSR);
            if (type==POSE_ONLY)
            {
                tempH.leftCols(7).swap(tempHP);
                tempH.rightCols(N).swap(tempHW);
            }
            else
                tempH.leftCols(7+N) << tempHP, tempHW;

            double val = chi2(tempH, tempr, NavSRinv.bottomRightCorner(7+nDimOfWinSR,7+nDimOfWinSR));
            if (val<CHI2_THRESHOLD[M-1])
            {
                if (xf(2)>0)
                {
                    double phi = xf(0);
                    double psi = xf(1);
                    double rho = xf(2);

                    vec3 epfinv;
                    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);
                    vec3 pf = mRic*(1./rho*epfinv)+mtic;

                    if (type==POSE_ONLY)
                    {
                        mat3 R = QuatToRot(Winx.tail(7).head(4));
                        vec3 t = Winx.tail(3);
                        pf = R*(pf-t);
                    }
                }
            }
            else
            {
                if (type==POSE_ONLY_M)
                    (*vitInfo).second = BAD;

                continue;
            }

            nNewRows += M;

            H.conservativeResize(nNewRows,7+nDimOfWinSR);
            H.bottomRows(M).swap(tempH);

            r.conservativeResize(nNewRows);
            r.tail(M).swap(tempr);
        }

        if (nNewRows>0)
        {
            H *= mnImageNoiseSigmaInv;
            r *= mnImageNoiseSigmaInv;
            LocalQR(H, r, LocalFactor);
        }
    }

    Eigen::VectorXd dLocalx = LocalFactor.rightCols(1);

    int L = LocalFactor.rows();
    LocalFactor.leftCols(L).triangularView<Eigen::Upper>().solveInPlace(dLocalx);
    LocalFactor.rightCols(1).setZero();

    int nStateOffset = 0;
    int nErrorOffset = 0;

    for (const int &id : mvLostActiveFeatureIDs)
    {
        mFeatures.at(id)->Position() += dLocalx.segment(nErrorOffset,3);
        nErrorOffset += 3;
    }

    for (const int &id : vActiveFeatureIDs)
    {
        mFeatures.at(id)->Position() += dLocalx.segment(nErrorOffset,3);
        nErrorOffset += 3;
    }

    for (const int &id : mvNewActiveFeatureIDs)
    {
        mFeatures.at(id)->Position() += dLocalx.segment(nErrorOffset,3);
        nErrorOffset += 3;
    }

    // xG
    vec4 dqG;
    dqG.head(3) = .5*dLocalx.segment(nErrorOffset,3);
    double dqGvn = dqG.head(3).norm();
    if (dqGvn>1)
    {
        dqG.head(3) *= 1./sqrt(1.+pow(dqGvn,2));
        dqG(3) = 1./sqrt(1.+pow(dqGvn,2));
    }
    else
        dqG(3) = sqrt(1.-pow(dqGvn,2));

    Localx.segment(nStateOffset,4) = QuatMul(dqG, Localx.segment(nStateOffset,4));
    Localx.segment(nStateOffset+4,6) += dLocalx.segment(nErrorOffset+3,6);
    Localx.segment(nStateOffset+7,3).normalize();

    nStateOffset += 10;
    nErrorOffset += 9;

    // xP
    vec4 dqP;
    dqP.head(3) = .5*dLocalx.segment(nErrorOffset,3);
    double dqPvn = dqP.head(3).norm();
    if (dqPvn>1)
    {
        dqP.head(3) *= 1./sqrt(1.+pow(dqPvn,2));
        dqP(3) = 1./sqrt(1.+pow(dqPvn,2));
    }
    else
        dqP(3) = sqrt(1.-pow(dqPvn,2));

    Localx.segment(nStateOffset,4) = QuatMul(dqP, Localx.segment(nStateOffset,4));
    Localx.segment(nStateOffset+4,3) += dLocalx.segment(nErrorOffset+3,3);
    Localx.segment(nStateOffset+7,1) += dLocalx.segment(nErrorOffset+6,1);

    nStateOffset += 8;
    nErrorOffset += 7;

    // xW
    for (int i=0; i<nWinSize; ++i)
    {
        vec4 dqw;
        dqw.head(3) = .5*dLocalx.segment(nErrorOffset,3);
        double dqwvn = dqw.head(3).norm();
        if (dqwvn>1)
        {
            dqw.head(3) *= 1./sqrt(1.+pow(dqwvn,2));
            dqw(3) = 1./sqrt(1.+pow(dqwvn,2));
        }
        else
            dqw(3) = sqrt(1.-pow(dqwvn,2));

        Localx.segment(nStateOffset,4) = QuatMul(dqw, Localx.segment(nStateOffset,4));
        Localx.segment(nStateOffset+4,3) += dLocalx.segment(nErrorOffset+3,3);

        nStateOffset += 7;
        nErrorOffset += 6;
    }

    Localx.segment(nStateOffset,9) += dLocalx.segment(nErrorOffset,9);

    mRci = QuatToRot(Localx.segment(10,4));
    mtci = Localx.segment(14,3);
    mRic = mRci.transpose();
    mtic = -mRic*mtci;

    composition(nImageId, mFeatures, vActiveFeatureIDs, nDimOfWinx, nDimOfWinSR, Localx, LocalFactor);
}


void Updater::ComposeQR(const int nIdx, 
                        const int nDim, 
                        Eigen::MatrixXd& LocalFactor)
{
    int L = LocalFactor.rows();

    Eigen::MatrixXd tempLocalSR;
    tempLocalSR = LocalFactor.block(nIdx,nIdx,nDim,L-nIdx);

    Eigen::JacobiRotation<double> GR;

    for (int n=0; n<nDim; ++n)
    {
        for (int m=nDim-1; m>n; --m)
        {
            if (tempLocalSR(m,n)!=0)
            {
                GR.makeGivens(tempLocalSR(m-1,n),tempLocalSR(m,n));
                tempLocalSR.applyOnTheLeft(m-1,m,GR.adjoint());
                tempLocalSR(m,n) = 0;
            }
        }
    }

    LocalFactor.block(nIdx,nIdx,nDim,L-nIdx).swap(tempLocalSR);
}


void Updater::composition(const int nImageId, 
                          const std::unordered_map<int,Feature*>& mFeatures, 
                          std::vector<int>& vActiveFeatureIDs, 
                          const int nDimOfWinx, 
                          const int nDimOfWinSR, 
                          Eigen::VectorXd& Localx, 
                          Eigen::MatrixXd& LocalFactor)
{
    Eigen::Matrix<double,10,1> xG = Localx.head(10);
    mat3 RG = QuatToRot(xG.head(4));
    vec3 pG = xG.segment(4,3);
    vec3 g = xG.tail(3);

    Eigen::Matrix<double,16,1> xk = Localx.tail(16);
    mat3 Rk = QuatToRot(xk.head(4));
    vec3 tk = xk.segment(4,3);
    mat3 RkT = Rk.transpose();

    mat3 RkG = Rk*RG;
    vec3 pkG = Rk*(pG-tk);
    mat3 RkGT = RkG.transpose();

    vec3 gk = Rk*g;
    gk.normalize();

    int L = LocalFactor.rows();
    int LG = L-9-7-nDimOfWinSR;
    int LC = LG+9;
    int Lk = L-15;

    Eigen::MatrixXd Jk, Hk, HG;
    Jk.setZero(9,6);
    Jk.block(0,0,3,3).setIdentity();
    Jk.block(3,0,3,3) = skewMat(pkG);
    Jk.block(3,3,3,3) = -Rk;
    Jk.block(6,0,3,3) = skewMat(gk);

    HG.setZero(9,9);
    HG.block(0,0,3,3) = RkT;
    HG.block(3,3,3,3) = RkT;
    HG.block(6,6,3,3) = RkT;
    Hk = -HG*Jk;

    Eigen::MatrixXd tempM;
    tempM = LocalFactor.block(0,LG,LG+9,9);
    LocalFactor.block(0,LG,LG+9,9) = tempM*HG;
    LocalFactor.block(0,Lk,LG+9,6) += tempM*Hk;

    ComposeQR(LG, 9, LocalFactor);

    Localx.head(10) << RotToQuat(RkG), pkG, gk;

    int nActiveFeatures = vActiveFeatureIDs.size();
    int nNewActiveFeatures = mvNewActiveFeatureIDs.size();
    int nLostActiveFeatures = mvLostActiveFeatureIDs.size();

    if (nNewActiveFeatures>0)
    {
        int l1 = 3*nLostActiveFeatures+3*nActiveFeatures;
        int l2 = 3*nNewActiveFeatures;

        Eigen::MatrixXd Jf, JG, JP, Hf, HG, HP;
        Jf.setZero(l2,l2);
        JG.setZero(l2,6);
        JP.setZero(l2,7);

        for (int i=0; i<nNewActiveFeatures; ++i)
        {
            int id = mvNewActiveFeatureIDs.at(i);
            Feature* pFeature = mFeatures.at(id);

            vec3 xf = pFeature->Position();
            double phi = xf(0);
            double psi = xf(1);
            double rho = xf(2);

            vec3 xf_fej = pFeature->FejPosition();
            double phi_fej = xf_fej(0);
            double psi_fej = xf_fej(1);
            double rho_fej = xf_fej(2);

            vec3 epfinv, epfinv_fej;
            epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);
            epfinv_fej << cos(phi_fej)*sin(psi_fej), sin(phi_fej), cos(phi_fej)*cos(psi_fej);

            vec3 pfG = RkGT*(1./rho*mRic*epfinv+mtic-pkG);
            vec3 pfG_fej = RkGT*(1./rho_fej*mRic*epfinv_fej+mtic-pkG);

            Eigen::Matrix<double,3,2> Jang;
            Jang << -sin(phi_fej)*sin(psi_fej), cos(phi_fej)*cos(psi_fej),
                     cos(phi_fej), 0,
                    -sin(phi_fej)*cos(psi_fej), -cos(phi_fej)*sin(psi_fej);

            Jf.block(3*i,3*i,3,3) << 1./rho_fej*RkGT*mRic*Jang, -1./pow(rho_fej,2)*RkGT*mRic*epfinv_fej;
            JG.block(3*i,0,3,6) << -skewMat(pfG_fej)*RkGT, -RkGT;
            const vec3 invrho_ep_minus_mtci = ((1./rho_fej)*epfinv_fej - mtci).eval();
            JP.block(3*i,0,3,7) << -RkGT*mRic*skewMat(invrho_ep_minus_mtci), -RkGT*mRic, vec3::Zero();

            pFeature->SetPosition(pfG);
            pFeature->SetFejPosition(pfG_fej);
            pFeature->Inited();

            vActiveFeatureIDs.push_back(id);
        }

        mvNewActiveFeatureIDs.clear();

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Jf);
        Hf = qr.inverse();
        HG = -Hf*JG;
        HP = -Hf*JP;

        tempM = LocalFactor.block(l1,l1,l2,l2);
        LocalFactor.block(l1,l1,l2,l2) = tempM*Hf;
        LocalFactor.block(l1,LG,l2,6) += tempM*HG;
        LocalFactor.block(l1,LC,l2,7) += tempM*HP;

        ComposeQR(l1, l2, LocalFactor);
    }

    if (nLostActiveFeatures>0)
    {
        int nDimOfSR = 3*(nActiveFeatures+nNewActiveFeatures)+9+7+nDimOfWinSR;
        Eigen::MatrixXd tempM = LocalFactor.bottomRightCorner(nDimOfSR, nDimOfSR+1);
        LocalFactor.swap(tempM);

        for (const int& id : mvLostActiveFeatureIDs)
            mFeatures.at(id)->Marginalized();

        mvLostActiveFeatureIDs.clear();
    }
}
