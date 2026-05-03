#include "lie_math.hpp"

const mat3 I = mat3::Identity();

mat3 skewMat(const vec3& v) {
    mat3 M;
    M <<  0.0,   -v.z(),  v.y(),
          v.z(),  0.0,   -v.x(),
         -v.y(),  v.x(),  0.0;
    return M;
}

mat3 diagSquare(const vec3& v) {
    return v.array().square().matrix().asDiagonal();
}

mat3 expSO3(const vec3& w, double sa) {
    const double theta = w.norm();
    const mat3 I = mat3::Identity();
    const mat3 W = skewMat(w);
    if (theta < sa) {
        Logger(WARN, "SMALLL ANGLE ON EXPSO3");
        return I + W + 0.5 * W * W;
    }

    // Standard right-handed SO(3) exponential used with body-frame gyro rates.
    return I + (std::sin(theta) * W / theta) + (((1.0 - std::cos(theta)) * W * W) / (theta * theta));
}

mat3 rightJacobianSO3(const vec3& w, double sa) {
    const double theta = w.norm();
    const mat3 I = mat3::Identity();
    const mat3 W = skewMat(w);
    if (theta < sa) {
        
        // Jder for small angle: I - 0.5 * W + (1/6) * W^2
        return I - 0.5 * W + (1.0 / 6.0) * W * W;
    }

    const double theta2 = theta * theta;
    const double theta3 = theta2 * theta;
    // Jder = I - ((1 - cos(theta)) * W / theta^2) + ((theta - sin(theta)) * W^2 / theta^3)
    return I - ((1.0 - std::cos(theta)) / theta2) * W + ((theta - std::sin(theta)) / theta3) * (W * W);
}

vec3 logSO3(const mat3& R) {
    const double cos_theta = std::clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0);
    const double theta = std::acos(cos_theta);

    if (theta < 1e-10)return vec3::Zero();
    const mat3 Omega = (0.5 * theta / std::sin(theta)) * (R - R.transpose());
    return vec3(Omega(2,1), Omega(0,2), Omega(1,0));
}


quat normalizeQ(const quat& q) {
    quat qn = q;
    const double n = qn.norm();
    if (n < 1e-12) {
        return quat::Identity();
    }
    qn.coeffs() /= n;
    if (qn.w() < 0.0) qn.coeffs() *= -1.0;
    return qn;
}

quat expQ(const vec3& w, double sa) {
    return normalizeQ(quat(expSO3(w, sa)));
}


vec3 quatToRpyRad(const quat& q) {
    const mat3 R = q.toRotationMatrix();
    const double roll = std::atan2(R(2, 1), R(2, 2));
    const double pitch = std::asin(-std::max(-1.0, std::min(1.0, R(2, 0))));
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return vec3(roll, pitch, yaw);
}

vec3 quatToCameraRpyRad(const quat& q) {
    const mat3 R = q.toRotationMatrix();

    // Virtual RH frame so standard ZYX extraction matches camera semantics:
    // roll about +z_cam, pitch about +x_cam, yaw about -y_cam.
    mat3 B;
    B << 0.0, -1.0,  0.0,
         0.0,  0.0, -1.0,
         1.0,  0.0,  0.0;

    const mat3 Rv = B.transpose() * R * B;
    const double roll = std::atan2(Rv(2, 1), Rv(2, 2));
    const double pitch_v = std::asin(-std::max(-1.0, std::min(1.0, Rv(2, 0))));
    const double yaw = std::atan2(Rv(1, 0), Rv(0, 0));
    return vec3(roll, -pitch_v, yaw);
}

quat quatFromAccel(const vec3& acc_body, const vec3& gv) {
    if (acc_body.norm() < 1e-9 || gv.norm() < 1e-9) {
        return quat::Identity();
    }

    return normalizeQ(quat::FromTwoVectors(acc_body.normalized(), gv.normalized()));
}


static vec4 computeFxSO3(double wn, double dt, double sa){
    double f1, f2, f3, f4;

    if(wn > sa){
        f1 = (wn*dt*std::cos(wn*dt) - std::sin(wn*dt)) / (wn*wn*wn);
        f2 = ((wn*dt)*(wn*dt) - 2.0 * std::cos(wn*dt) - 2*wn*dt*sin(wn*dt) + 2) / (2 * wn*wn*wn*wn); 
        f3 = (std::cos(wn*dt) - 1.0) / (wn*wn);
        f4 = (wn*dt - std::sin(wn*dt)) / (wn*wn*wn);
        
    }else{
        Logger(WARN, "SMALLL ANGLE ON JxSO3");
        f1 = -1.0 * dt*dt*dt / 3;
        f2 = dt*dt*dt*dt / 8;
        f3 = -0.5 * dt*dt;
        f4 = dt*dt*dt / 6;
        
    }
    return vec4(f1,f2,f3,f4);
}
mat3 computeJ1SO3(vec3 w, double dt, double sa){
    vec4 fx = computeFxSO3(w.norm(), dt, sa);
    return ((dt * I) + (fx[2] * skewMat(w)) + (fx[3] * skewMat(w) * skewMat(w)));
}
mat3 computeJ2SO3(vec3 w, double dt, double sa){
    vec4 fx = computeFxSO3(w.norm(), dt, sa);
    return ((dt*dt * I * 0.5) + fx[0] * skewMat(w) + (fx[1] * skewMat(w) * skewMat(w)));
}


Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& Mat, double epsilon){
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Mat, Eigen::ComputeFullU|Eigen::ComputeFullV);
    double tolerance = epsilon*std::max(Mat.cols(), Mat.rows())*svd.singularValues().array().abs()(0);
    return svd.matrixV()*(svd.singularValues().array().abs()>tolerance).select(svd.singularValues().array().inverse(),0).matrix().asDiagonal()*svd.matrixU().adjoint();
}

Eigen::Matrix3d cvMat33ToEigen(const cv::Mat& R_cv) {
    cv::Mat R64;
    R_cv.convertTo(R64, CV_64F);

    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_map( R64.ptr<double>() );

    return Eigen::Matrix3d(R_map);
}


Eigen::Vector3d cvVec3ToEigen(const cv::Mat& v_cv) {
    cv::Mat v64;
    v_cv.convertTo(v64, CV_64F);
    return Eigen::Vector3d( v64.at<double>(0, 0), v64.at<double>(1, 0), v64.at<double>(2, 0) );
}


vec4 QuatMul(const vec4& q1, const vec4& q2)
{
    Eigen::Matrix4d mat;

    mat(0,0) =  q1(3,0);
    mat(0,1) =  q1(2,0);
    mat(0,2) = -q1(1,0);
    mat(0,3) =  q1(0,0);

    mat(1,0) = -q1(2,0);
    mat(1,1) =  q1(3,0);
    mat(1,2) =  q1(0,0);
    mat(1,3) =  q1(1,0);

    mat(2,0) =  q1(1,0);
    mat(2,1) = -q1(0,0);
    mat(2,2) =  q1(3,0);
    mat(2,3) =  q1(2,0);

    mat(3,0) = -q1(0,0);
    mat(3,1) = -q1(1,0);
    mat(3,2) = -q1(2,0);
    mat(3,3) =  q1(3,0);

    vec4 q = mat*q2;

    q.normalize();

    if (q(3)<0)
        q *= -1;

    return q;
}


vec4 QuatInv(const vec4& q)
{
    vec4 q_inv;

    if (q(3)>0)
    {
        q_inv(0) = -q(0);
        q_inv(1) = -q(1);
        q_inv(2) = -q(2);
        q_inv(3) =  q(3);
    }
    else
    {
        q_inv(0) =  q(0);
        q_inv(1) =  q(1);
        q_inv(2) =  q(2);
        q_inv(3) = -q(3);
    }

    q_inv.normalize();

    return q_inv;
}



mat3 QuatToRot(const vec4& q)
{
    mat3 I = mat3::Identity();
    const vec3 v = q.head<3>();
    mat3 qx;
    qx << 0.0,   -v.z(),  v.y(),
          v.z(),   0.0,  -v.x(),
          -v.y(),   v.x(),  0.0;

    return I-2*q(3)*qx+2*qx*qx;;
}


vec4 RotToQuat(const mat3& R)
{
    double T = R.trace();

    vec4 q;

    if ((R(0,0)>T)&&(R(0,0)>R(1,1))&&(R(0,0)>R(2,2)))
    {
        q(0) = sqrt((1+(2*R(0,0))-T)/4);
        q(1) = (1/(4*q(0)))*(R(0,1)+R(1,0));
        q(2) = (1/(4*q(0)))*(R(0,2)+R(2,0));
        q(3) = (1/(4*q(0)))*(R(1,2)-R(2,1));
    }
    else if ((R(1,1)>T)&&(R(1,1)>R(0,0))&&(R(1,1)>R(2,2)))
    {
        q(1) = sqrt((1+(2*R(1,1))-T)/4);
        q(0) = (1/(4*q(1)))*(R(0,1)+R(1,0));
        q(2) = (1/(4*q(1)))*(R(1,2)+R(2,1));
        q(3) = (1/(4*q(1)))*(R(2,0)-R(0,2));
    }
    else if ((R(2,2)>T)&&(R(2,2)>R(0,0))&&(R(2,2)>R(1,1)))
    {
        q(2) = sqrt((1+(2*R(2,2))-T)/4);
        q(0) = (1/(4*q(2)))*(R(0,2)+R(2,0));
        q(1) = (1/(4*q(2)))*(R(1,2)+R(2,1));
        q(3) = (1/(4*q(2)))*(R(0,1)-R(1,0));
    }
    else
    {
        q(3) = sqrt((1+T)/4);
        q(0) = (1/(4*q(3)))*(R(1,2)-R(2,1));
        q(1) = (1/(4*q(3)))*(R(2,0)-R(0,2));
        q(2) = (1/(4*q(3)))*(R(0,1)-R(1,0));
    }

    q.normalize();

    if (q(3)<0)
        q *= -1;

    return q;
}

cv::Mat NormalizeTransform4x4(const cv::Mat& input)
{
    if (input.empty()) {
        return cv::Mat();
    }

    cv::Mat output = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat source;
    input.convertTo(source, CV_64F);

    if (source.rows >= 3 && source.cols >= 4) {
        source(cv::Range(0, 3), cv::Range(0, 4))
            .copyTo(output(cv::Range(0, 3), cv::Range(0, 4)));
        return output;
    }

    return cv::Mat();
}

cv::Mat InvertRigidTransform(const cv::Mat& input)
{
    cv::Mat T = NormalizeTransform4x4(input);
    if (T.empty()) {
        return cv::Mat();
    }

    cv::Mat R = T(cv::Range(0, 3), cv::Range(0, 3)).clone();
    cv::Mat t = T(cv::Range(0, 3), cv::Range(3, 4)).clone();
    cv::Mat R_inv = R.t();
    cv::Mat t_inv = -R_inv * t;

    cv::Mat output = cv::Mat::eye(4, 4, CV_64F);
    R_inv.copyTo(output(cv::Range(0, 3), cv::Range(0, 3)));
    t_inv.copyTo(output(cv::Range(0, 3), cv::Range(3, 4)));
    return output;
}
