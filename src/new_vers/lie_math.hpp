#pragma once
#include "config.hpp"


mat3 skewMat(const vec3& v);                                ///> W^ antisimetrical Matrix
mat3 diagSquare(const vec3& v);                             ///> create diagonal Matrix
mat3 expSO3(const vec3& w, double sa);                      ///> Exponential for SO3
mat3 rightJacobianSO3(const vec3& w, double sa);            ///> Right Jacobian in SO3
vec3 logSO3(const mat3& R);                                 ///> Logaritmic in SO3
quat normalizeQ(const quat& q);                             ///> Normalization for q
quat expQ(const vec3& w, double sa);                        ///> q^e
vec3 quatToRpyRad(const quat& q);                           ///> q -> rpy
vec3 quatToCameraRpyRad(const quat& q);                     ///> q -> camera rpy (roll:z, pitch:x, yaw:-y)
quat quatFromAccel(const vec3& acc_body, const vec3& gv);   ///> (a, gv_rest) -> q
mat3 computeJ1SO3(vec3 w, double dt, double sa);
mat3 computeJ2SO3(vec3 w, double dt, double sa);
// Eigen::MatrixXf pseudoInverse(const Eigen::MatrixXf& Mat, float epsilon=std::numeric_limits<float>::epsilon());
Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& Mat, double epsilon=std::numeric_limits<double>::epsilon());
