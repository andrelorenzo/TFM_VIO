#pragma once
#include "config.hpp"

///> Exponential smoothing based on alpha [0..1]
cv::Mat CommonExpSmooth(cv::Mat &current, float alpha);

///> Mean moothing (rolling window), do not call it twice
cv::Mat CommonMeanSmooth(cv::Mat& current,cv::Mat* window, uint8_t size,uint8_t& idx, uint8_t& filled);
// --- Shift / unshift (mismo código, aplicarlo 2 veces revierte) ---
void shiftDFT(cv::Mat& img);
// --- Convertir a gris ---
cv::Mat toGray(const cv::Mat& frame);
// --- FFT complejo centrado (CV_32FC2) ---
cv::Mat fftComplexCentered(const cv::Mat& frame);
// --- Espectro para visualizar (CV_8U) ---
cv::Mat spectrumForDisplay(const cv::Mat& complexI);
// --- Quitar cruz central (sobre DFT complejo centrado) ---
cv::Mat removeCrossInDFT(const cv::Mat& complexI, int thickness, int keepRadius);
// --- IDFT a imagen espacial (CV_8U) ---
cv::Mat ifftToDisplayImage(cv::Mat complexI, cv::Size originalSize);
void processAndShow(const cv::Mat& output_frame);
cv::Mat anotatePointsOnScreen(const cv::Mat& frame, int nw, int nh);
cv::Vec2f calculateMoveDirection(cv::Mat frame, cv::Mat * outframe, bool debug);
