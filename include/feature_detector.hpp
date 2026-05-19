#pragma once
#include <vector>

#include <opencv2/core/core.hpp>
#include "config.hpp"

class FeatureDetector
{
public:

    FeatureDetector(Config config);

    int DetectWithSubPix(const cv::Mat& im, const int nCorners, const int s, std::vector<cv::Point2f>& vCorners);

    int FindNewer(const std::vector<cv::Point2f>& vCorners, const std::vector<cv::Point2f>& vRefCorners, std::vector<cv::Point2f>& vNewCorners);

void DebugDrawGrid(const cv::Mat& image, cv::Mat& imOut) const;

void DebugDrawShiTomasi(const cv::Mat& image, cv::Mat& imOut) const;

void DebugDrawSubPix(const cv::Mat& image, const int nCorners, const int s, cv::Mat& imOut) const;

void DebugDrawCandidates(const cv::Mat& image, const std::vector<cv::Point2f>& vCandidates, const std::vector<cv::Point2f>& vRefCorners, const std::vector<cv::Point2f>& vNewCorners, cv::Mat& imOut) const;

void DebugDrawSelection(const cv::Mat& image, const std::vector<cv::Point2f>& vCandidates, const std::vector<cv::Point2f>& vRefCorners, const std::vector<cv::Point2f>& vNewCorners, cv::Mat& imOut) const;

void DebugDrawCellOccupancy(const cv::Mat& image, const std::vector<cv::Point2f>& vRefCorners, const std::vector<cv::Point2f>& vNewCorners, cv::Mat& imOut) const;

private:

    void ChessGrid(const std::vector<cv::Point2f>& vCorners);

private:

    int mnImageCols;
    int mnImageRows;

    int mnBlocks;
    int mnGridCols;
    int mnGridRows;

    int mnOffsetX;
    int mnOffsetY;

    float mnBlockSizeX;
    float mnBlockSizeY;

    float mnMinDistance;
    float mnQualityLevel;

    int mnMaxFeatsPerBlock;

    // Chess grid of features
    std::vector<std::vector<cv::Point2f> > mvvGrid;
};

