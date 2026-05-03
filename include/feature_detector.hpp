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

