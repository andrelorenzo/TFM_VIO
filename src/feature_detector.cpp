#include <cassert>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "feature_detector.hpp"



FeatureDetector::FeatureDetector(Config config){
    mnImageCols = config.cam.width;
    mnImageRows = config.cam.height;
    if (mnImageCols <= 0 || mnImageRows <= 0) {
        Logger(WARN, "FeatureDetector received invalid image size (%d x %d); using fallback 1280x720", mnImageCols, mnImageRows);
        mnImageCols = 1280;
        mnImageRows = 720;
    }

    mnMinDistance = config.vio.feat_mindist;
    mnQualityLevel = config.vio.feat_quat;

    mnBlockSizeX = config.vio.feat_bsizex;
    mnBlockSizeY = config.vio.feat_bsizey;
    if (mnBlockSizeX <= 0.0f || mnBlockSizeY <= 0.0f) {
        Logger(WARN, "FeatureDetector received invalid block size (%.3f x %.3f); using fallback 150x120", mnBlockSizeX, mnBlockSizeY);
        mnBlockSizeX = 150.0f;
        mnBlockSizeY = 120.0f;
    }
    
    
    
    mnGridCols = std::max(1, static_cast<int>(mnImageCols/mnBlockSizeX));
    mnGridRows = std::max(1, static_cast<int>(mnImageRows/mnBlockSizeY));
    
    mnOffsetX = .5*(mnImageCols-mnGridCols*mnBlockSizeX);
    mnOffsetY = .5*(mnImageRows-mnGridRows*mnBlockSizeY);
    
    mnBlocks = std::max(1, mnGridCols*mnGridRows);
    mnMaxFeatsPerBlock = std::max(1, config.vio.feat_max/mnBlocks);

    Logger(INFO,
           "FeatureDetector: image=%dx%d block=[%.3f %.3f] grid=%dx%d blocks=%d max_feat_block=%d",
           mnImageCols,
           mnImageRows,
           mnBlockSizeX,
           mnBlockSizeY,
           mnGridCols,
           mnGridRows,
           mnBlocks,
           mnMaxFeatsPerBlock);

    mvvGrid.resize(mnBlocks);
}


int FeatureDetector::DetectWithSubPix(const cv::Mat& im, const int nCorners, const int s, std::vector<cv::Point2f>& vCorners){
    vCorners.clear();
    vCorners.reserve(nCorners);

    cv::goodFeaturesToTrack(im, vCorners, nCorners, mnQualityLevel, s*mnMinDistance);

    if (!vCorners.empty())
    {
        cv::Size subPixWinSize(std::floor(.5*mnMinDistance),std::floor(.5*mnMinDistance));
        cv::Size subPixZeroZone(-1,-1);
        cv::TermCriteria subPixCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-2);
        cv::cornerSubPix(im, vCorners, subPixWinSize, subPixZeroZone, subPixCriteria);
    }

    return (int)vCorners.size();
}


void FeatureDetector::ChessGrid(const std::vector<cv::Point2f>& vCorners)
{
    mvvGrid.clear();
    mvvGrid.resize(mnBlocks);

    for (const cv::Point2f& pt : vCorners)
    {
        if (pt.x<=mnOffsetX || pt.y<=mnOffsetY || pt.x>=(mnImageCols-mnOffsetX) || pt.y>=(mnImageRows-mnOffsetY))
            continue;

        int col = (pt.x-mnOffsetX)/mnBlockSizeX;
        int row = (pt.y-mnOffsetY)/mnBlockSizeY;
        assert((col>=0 && col<mnGridCols) && (row>=0 && row<mnGridRows));

        int nBlockIdx = row*mnGridCols+col;
        mvvGrid.at(nBlockIdx).emplace_back(pt);
    }
}


int FeatureDetector::FindNewer(const std::vector<cv::Point2f>& vCorners, 
                               const std::vector<cv::Point2f>& vRefCorners, 
                               std::vector<cv::Point2f>& vNewCorners)
{
    ChessGrid(vRefCorners);

    for (const cv::Point2f& pt : vCorners)
    {
        if (pt.x<=mnOffsetX || pt.y<=mnOffsetY || pt.x>=(mnImageCols-mnOffsetX) || pt.y>=(mnImageRows-mnOffsetY))
            continue;

        int col = (pt.x-mnOffsetX)/mnBlockSizeX;
        int row = (pt.y-mnOffsetY)/mnBlockSizeY;
        assert((col>=0 && col<mnGridCols) && (row>=0 && row<mnGridRows));

        float xl = col*mnBlockSizeX+mnOffsetX;
        float xr = xl+mnBlockSizeX;
        float yt = row*mnBlockSizeY+mnOffsetY;
        float yb = yt+mnBlockSizeY;

        if (fabs(pt.x-xl)<mnMinDistance || fabs(pt.x-xr)<mnMinDistance || fabs(pt.y-yt)<mnMinDistance || fabs(pt.y-yb)<mnMinDistance)
            continue;

        int nBlockIdx = row*mnGridCols+col;

        if ((float)mvvGrid.at(nBlockIdx).size()<.75*mnMaxFeatsPerBlock)
        {
            if (!mvvGrid.at(nBlockIdx).empty())
            {
                int cnt = 0;

                for (const cv::Point2f& bpt : mvvGrid.at(nBlockIdx))
                {
                    if (cv::norm(pt-bpt)>mnMinDistance)
                        cnt++;
                    else
                        break;
                }

                if (cnt==(int)mvvGrid.at(nBlockIdx).size())
                {
                    vNewCorners.push_back(pt);
                    mvvGrid.at(nBlockIdx).push_back(pt);
                }
            }
            else
            {
                vNewCorners.push_back(pt);
                mvvGrid.at(nBlockIdx).push_back(pt);
            }
        }
    }

    return (int)vNewCorners.size();
}

