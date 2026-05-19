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


static int ClampInt(const int v, const int lo, const int hi)
{
    return std::max(lo, std::min(v, hi));
}

static void MakeDebugBgr(const cv::Mat& image, cv::Mat& imOut)
{
    if (image.empty()) {
        imOut = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(imOut, "empty image", cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0,0,255), 2, cv::LINE_AA);
        return;
    }

    if (image.channels()==1) cv::cvtColor(image, imOut, cv::COLOR_GRAY2BGR);
    else if (image.channels()==3) image.copyTo(imOut);
    else if (image.channels()==4) cv::cvtColor(image, imOut, cv::COLOR_BGRA2BGR);
    else {
        imOut = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
        cv::putText(imOut, "unsupported image", cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0,0,255), 2, cv::LINE_AA);
    }
}

static bool ContainsPoint(const std::vector<cv::Point2f>& vPts, const cv::Point2f& pt)
{
    for (const cv::Point2f& p : vPts) {
        if (cv::norm(p-pt)<0.75f) return true;
    }

    return false;
}

static void DrawBaseGrid(cv::Mat& imOut, const int nGridCols, const int nGridRows, const int nOffsetX, const int nOffsetY, const float nBlockSizeX, const float nBlockSizeY, const float nMinDistance, const bool bDrawMargin)
{
    if (imOut.empty()) return;

    const cv::Scalar gridColor(120,120,120);
    const cv::Scalar borderColor(255,255,255);
    const cv::Scalar marginColor(80,180,80);

    int x0 = ClampInt(cvRound(nOffsetX), 0, imOut.cols-1);
    int y0 = ClampInt(cvRound(nOffsetY), 0, imOut.rows-1);
    int x1 = ClampInt(cvRound(nOffsetX+nGridCols*nBlockSizeX), 0, imOut.cols-1);
    int y1 = ClampInt(cvRound(nOffsetY+nGridRows*nBlockSizeY), 0, imOut.rows-1);

    cv::rectangle(imOut, cv::Point(x0,y0), cv::Point(x1,y1), borderColor, 2, cv::LINE_AA);

    for (int c=0; c<=nGridCols; ++c) {
        int x = ClampInt(cvRound(nOffsetX+c*nBlockSizeX), 0, imOut.cols-1);
        cv::line(imOut, cv::Point(x,y0), cv::Point(x,y1), gridColor, 1, cv::LINE_AA);
    }

    for (int r=0; r<=nGridRows; ++r) {
        int y = ClampInt(cvRound(nOffsetY+r*nBlockSizeY), 0, imOut.rows-1);
        cv::line(imOut, cv::Point(x0,y), cv::Point(x1,y), gridColor, 1, cv::LINE_AA);
    }

    if (!bDrawMargin) return;

    int m = std::max(1, cvRound(nMinDistance));

    for (int r=0; r<nGridRows; ++r) {
        for (int c=0; c<nGridCols; ++c) {
            int xl = ClampInt(cvRound(nOffsetX+c*nBlockSizeX+m), 0, imOut.cols-1);
            int xr = ClampInt(cvRound(nOffsetX+(c+1)*nBlockSizeX-m), 0, imOut.cols-1);
            int yt = ClampInt(cvRound(nOffsetY+r*nBlockSizeY+m), 0, imOut.rows-1);
            int yb = ClampInt(cvRound(nOffsetY+(r+1)*nBlockSizeY-m), 0, imOut.rows-1);

            if (xl<xr && yt<yb) cv::rectangle(imOut, cv::Point(xl,yt), cv::Point(xr,yb), marginColor, 1, cv::LINE_AA);
        }
    }
}

void FeatureDetector::DebugDrawGrid(const cv::Mat& image, cv::Mat& imOut) const
{
    MakeDebugBgr(image, imOut);
    DrawBaseGrid(imOut, mnGridCols, mnGridRows, mnOffsetX, mnOffsetY, mnBlockSizeX, mnBlockSizeY, mnMinDistance, true);

    cv::putText(imOut, "Rejilla de distribucion espacial", cv::Point(15,30), cv::FONT_HERSHEY_PLAIN, 1.4, cv::Scalar(64,255,64), 2, cv::LINE_AA);
    cv::putText(imOut, "Rectangulos verdes: zona valida dentro de cada celda", cv::Point(15,55), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(64,255,64), 1, cv::LINE_AA);
}

void FeatureDetector::DebugDrawShiTomasi(const cv::Mat& image, cv::Mat& imOut) const
{
    if (image.empty()) {
        MakeDebugBgr(image, imOut);
        return;
    }

    cv::Mat gray;

    if (image.channels()==1) gray = image.clone();
    else if (image.channels()==3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels()==4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else {
        MakeDebugBgr(image, imOut);
        return;
    }

    cv::Mat response;
    cv::Mat response8u;

    cv::cornerMinEigenVal(gray, response, 3, 3);
    cv::normalize(response, response8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(response8u, imOut, cv::COLORMAP_JET);

    cv::putText(imOut, "Respuesta Shi-Tomasi: min(lambda1, lambda2)", cv::Point(15,30), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255,255,255), 2, cv::LINE_AA);
}

void FeatureDetector::DebugDrawSubPix(const cv::Mat& image, const int nCorners, const int s, cv::Mat& imOut) const
{
    if (image.empty()) {
        MakeDebugBgr(image, imOut);
        return;
    }

    cv::Mat gray;

    if (image.channels()==1) gray = image.clone();
    else if (image.channels()==3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels()==4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else {
        MakeDebugBgr(image, imOut);
        return;
    }

    std::vector<cv::Point2f> vRawCorners;
    std::vector<cv::Point2f> vSubPixCorners;

    cv::goodFeaturesToTrack(gray, vRawCorners, nCorners, mnQualityLevel, s*mnMinDistance);
    vSubPixCorners = vRawCorners;

    if (!vSubPixCorners.empty()) {
        cv::Size subPixWinSize(std::floor(.5*mnMinDistance), std::floor(.5*mnMinDistance));
        cv::Size subPixZeroZone(-1,-1);
        cv::TermCriteria subPixCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-2);
        cv::cornerSubPix(gray, vSubPixCorners, subPixWinSize, subPixZeroZone, subPixCriteria);
    }

    MakeDebugBgr(image, imOut);

    for (const cv::Point2f& pt : vRawCorners) cv::drawMarker(imOut, pt, cv::Scalar(180,180,180), cv::MARKER_CROSS, 8, 1, cv::LINE_AA);

    int N = std::min((int)vRawCorners.size(), (int)vSubPixCorners.size());

    for (int i=0; i<N; ++i) {
        cv::Point2f p0 = vRawCorners.at(i);
        cv::Point2f p1 = vSubPixCorners.at(i);
        if (cv::norm(p0-p1)>0.05f) cv::line(imOut, p0, p1, cv::Scalar(0,255,255), 1, cv::LINE_AA);
        cv::circle(imOut, p1, 3, cv::Scalar(64,255,64), -1, cv::LINE_AA);
    }

    cv::putText(imOut, "Gris: deteccion inicial | Verde: refinamiento subpixel", cv::Point(15,30), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0,0,0), 2, cv::LINE_AA);
}

void FeatureDetector::DebugDrawCandidates(const cv::Mat& image, const std::vector<cv::Point2f>& vCandidates, const std::vector<cv::Point2f>& vRefCorners, const std::vector<cv::Point2f>& vNewCorners, cv::Mat& imOut) const
{
    MakeDebugBgr(image, imOut);
    DrawBaseGrid(imOut, mnGridCols, mnGridRows, mnOffsetX, mnOffsetY, mnBlockSizeX, mnBlockSizeY, mnMinDistance, false);

    for (const cv::Point2f& pt : vCandidates) cv::circle(imOut, pt, 2, cv::Scalar(180,180,180), -1, cv::LINE_AA);

    for (const cv::Point2f& pt : vRefCorners) {
        cv::circle(imOut, pt, std::max(1, cvRound(mnMinDistance)), cv::Scalar(255,64,64), 1, cv::LINE_AA);
        cv::circle(imOut, pt, 3, cv::Scalar(255,64,64), -1, cv::LINE_AA);
    }

    for (const cv::Point2f& pt : vNewCorners) {
        cv::circle(imOut, pt, 6, cv::Scalar(0,0,0), 2, cv::LINE_AA);
        cv::circle(imOut, pt, 4, cv::Scalar(64,255,64), -1, cv::LINE_AA);
    }

    cv::putText(imOut, "Gris: candidatos | Azul: activas | Verde: nuevas aceptadas", cv::Point(15,30), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0,0,0), 2, cv::LINE_AA);
}

void FeatureDetector::DebugDrawSelection(const cv::Mat& image, const std::vector<cv::Point2f>& vCandidates, const std::vector<cv::Point2f>& vRefCorners, const std::vector<cv::Point2f>& vNewCorners, cv::Mat& imOut) const
{
    MakeDebugBgr(image, imOut);
    DrawBaseGrid(imOut, mnGridCols, mnGridRows, mnOffsetX, mnOffsetY, mnBlockSizeX, mnBlockSizeY, mnMinDistance, true);

    for (const cv::Point2f& pt : vRefCorners) {
        cv::circle(imOut, pt, std::max(1, cvRound(mnMinDistance)), cv::Scalar(255,64,64), 1, cv::LINE_AA);
        cv::circle(imOut, pt, 3, cv::Scalar(255,64,64), -1, cv::LINE_AA);
    }

    int nAccepted = 0;
    int nRejected = 0;

    for (const cv::Point2f& pt : vCandidates) {
        if (ContainsPoint(vNewCorners, pt)) {
            cv::circle(imOut, pt, 5, cv::Scalar(64,255,64), -1, cv::LINE_AA);
            nAccepted++;
        } else {
            cv::drawMarker(imOut, pt, cv::Scalar(64,64,255), cv::MARKER_TILTED_CROSS, 9, 1, cv::LINE_AA);
            nRejected++;
        }
    }

    for (const cv::Point2f& pt : vNewCorners) cv::circle(imOut, pt, 7, cv::Scalar(0,0,0), 1, cv::LINE_AA);

    cv::rectangle(imOut, cv::Point(8,8), cv::Point(620,92), cv::Scalar(0,0,0), -1);
    cv::putText(imOut, "Azul: features activas con radio d_min", cv::Point(18,30), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,64,64), 1, cv::LINE_AA);
    cv::putText(imOut, "Verde: nuevos puntos aceptados", cv::Point(18,50), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(64,255,64), 1, cv::LINE_AA);
    cv::putText(imOut, "Rojo: candidatos descartados por rejilla/distancia/ocupacion", cv::Point(18,70), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(64,64,255), 1, cv::LINE_AA);
    cv::putText(imOut, "aceptados=" + std::to_string(nAccepted) + " rechazados=" + std::to_string(nRejected), cv::Point(18,90), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}

void FeatureDetector::DebugDrawCellOccupancy(const cv::Mat& image, const std::vector<cv::Point2f>& vRefCorners, const std::vector<cv::Point2f>& vNewCorners, cv::Mat& imOut) const
{
    MakeDebugBgr(image, imOut);
    DrawBaseGrid(imOut, mnGridCols, mnGridRows, mnOffsetX, mnOffsetY, mnBlockSizeX, mnBlockSizeY, mnMinDistance, false);

    std::vector<int> vOccupancy(mnBlocks, 0);

    for (const cv::Point2f& pt : vRefCorners) {
        if (pt.x<=mnOffsetX || pt.y<=mnOffsetY || pt.x>=(mnImageCols-mnOffsetX) || pt.y>=(mnImageRows-mnOffsetY)) continue;
        int col = (pt.x-mnOffsetX)/mnBlockSizeX;
        int row = (pt.y-mnOffsetY)/mnBlockSizeY;
        int idx = row*mnGridCols+col;
        if (idx>=0 && idx<(int)vOccupancy.size()) vOccupancy.at(idx)++;
    }

    for (const cv::Point2f& pt : vNewCorners) {
        if (pt.x<=mnOffsetX || pt.y<=mnOffsetY || pt.x>=(mnImageCols-mnOffsetX) || pt.y>=(mnImageRows-mnOffsetY)) continue;
        int col = (pt.x-mnOffsetX)/mnBlockSizeX;
        int row = (pt.y-mnOffsetY)/mnBlockSizeY;
        int idx = row*mnGridCols+col;
        if (idx>=0 && idx<(int)vOccupancy.size()) vOccupancy.at(idx)++;
    }

    for (int r=0; r<mnGridRows; ++r) {
        for (int c=0; c<mnGridCols; ++c) {
            int idx = r*mnGridCols+c;
            int x = cvRound(mnOffsetX+c*mnBlockSizeX+6);
            int y = cvRound(mnOffsetY+r*mnBlockSizeY+20);
            cv::putText(imOut, std::to_string(vOccupancy.at(idx)) + "/" + std::to_string(mnMaxFeatsPerBlock), cv::Point(x,y), cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(0,0,0), 1, cv::LINE_AA);
        }
    }

    for (const cv::Point2f& pt : vRefCorners) cv::circle(imOut, pt, 3, cv::Scalar(255,64,64), -1, cv::LINE_AA);
    for (const cv::Point2f& pt : vNewCorners) cv::circle(imOut, pt, 5, cv::Scalar(64,255,64), -1, cv::LINE_AA);

    cv::putText(imOut, "Ocupacion por celda: n / max", cv::Point(15,30), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(64,255,64), 2, cv::LINE_AA);
}