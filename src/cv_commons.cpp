#include "cv_commons.h"

cv::Mat CommonExpSmooth(cv::Mat& current, float alpha)
{
    CV_Assert(alpha >= 0.0f && alpha <= 1.0f);
    static cv::Mat past;

    if (current.empty()) {
        return past;
    }

    if (past.empty() || past.size() != current.size() || past.type() != current.type()) {
        past = current.clone();
        return past;
    }

    cv::Mat next;
    cv::addWeighted(past, alpha, current, 1.0 - alpha, 0.0, next);

    past = next;
    return past;
}


cv::Mat CommonMeanSmooth(cv::Mat& current,cv::Mat* window, uint8_t size,uint8_t& idx, uint8_t& filled){
    CV_Assert(window != nullptr);
    CV_Assert(size > 0);

    if (current.empty()) {
        if (filled == 0) return cv::Mat();
        cv::Mat acc = window[0].clone();
        for (uint8_t i = 1; i < filled; ++i) acc += window[i];
        acc.convertTo(acc, acc.type(), 1.0 / double(filled));
        return acc;
    }

    if (filled > 0) {
        if (window[0].empty() || window[0].size() != current.size() || window[0].type() != current.type()){
            for (uint8_t i = 0; i < size; ++i) window[i].release();
            idx = 0;
            filled = 0;
        }
    }

    window[idx] = current.clone();
    idx = uint8_t((idx + 1) % size);
    if (filled < size) ++filled;

    cv::Mat acc;
    uint8_t first = 0;
    while (first < filled && window[first].empty()) ++first;
    if (first == filled) return cv::Mat();

    acc = window[first].clone();
    for (uint8_t i = first + 1; i < filled; ++i) {
        if (!window[i].empty()) acc += window[i];
    }

    acc.convertTo(acc, acc.type(), 1.0 / double(filled));
    return acc;
}


// --- Shift / unshift (mismo código, aplicarlo 2 veces revierte) ---
void shiftDFT(cv::Mat& img)
{
    // img: CV_32FC2
    img = img(cv::Rect(0, 0, img.cols & -2, img.rows & -2)); // asegurar par
    int cx = img.cols / 2;
    int cy = img.rows / 2;

    cv::Mat q0(img, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(img, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(img, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(img, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

// --- Convertir a gris ---
cv::Mat toGray(const cv::Mat& frame)
{
    CV_Assert(!frame.empty());

    cv::Mat gray8;
    if (frame.channels() == 1) gray8 = frame;
    else if (frame.channels() == 3) cv::cvtColor(frame, gray8, cv::COLOR_BGR2GRAY);
    else if (frame.channels() == 4) cv::cvtColor(frame, gray8, cv::COLOR_BGRA2GRAY);
    else {
        std::vector<cv::Mat> ch;
        cv::split(frame, ch);
        gray8 = ch[0];
    }

    cv::Mat gray32;

    gray8.convertTo(gray32, CV_32FC1, 1.0/255.0);
    return gray32;
}

// --- FFT complejo centrado (CV_32FC2) ---
cv::Mat fftComplexCentered(const cv::Mat& frame)
{
    cv::Mat gray = toGray(frame);

    cv::Mat padded;
    int optRows = cv::getOptimalDFTSize(gray.rows);
    int optCols = cv::getOptimalDFTSize(gray.cols);
    cv::copyMakeBorder(gray, padded, 0, optRows - gray.rows, 0, optCols - gray.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat padded32f;
    padded.convertTo(padded32f, CV_32F);

    cv::Mat planes[] = { padded32f, cv::Mat::zeros(padded32f.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    cv::dft(complexI, complexI);
    shiftDFT(complexI); // centrar DC
    return complexI;    // CV_32FC2
}

// --- Espectro para visualizar (CV_8U) ---
cv::Mat spectrumForDisplay(const cv::Mat& complexI)
{
    CV_Assert(complexI.type() == CV_32FC2);

    std::vector<cv::Mat> p(2);
    cv::split(complexI, p);

    cv::Mat mag;
    cv::magnitude(p[0], p[1], mag); // CV_32F, 1 canal
    mag += 1.0f;
    cv::log(mag, mag);

    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);

    cv::Mat mag8u;
    mag.convertTo(mag8u, CV_8U);
    return mag8u;
}

// --- Quitar cruz central (sobre DFT complejo centrado) ---
cv::Mat removeCrossInDFT(const cv::Mat& complexI, int thickness, int keepRadius)
{
    CV_Assert(!complexI.empty() && complexI.type() == CV_32FC2);

    cv::Mat mask(complexI.size(), CV_32F, cv::Scalar(1.0f));
    int cx = mask.cols / 2;
    int cy = mask.rows / 2;

    thickness = std::max(0, thickness);
    keepRadius = std::max(0, keepRadius);

    // Bandas (horizontal + vertical)
    int x0 = std::max(0, cx - thickness);
    int x1 = std::min(mask.cols, cx + thickness + 1);
    int y0 = std::max(0, cy - thickness);
    int y1 = std::min(mask.rows, cy + thickness + 1);

    mask(cv::Rect(x0, 0, x1 - x0, mask.rows)).setTo(0.0f);
    mask(cv::Rect(0, y0, mask.cols, y1 - y0)).setTo(0.0f);

    // Mantener un disco en el centro (no tocar bajas frecuencias)
    cv::circle(mask, {cx, cy}, keepRadius, cv::Scalar(1.0f), -1);

    // Aplicar máscara a real e imag
    std::vector<cv::Mat> planes(2);
    cv::split(complexI, planes);
    planes[0] = planes[0].mul(mask);
    planes[1] = planes[1].mul(mask);

    cv::Mat filtered;
    cv::merge(planes, filtered);
    return filtered; // CV_32FC2
}

// --- IDFT a imagen espacial (CV_8U) ---
cv::Mat ifftToDisplayImage(cv::Mat complexI, cv::Size originalSize)
{
    CV_Assert(!complexI.empty() && complexI.type() == CV_32FC2);

    shiftDFT(complexI); // des-centrar

    cv::Mat inv32f;
    cv::idft(complexI, inv32f, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    // recortar al tamaño original (antes del padding)
    inv32f = inv32f(cv::Rect(0, 0, originalSize.width, originalSize.height)).clone();

    // normalizar a 8U para mostrar
    cv::Mat inv8u;
    cv::normalize(inv32f, inv32f, 0, 255, cv::NORM_MINMAX);
    inv32f.convertTo(inv8u, CV_8U);

    return inv8u;
}

cv::Mat notchMaskFromPeaks(const cv::Mat& mag, float thresh, int notchRadius, int axisBand)
{
    // mag: magnitud centrada (CV_32F)
    cv::Mat mask(mag.size(), CV_32F, cv::Scalar(1.0f));
    int cx = mag.cols/2, cy = mag.rows/2;

    for (int y = 0; y < mag.rows; ++y) {
        for (int x = 0; x < mag.cols; ++x) {
            bool nearAxis = (std::abs(x - cx) <= axisBand) || (std::abs(y - cy) <= axisBand);
            if (!nearAxis) continue;

            if (mag.at<float>(y,x) > thresh) {
                // notch en (x,y)
                cv::circle(mask, {x,y}, notchRadius, cv::Scalar(0.0f), -1);

                // notch simétrico respecto al centro
                int xs = 2*cx - x;
                int ys = 2*cy - y;
                if (0 <= xs && xs < mag.cols && 0 <= ys && ys < mag.rows)
                    cv::circle(mask, {xs,ys}, notchRadius, cv::Scalar(0.0f), -1);
            }
        }
    }

    // NO tocar el DC (centro)
    cv::circle(mask, {cx,cy}, notchRadius*2, cv::Scalar(1.0f), -1);
    return mask;
}
cv::Mat removeCrossByNotchesFromPeaks(const cv::Mat& complexI,float kStd = 4.0f,int notchRadius = 6,int axisBand = 4){
    CV_Assert(!complexI.empty() && complexI.type() == CV_32FC2);

    // 1) Magnitud lineal (CV_32F, 1 canal)
    std::vector<cv::Mat> p(2);
    cv::split(complexI, p);

    cv::Mat mag;
    cv::magnitude(p[0], p[1], mag);

    // 2) Umbral automático: mean + kStd * std
    cv::Scalar mean, stddev;
    cv::meanStdDev(mag, mean, stddev);
    float thresh = static_cast<float>(mean[0] + kStd * stddev[0]);

    // 3) Máscara notch (1 canal) usando tu función
    cv::Mat mask = notchMaskFromPeaks(mag, thresh, notchRadius, axisBand);

    // 4) Aplicar máscara al complejo (real e imag)
    p[0] = p[0].mul(mask);
    p[1] = p[1].mul(mask);

    cv::Mat filtered;
    cv::merge(p, filtered);
    return filtered; // CV_32FC2
}


void processAndShow(const cv::Mat& output_frame)
{
    CV_Assert(!output_frame.empty());

    cv::Mat inputGray = toGray(output_frame);
    cv::imshow("1) Input (gray)", inputGray);

    cv::Mat complex0 = fftComplexCentered(output_frame);
    cv::imshow("2) Spectrum (before)", spectrumForDisplay(complex0));

    // cv::Mat complex1 = removeCrossInDFT(complex0, /*thickness*/2, /*keepRadius*/10);
    cv::Mat complex1 = removeCrossByNotchesFromPeaks(complex0, /*kStd*/3.0f, /*notchRadius*/4, /*axisBand*/8);
    cv::imshow("3) Spectrum (after)", spectrumForDisplay(complex1));

    cv::Mat filteredSpatial = ifftToDisplayImage(complex1, inputGray.size());
    cv::imshow("4) Output (filtered)", filteredSpatial);
}


cv::Mat anotatePointsOnScreen(const cv::Mat& frame, int nw, int nh)
{
    
    CV_Assert(!frame.empty());
    CV_Assert(nw > 0 && nh > 0);
    CV_Assert(frame.type() == CV_32FC1);   // porque lees float (como en tu código)

    const int w = frame.cols;
    const int h = frame.rows;

    // salida: clon para dibujar (convertimos a BGR para que el texto/círculos se vean)
    cv::Mat output;
    cv::Mat frame_u8, frame_norm;

    // normalizamos para visualizar; si no quieres normalizar, puedes hacer output = frame.clone();
    cv::normalize(frame, frame_norm, 0, 255, cv::NORM_MINMAX);
    frame_norm.convertTo(frame_u8, CV_8U);
    cv::cvtColor(frame_u8, output, cv::COLOR_GRAY2BGR);

    // paso de la rejilla (centro de cada celda)
    const float stepX = (float)w / (float)nw;
    const float stepY = (float)h / (float)nh;

    for (int gy = 0; gy < nh; ++gy) {
        for (int gx = 0; gx < nw; ++gx) {

            int x = (int)((gx + 0.5f) * stepX);
            int y = (int)((gy + 0.5f) * stepY);

            // clamp para no salir
            x = std::max(0, std::min(w - 1, x));
            y = std::max(0, std::min(h - 1, y));

            float d = frame.at<float>(y, x);

            cv::Point pos(x, y);
            cv::circle(output, pos, 3, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);

            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.2f", d);

            cv::Point textPos = pos + cv::Point(5, -5);
            if (textPos.y < 12) textPos.y = pos.y + 12;
            if (textPos.x > w - 80) textPos.x = w - 80;

            cv::putText(output, buf, textPos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

    return output;
}


cv::Vec2f calculateMoveDirection(cv::Mat frame, cv::Mat * outframe, bool debug){
    double min, max;
    cv::minMaxLoc(frame, &min, &max, nullptr, nullptr);

    cv::patchNaNs(frame, 0.0);
    // min is close, max is far

    *outframe = frame.clone();
    cv::Mat dx, dy;
    cv::Sobel(frame, dx, CV_32F, 1, 0, 3);
    cv::Sobel(frame, dy, CV_32F, 0, 1, 3);

    cv::Scalar mean_dx = cv::mean(dx);
    cv::Scalar mean_dy = cv::mean(dy);

    
    if(debug){

        cv::Point2f dir(mean_dx[0] * 10000.0f, mean_dy[0] * 10000.0f);
        cv::Point2f center(frame.cols / 2.0f, frame.rows / 2.0f);

        float norm = std::sqrt(dir.x * dir.x + dir.y * dir.y);

        // printf("Mag: %f (%f, %f)\n", norm, dir.x, dir.y);
        // cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::arrowedLine(*outframe, center, center + (dir), cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
    
    return cv::Vec2f(mean_dx[0],mean_dy[0]);
}