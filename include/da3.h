#pragma once
#include "config.hpp"

typedef enum{
    VIDEOFEED_TEST = 0U,
    VIDEOFEED_CAM,
    VIDEOFEED_RTSP
}video_feed_e;

typedef enum{
    MODEL_DA3_SMALL = 0u,
    MODEL_DA3_BASE,

    MODEL__COUNT
}models_e;


struct onnx_ctx_t;


void DA3InitInfer(models_e model);

///> Infer from input without moving to CPU
void DA3InferInputHeadLess(float * input, onnx_ctx_t * ctx);

///> Infer from input moving to CPU
cv::Mat DA3InferInput(cv::Mat frame_cpu, cv::cuda::Stream cv_stream);

///> Infer thread, call this as a thread to start infering
void DA3InferThread(video_feed_e video_feed, models_e model);

///> Prepare cap video for different cases
void DA3PrepareVideo(cv::VideoCapture * cap, video_feed_e video_feed);

cv::cuda::GpuMat * DA3Infer(cv::Mat frame_cpu, cv::cuda::Stream cv_stream);
cv::Mat DA3Getframe(cv::cuda::Stream cv_stream);