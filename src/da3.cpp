#include "da3.h"

#define PARSER_IMP
#include "seconds/file_parser.h"
#include "preprocess.h"
#include "cv_commons.h"

#include <cuda_runtime.h>
#include <memory>

#include "onnxruntime_cxx_api.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#define LOGGER_IMP
#include "seconds/logger.h"

struct onnx_ctx_t{
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name;
    std::array<int64_t, 4> input_shape{1, 3, MODEL_HEIGHT, MODEL_WIDTH};
    Ort::MemoryInfo cuda_mem{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    Ort::MemoryInfo cpu_mem  = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::unique_ptr<Ort::IoBinding> binding;
    float * input = nullptr;
};

const char * modelsf[MODEL__COUNT] = {
    "../../models/depth_anything/DA3METRIC-SMALL.onnx",
    "../../models/depth_anything/DA3METRIC-BASE.onnx"
};
static bool show_video = true;
static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

onnx_ctx_t da3_ctx;

static inline double NowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void PrintPerf(const char* tag, const std::vector<double>& ms_samples) {
    if (ms_samples.empty()) return;
    double sum = 0.0;
    double mn = ms_samples[0], mx = ms_samples[0];
    for (double v : ms_samples) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    double avg = sum / ms_samples.size();
    double fps = 1000.0 / avg;
    printf("\033[2J\033[1;1H");
    Logger(DEBUG, "[%s] frames=%zu | avg=%.3f ms | min=%.3f ms | max=%.3f ms | FPS=%.2f\n", tag, ms_samples.size(), avg, mn, mx, fps);
}

void DA3InitInfer(models_e model){
    da3_ctx.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "da3");
    const char * model_file = modelsf[model];

    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Logger(DEBUG, "Setting Optimization to ALL");

    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    cuda_opts.arena_extend_strategy = 0;
    cuda_opts.gpu_mem_limit = SIZE_MAX;
    cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_opts.do_copy_in_default_stream = 1;
    so.AppendExecutionProvider_CUDA(cuda_opts);
    Logger(DEBUG, "Setting cuda opts to OrtCudnnConvAlgoSearchExhaustive");

    wchar_t buff[512];
    swprintf_s(buff, _countof(buff), L"%hs", model_file);
    da3_ctx.session = std::make_unique<Ort::Session>(*da3_ctx.env, buff, so);
    Logger(DEBUG, "Loading Model: %s", model_file);


    Ort::AllocatorWithDefaultOptions allocator;
    da3_ctx.input_name = da3_ctx.session->GetInputNameAllocated(0, allocator).get();
    da3_ctx.output_name = da3_ctx.session->GetOutputNameAllocated(0, allocator).get();
    da3_ctx.binding = std::make_unique<Ort::IoBinding>(*da3_ctx.session);
    Logger(DEBUG, "Input (%s) => Output (%s)", da3_ctx.input_name, da3_ctx.output_name);


    size_t input_bytes = (1ull * 3 * MODEL_WIDTH * MODEL_HEIGHT) * sizeof(float);
    if(cudaMalloc((void**)&da3_ctx.input, input_bytes) != cudaSuccess){
        throw std::runtime_error("Failed to allocate memory on CUDA");
    }
    Logger(INFO, "Succesfully allocated %lu bytes in CUDA memory", input_bytes);
}

void DA3PrepareVideo(cv::VideoCapture * cap, video_feed_e video_feed){
    switch(video_feed){
        case VIDEOFEED_TEST: *cap = cv::VideoCapture("../../models/tests/test.mp4"); break;
        case VIDEOFEED_CAM: *cap = cv::VideoCapture(0); break;
        case VIDEOFEED_RTSP:{
            char buf[256];
            snprintf(buf, sizeof(buf), "rtsp://%s:%u/%s", RTSP_IP, RTSP_PORT, RTSP_MOUNT);
            *cap = cv::VideoCapture(buf);
        }break;
    }
    if (!cap->isOpened()) {
        Logger(ERROR, "Could not open from: %s", video_feed == 0 ? "TEST" : video_feed == 1 ? "CAM" : "RTSP");
        return;
    }
}

static inline void DA3preprocessToGPU(const cv::cuda::GpuMat * input, float* ouput, cv::cuda::Stream& stream){
    cv::cuda::GpuMat resized, rgb;

    cv::cuda::resize(*input, resized, cv::Size(MODEL_WIDTH, MODEL_HEIGHT), 0, 0, cv::INTER_LINEAR, stream);
    cv::cuda::cvtColor(resized, rgb, cv::COLOR_BGR2RGB, 0, stream);

    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(stream);
    LaunchRgb8ToChwF32Norm(rgb.ptr<unsigned char>(), rgb.step,MODEL_HEIGHT, MODEL_WIDTH,ouput, MEAN, STD, custream);
}

void DA3InferInputHeadLess(float * input, onnx_ctx_t * ctx){
    ctx->binding->BindOutput(ctx->output_name.c_str(), ctx->cuda_mem);

    Ort::RunOptions run_options;
    run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
    ctx->session->Run(run_options, *ctx->binding);
}


static cv::Mat DA3ColorizeDepth(const cv::Mat& depth32f)
{
    double minv, maxv;
    cv::minMaxLoc(depth32f, &minv, &maxv);

    cv::Mat norm8u;
    if (maxv - minv < 1e-6) {
        norm8u = cv::Mat::zeros(depth32f.size(), CV_8U);
    } else {
        cv::Mat norm01 = (depth32f - minv) / (maxv - minv);
        norm01.convertTo(norm8u, CV_8U, 255.0);
    }

    cv::Mat colored;
    cv::applyColorMap(norm8u, colored, cv::COLORMAP_INFERNO);
    return colored;
}
static cv::cuda::GpuMat frame_gpu;
static Ort::Value input_tensor{nullptr};

cv::Mat DA3InferInput(cv::Mat frame_cpu, cv::cuda::Stream cv_stream){
    frame_gpu.upload(frame_cpu, cv_stream);
    DA3preprocessToGPU(&frame_gpu, da3_ctx.input, cv_stream);

    da3_ctx.binding->ClearBoundInputs();
    da3_ctx.binding->ClearBoundOutputs();

    const size_t input_elem = 1ull * 3 * MODEL_WIDTH * MODEL_HEIGHT;
    input_tensor = Ort::Value::CreateTensor<float>(da3_ctx.cuda_mem,da3_ctx.input,input_elem,da3_ctx.input_shape.data(),da3_ctx.input_shape.size());
    da3_ctx.binding->BindInput(da3_ctx.input_name.c_str(), input_tensor);

    da3_ctx.binding->BindOutput(da3_ctx.output_name.c_str(), da3_ctx.cpu_mem);

    da3_ctx.session->Run(Ort::RunOptions{nullptr}, *da3_ctx.binding);
    da3_ctx.binding->SynchronizeOutputs();

    auto outs = da3_ctx.binding->GetOutputValues();
    auto& depth_val = outs[0];

    const float* dptr = depth_val.GetTensorData<float>();
    cv::Mat depth(MODEL_HEIGHT, MODEL_WIDTH, CV_32F, (void*)dptr);

    return DA3ColorizeDepth(depth);
}


cv::cuda::GpuMat * DA3Infer(cv::Mat frame_cpu, cv::cuda::Stream cv_stream){
    frame_gpu.upload(frame_cpu, cv_stream);
    DA3preprocessToGPU(&frame_gpu, da3_ctx.input, cv_stream);

    da3_ctx.binding->ClearBoundInputs();
    da3_ctx.binding->ClearBoundOutputs();

    const size_t input_elem = 1ull * 3 * MODEL_WIDTH * MODEL_HEIGHT;
    input_tensor = Ort::Value::CreateTensor<float>(da3_ctx.cuda_mem,da3_ctx.input,input_elem,da3_ctx.input_shape.data(),da3_ctx.input_shape.size());
    da3_ctx.binding->BindInput(da3_ctx.input_name.c_str(), input_tensor);

    DA3InferInputHeadLess(da3_ctx.input, &da3_ctx);

    return &frame_gpu;

}

cv::Mat DA3Getframe(cv::cuda::Stream cv_stream){
    static cv::Mat frame;
    frame_gpu.download(frame, cv_stream);
    cv_stream.waitForCompletion();
    return DA3ColorizeDepth(frame);
}



void DA3InferThread(video_feed_e video_feed, models_e model){

    DA3InitInfer(model);

    cv::VideoCapture cap;
    DA3PrepareVideo(&cap, video_feed);

    cv::Mat frame_cpu;
    cv::cuda::GpuMat frame_gpu;
    cv::cuda::Stream cv_stream;

    Ort::Value input_tensor{nullptr};
    double t0,t1;
    std::vector<double> samples;
    double start, end;

    while (true) {
        start = NowMs();
        if (!cap.read(frame_cpu) || frame_cpu.empty()) {
            if (video_feed == VIDEOFEED_RTSP) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }
            break;
        }

        frame_gpu.upload(frame_cpu, cv_stream);
        DA3preprocessToGPU(&frame_gpu, da3_ctx.input, cv_stream);

        // cv_stream.waitForCompletion();

        da3_ctx.binding->ClearBoundInputs();
        da3_ctx.binding->ClearBoundOutputs();

        const size_t input_elem = 1ull * 3 * MODEL_WIDTH * MODEL_HEIGHT;
        input_tensor = Ort::Value::CreateTensor<float>(da3_ctx.cuda_mem,da3_ctx.input,input_elem,da3_ctx.input_shape.data(),da3_ctx.input_shape.size());
        da3_ctx.binding->BindInput(da3_ctx.input_name.c_str(), input_tensor);

        if (show_video) {
            t0 = NowMs();
            cv::Mat output_frame = DA3InferInput(frame_cpu, cv_stream);
            t1 = NowMs();
            samples.push_back(t1-t0);

            cv::Mat frame_show;
            frame_gpu.download(frame_show, cv_stream);
            cv_stream.waitForCompletion();

            cv::Mat out_gray = toGray(output_frame);
            cv::Mat exp_smooth = CommonExpSmooth(out_gray, 0.70);
            // processAndShow(output_frame);

            cv::imshow("Input", out_gray);
            cv::imshow("Output", exp_smooth);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') break;
        } else {
            DA3InferInputHeadLess(da3_ctx.input, &da3_ctx);
        }

        PrintPerf("[DA3]", samples);
        end = NowMs();

        if(end-start < 67){
            int dift = (int)(end-start);
            std::this_thread::sleep_for(std::chrono::milliseconds(dift));
        }
    }

    if (da3_ctx.input) cudaFree(da3_ctx.input);
    if (show_video) cv::destroyAllWindows();

    Logger(INFO, "Finished sucessfully");
}