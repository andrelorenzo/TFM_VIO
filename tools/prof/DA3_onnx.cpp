
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "onnxruntime_cxx_api.h"
#include <cuda_runtime.h>
#include "preprocess.h"

#define NOMINMAX
#include "windows.h"
#undef ERROR

#include <chrono>
#include <memory>
#include <ctime>
#include <filesystem>

#define PARSER_IMP
#include "file_parser.h"

#define LOGGER_IMP
#include "logger.h"

static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

typedef enum{
    VIDEOFEED_TEST = 0U,
    VIDEOFEED_CAM,
    VIDEOFEED_RTSP
}video_feed_e;
#define RTSP_PORT 5004
#define RTSP_IP "192.168.1.145"
#define RTSP_MOUNT "realsense"
#define MODEL_WIDTH 504
#define MODEL_HEIGHT 280

typedef enum{
    MODEL_DA3_SMALL = 0u,
    MODEL_DA3_BASE,

    MODEL__COUNT
}models_e;
const char * modelsf[MODEL__COUNT] = {
    "../../model/DA3METRIC-SMALL.onnx",
    "../../model/DA3METRIC-BASE.onnx"
};
static bool show_video = false; 

typedef struct{
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name;
    std::array<int64_t, 4> input_shape{1, 3, MODEL_HEIGHT, MODEL_WIDTH};
    Ort::MemoryInfo cuda_mem{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    Ort::MemoryInfo cpu_mem  = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::unique_ptr<Ort::IoBinding> binding;
    float * input = nullptr;
}onnx_ctx_t;

// void OnnxInferDA3(const char * filename, const char * model_file, bool show, bool cam);
void DA3InferThread(video_feed_e video_feed, models_e model);

int main(int argc, char ** argv){

    const char * video_test = "../../models/tests/test_video.mp4";
    const char * config_file = "../../config/da3_onnx.txt";
    LoggerSetVerbsity(DEBUG);

    char model[256];
    bool cam = false;

    ParamStr((char**)&model, "MODEL", true, (char*)"../../models/midas/model-small.onnx", "Choose the model to run");
    ParamBool(&show_video, "SHOW", false, false, "Show video windwos for debugging");
    ParamBool(&cam, "CAM", false, false, "Get video from camera (0) feed");

    if(!ParamParse(config_file, FILE_TYPE_TXT)){
        ParamPrintError(stdout);
        return -1;
    }

    DA3InferThread(cam ? VIDEOFEED_CAM : VIDEOFEED_TEST, MODEL_DA3_SMALL);

    // OnnxInferDA3(video_test, model, video, cam);

    return 0;
}




void DA3InitInfer(models_e model, onnx_ctx_t * onnx_ctx){
    onnx_ctx->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "da3");
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
    onnx_ctx->session = std::make_unique<Ort::Session>(*onnx_ctx->env, buff, so);
    Logger(DEBUG, "Loading Model: %s", model_file);


    Ort::AllocatorWithDefaultOptions allocator;
    onnx_ctx->input_name = onnx_ctx->session->GetInputNameAllocated(0, allocator).get();
    onnx_ctx->output_name = onnx_ctx->session->GetOutputNameAllocated(0, allocator).get();
    onnx_ctx->binding = std::make_unique<Ort::IoBinding>(*onnx_ctx->session);
    Logger(DEBUG, "Input (%s) => Output (%s)", onnx_ctx->input_name, onnx_ctx->output_name);


    size_t input_bytes = (1ull * 3 * MODEL_WIDTH * MODEL_HEIGHT) * sizeof(float);
    if(cudaMalloc((void**)&onnx_ctx->input, input_bytes) != cudaSuccess){
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
    if(show_video){
        cv::namedWindow("Input", cv::WINDOW_NORMAL);
        cv::namedWindow("Output", cv::WINDOW_NORMAL);
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

cv::Mat DA3InferInput(float * input, onnx_ctx_t * ctx){
    ctx->binding->BindOutput(ctx->output_name.c_str(), ctx->cpu_mem);

    ctx->session->Run(Ort::RunOptions{nullptr}, *ctx->binding);
    ctx->binding->SynchronizeOutputs();

    auto outs = ctx->binding->GetOutputValues();     // <-- guardar
    auto& depth_val = outs[0];

    const float* dptr = depth_val.GetTensorData<float>();
    cv::Mat depth(MODEL_HEIGHT, MODEL_WIDTH, CV_32F, (void*)dptr);

    // colorizeDepth crea un Mat nuevo (CPU), así que ya no dependes del buffer ORT
    return DA3ColorizeDepth(depth);
}

void DA3InferThread(video_feed_e video_feed, models_e model){
    onnx_ctx_t da3_ctx;
    DA3InitInfer(model, &da3_ctx);

    cv::VideoCapture cap;
    DA3PrepareVideo(&cap, video_feed);

    cv::Mat frame_cpu;
    cv::cuda::GpuMat frame_gpu;
    cv::cuda::Stream cv_stream;

    Ort::Value input_tensor{nullptr};

    while (true) {
        if (!cap.read(frame_cpu) || frame_cpu.empty()) {
            if (video_feed == VIDEOFEED_RTSP) { Sleep(20); continue; }
            break;
        }

        // SUBIR A GPU
        frame_gpu.upload(frame_cpu, cv_stream);

        // PREPROCESS EN GPU (ojo: tu función ahora debe NO modificar in-place el mismo GpuMat)
        DA3preprocessToGPU(&frame_gpu, da3_ctx.input, cv_stream);

        // asegúrate de que el kernel terminó antes de correr ORT (si no compartes stream con ORT)
        cv_stream.waitForCompletion();

        da3_ctx.binding->ClearBoundInputs();
        da3_ctx.binding->ClearBoundOutputs();

        const size_t input_elem = 1ull * 3 * MODEL_WIDTH * MODEL_HEIGHT;
        input_tensor = Ort::Value::CreateTensor<float>(da3_ctx.cuda_mem,da3_ctx.input,input_elem,da3_ctx.input_shape.data(),da3_ctx.input_shape.size());
        da3_ctx.binding->BindInput(da3_ctx.input_name.c_str(), input_tensor);

        if (show_video) {
            cv::Mat output_frame = DA3InferInput(da3_ctx.input, &da3_ctx);

            // IMGUI: frame_gpu -> frame_cpu (porque imshow no soporta GpuMat)
            cv::Mat frame_show;
            frame_gpu.download(frame_show, cv_stream);
            cv_stream.waitForCompletion();

            cv::imshow("Input", frame_show);
            cv::imshow("Output", output_frame);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') break;
        } else {
            DA3InferInputHeadLess(da3_ctx.input, &da3_ctx);
        }
    }

    if (da3_ctx.input) cudaFree(da3_ctx.input);
    if (show_video) cv::destroyAllWindows();

    Logger(INFO, "Finished sucessfully");
}

/*

static void PrintPerf(const char* tag, const std::vector<double>& ms_samples) {
    if (ms_samples.empty()) return;
    double sum = 0.0;
    double mn = ms_samples[0], mx = ms_samples[0];
    for (double v : ms_samples) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    double avg = sum / ms_samples.size();
    double fps = 1000.0 / avg;
    printf("[%s] frames=%zu | avg=%.3f ms | min=%.3f ms | max=%.3f ms | FPS=%.2f\n",
           tag, ms_samples.size(), avg, mn, mx, fps);
}

static inline double NowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}


static int findOutputIndex(const std::vector<std::string>& names, const std::string& target)
{
    auto it = std::find(names.begin(), names.end(), target);
    if (it == names.end()) return -1;
    return int(std::distance(names.begin(), it));
}


static inline void preprocessToNCHW_GPU_intoDevice(const cv::Mat& bgr_cpu,cv::cuda::GpuMat& d_bgr,cv::cuda::GpuMat& d_resized,cv::cuda::GpuMat& d_rgb,float* d_dstCHW,int H, int W,const float mean[3],const float stdv[3],cv::cuda::Stream& stream){
    // 1) CPU -> GPU
    d_bgr.upload(bgr_cpu, stream);

    // 2) resize GPU
    cv::cuda::resize(d_bgr, d_resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR, stream);

    // 3) BGR -> RGB GPU (8-bit)
    cv::cuda::cvtColor(d_resized, d_rgb, cv::COLOR_BGR2RGB, 0, stream);

    // 4) RGB8 HWC -> CHW float32 normalizado (kernel)
    cudaStream_t cu = cv::cuda::StreamAccessor::getStream(stream);
    LaunchRgb8ToChwF32Norm(d_rgb.ptr<unsigned char>(),d_rgb.step,H, W,d_dstCHW,mean, stdv,cu);
}



static inline void CUDA_THROW_IF(cudaError_t e, const char* msg){
    if(e != cudaSuccess){
        std::string s = std::string(msg) + " : " + cudaGetErrorString(e);
        throw std::runtime_error(s);
    }
}


void OnnxInferDA3(const char * filename, const char * model_file, bool show, bool cam){
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "da3");
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    cuda_opts.arena_extend_strategy = 0;
    cuda_opts.gpu_mem_limit = SIZE_MAX;
    cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_opts.do_copy_in_default_stream = 1;
    so.AppendExecutionProvider_CUDA(cuda_opts);
    std::cout << "[ORT] CUDAExecutionProvider activo\n";

    wchar_t buff[512];
    swprintf_s(buff, _countof(buff), L"%hs", model_file);
    Ort::Session session(env, buff, so);
    Ort::AllocatorWithDefaultOptions allocator;

    auto inNameAlloc = session.GetInputNameAllocated(0, allocator);
    std::string inputName = inNameAlloc.get();

    std::vector<std::string> outputNames;
    size_t nOut = session.GetOutputCount();
    outputNames.reserve(nOut);
    for (size_t i = 0; i < nOut; ++i) {
        auto outNameAlloc = session.GetOutputNameAllocated(i, allocator);
        outputNames.emplace_back(outNameAlloc.get());
    }

    const int depthIdx = (findOutputIndex(outputNames, "depth") >= 0) ? findOutputIndex(outputNames, "depth") : 0;
    cv::VideoCapture cap;
    if(cam){
        cap = cv::VideoCapture(0);
    }else{
        cap = cv::VideoCapture(filename);
    }
    if (!cap.isOpened()) {
        Logger(ERROR, "Could not open video/cam: %s", filename);
        return;
    }

    if (show) {
        cv::namedWindow("Input", cv::WINDOW_NORMAL);
        cv::namedWindow("Output", cv::WINDOW_NORMAL);
    }

    // Shapes
    std::array<int64_t, 4> inputShape{1, 3, IN_H, IN_W};
    const size_t inputElems = 1ull * 3 * IN_H * IN_W;
    const size_t inputBytes = inputElems * sizeof(float);

    

    // Memory infos
    Ort::MemoryInfo cpu_mem  = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo cuda_mem{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

    // IOBinding reusable
    Ort::IoBinding binding(session);

    const int warmup = 10;
    std::vector<double> samples;
    samples.reserve(1000);

    float* dInput = nullptr;
    CUDA_THROW_IF(cudaMalloc((void**)&dInput, inputBytes), "cudaMalloc dInput");
    Logger(INFO, "[ORT] dInput device allocated (%zu bytes)", inputBytes);

    // OpenCV CUDA reusable buffers + stream
    cv::cuda::GpuMat d_bgr, d_resized, d_rgb;
    cv::cuda::Stream cvStream;

    // ORT tensors (wrappers)
    Ort::Value inputGpuTensor{nullptr};

    cv::Mat frame;
    int idx = 0;
    double t0 = 0, t1 = 0;

    try {
        while (cap.read(frame)) {
            if (frame.empty()) break;


            preprocessToNCHW_GPU_intoDevice(frame,d_bgr, d_resized, d_rgb,dInput,IN_H, IN_W,MEAN, STD,cvStream);

            cvStream.waitForCompletion();
            CUDA_THROW_IF(cudaGetLastError(), "CUDA kernel/preprocess error");

            binding.ClearBoundInputs();
            binding.ClearBoundOutputs();

            inputGpuTensor = Ort::Value::CreateTensor<float>(cuda_mem,dInput,inputElems,inputShape.data(),inputShape.size());
            binding.BindInput(inputName.c_str(), inputGpuTensor);

            if (show) {
                binding.BindOutput(outputNames[depthIdx].c_str(), cpu_mem);

                t0 = NowMs();
                session.Run(Ort::RunOptions{nullptr}, binding);
                binding.SynchronizeOutputs();
                t1 = NowMs();

                auto outs = binding.GetOutputValues();
                auto& depthVal = outs[0];
                const float* dptr = depthVal.GetTensorData<float>();
                auto dshape = depthVal.GetTensorTypeAndShapeInfo().GetShape();

                int outH = IN_H, outW = IN_W;
                if (dshape.size() == 4) { outH = (int)dshape[2]; outW = (int)dshape[3]; }
                else if (dshape.size() == 3) { outH = (int)dshape[1]; outW = (int)dshape[2]; }

                cv::Mat depth(outH, outW, CV_32F, (void*)dptr);
                cv::Mat depthVis = DA3ColorizeDepth(depth);

                cv::Mat frameRS;
                cv::resize(frame, frameRS, cv::Size(outW, outH), 0, 0, cv::INTER_LINEAR);

                cv::imshow("Input", frameRS);
                cv::imshow("Output", depthVis);
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q') break;

            } else {
                // PERF: output en GPU + no synchronize EPs
                binding.BindOutput(outputNames[depthIdx].c_str(), cuda_mem);

                Ort::RunOptions run_options;
                run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");

                t0 = NowMs();
                session.Run(run_options, binding);
                t1 = NowMs();
            }

            if (idx >= warmup) samples.push_back(t1 - t0);
            idx++;
        }

    } catch (const std::exception& e) {
        Logger(ERROR, "Exception: %s", e.what());
    }

    // Cleanup
    if (dInput) cudaFree(dInput);

    PrintPerf("ORT-DA3", samples);
    if (show) cv::destroyAllWindows();
}
*/