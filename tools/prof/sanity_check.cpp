// Opencv
#include <opencv2/world.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>

// Torch
#include <torch/torch.h>
#include <torch/version.h>

// OnnxRuntime
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

// C Headers
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

// Own Headers

static void StaticCheck();
static void CudaCheck();
static void CudnnCheck();
static void GstreamerCheck();
static void LibtorchCheck();
static void OnnxCheck();

static std::string findVersionAfter(const std::string& info,const std::string& key,const std::string& token);
static bool containsLine(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

int main(int argc, char** argv) {
    printf("\n\n====================SANITY CHECKS FOR OPENCV + CUDE + CUDNN + GSTREAMER + LIBTORCH====================\n");
    printf("=====Build (Static) check=====\n");
        StaticCheck();
    printf("=====Runtime (Dynamic) check=====\n");
        CudaCheck();
        CudnnCheck();
        GstreamerCheck();
        LibtorchCheck();
        OnnxCheck();
    printf("=================================\n");

  return 0;
}


static void StaticCheck(){
    const std::string bi = cv::getBuildInformation();
    const std::string cudaVer  = findVersionAfter(bi, "NVIDIA CUDA:", "ver");
    const std::string cudnnVer = findVersionAfter(bi, "cuDNN:", "ver");
    const std::string gstVer   = findVersionAfter(bi, "GStreamer:", "(");

    printf("\t[%s] CUDA      -> Vers: %s\n",(bi.find("NVIDIA CUDA:") != std::string::npos ? "OK" : "ERROR"),(cudaVer.empty() ? "unknown" : cudaVer.c_str()));
    printf("\t[%s] CUDNN     -> Vers: %s\n",(bi.find("cuDNN:") != std::string::npos ? "OK" : "ERROR"),(cudnnVer.empty() ? "unknown" : cudnnVer.c_str()));
    printf("\t[%s] Gstreamer -> Vers: %s\n",(bi.find("GStreamer:") != std::string::npos ? "OK" : "ERROR"),(gstVer.empty() ? "unknown" : gstVer.c_str()));
    bool libtorch_withcu_cudnn = (torch::cuda::is_available()) && (torch::cuda::cudnn_is_available());
    printf("\t[%s] Libtorch  -> Vers: %s\n", libtorch_withcu_cudnn ? "OK" : "ERROR", TORCH_VERSION);
    printf("\t[%s] Onnx      -> Vers: %s\n",cudaSetDevice(0) == cudaSuccess ? "OK" : "ERROR", OrtGetApiBase()->GetVersionString());

    c10::NameList providers = Ort::GetAvailableProviders();
    bool has_cuda = false;
    std::string provs;
    for (const auto& p : providers) {
        if (p == "CUDAExecutionProvider") has_cuda = true;
        provs += " " + p;
    }
    printf("\t[%s] Onnx Prov ->%s\n", has_cuda ? "OK" : "ERROR", provs.c_str());
}


static void OnnxCheck(){
    try {
        const ORTCHAR_T* model_path = (const ORTCHAR_T*)L"../../models/midas/model-small.onnx";

        // Env
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

        // Session options
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // CUDA EP
        OrtCUDAProviderOptions options;
        options.device_id = 0;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(so, options.device_id));


        // Session
        Ort::Session session(env, model_path, so);

        // Nombres / shapes
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session.GetInputCount();
        size_t num_outputs = session.GetOutputCount();
        if (num_inputs == 0 || num_outputs == 0) {
            std::printf("\t[ERROR] Model has no inputs/outputs\n");
            return;
        }

        auto in_name_alloc = session.GetInputNameAllocated(0, allocator);
        std::string input_name = in_name_alloc.get();

        Ort::TypeInfo in_type_info = session.GetInputTypeInfo(0);
        auto in_tensor_info = in_type_info.GetTensorTypeAndShapeInfo();

        auto in_shape = in_tensor_info.GetShape();
        for (auto& d : in_shape) if (d <= 0) d = 1;   // dims dinámicas -> 1

        if (in_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::printf("\t[ERROR] Input[0] is not float (elem_type=%d)\n", (int)in_tensor_info.GetElementType());
            return;
        }

        // Crear input dummy
        size_t n = 1;
        for (auto d : in_shape) n *= (size_t)d;
        std::vector<float> input_data(n, 1.0f);

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            input_data.data(), input_data.size(),
            in_shape.data(), in_shape.size()
        );

        // Outputs: pedimos solo el output[0] para simplificar
        auto out_name_alloc = session.GetOutputNameAllocated(0, allocator);
        std::string out_name = out_name_alloc.get();

        const char* in_names[] = { input_name.c_str() };
        const char* out_names[] = { out_name.c_str() };

        // Run
        auto outputs = session.Run(
            Ort::RunOptions{ nullptr },
            in_names, &input_tensor, 1,
            out_names, 1
        );

        // Check salida
        if (outputs.size() == 1 && outputs[0].IsTensor()) {
            auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
            auto out_shape = out_info.GetShape();

            std::printf("\t[OK] ONNX Runtime ran on CUDA EP and produced output[0]. Shape: [");
            for (size_t i = 0; i < out_shape.size(); ++i) {
                std::printf("%lld%s", (long long)out_shape[i], (i + 1 < out_shape.size() ? ", " : ""));
            }
            std::printf("]\n");
            return;
        }

        std::printf("\t[ERROR] Run returned no valid output tensor\n");
    }
    catch (const Ort::Exception& e) {
        std::printf("\t[ERROR] ORT exception: %s\n", e.what());
    }
    catch (const std::exception& e) {
        std::printf("\t[ERROR] Exception: %s\n", e.what());
    }
}

static void LibtorchCheck(){
    // Libtorch
    try {
    auto dev = torch::Device(torch::kCUDA, 0);
    auto a = torch::rand({512, 512}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto b = torch::rand({512, 512}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));

    auto c = a.matmul(b);
    c = c.relu();
    torch::cuda::synchronize();

    std::cout << "\t[OK] (GPU "<< static_cast<int>(torch::cuda::device_count()) << ") Libtorch on CUDA.\n";
    }
    catch (const c10::Error& e) {
        std::cout << "\t[ERROR] LibTorch runtime failed:\n" << e.what() << "\n";
    }
    catch (const std::exception& e) {
        std::cout << "\t[ERROR] LibTorch runtime failed (std::exception):\n" << e.what() << "\n";
    }


}

static void CudnnCheck(){
    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX("../../models/midas/model-small.onnx");
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        const int W = 256, H = 256;  // MiDaS small típico
        cv::Mat img(H, W, CV_8UC3, cv::Scalar(127, 127, 127));
        cv::Mat blob = cv::dnn::blobFromImage(img,1.0 / 255.0,cv::Size(W, H),cv::Scalar(),true,false,CV_32F);
        net.setInput(blob);
        cv::Mat out = net.forward();
        std::cout << "\t[OK] CUDNN forward ran. out.dims=" << out.dims << " total=" << out.total() << "\n";
    }catch (const cv::Exception& e) {
        std::cout << "\t[FAIL] CUDNN forward failed:\n" << e.what() << "\n";
    }
}

static void GstreamerCheck(){
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    const std::string pipe ="videotestsrc is-live=true pattern=smpte ! videoconvert ! textoverlay text=\"GStreamer OK\" font-desc=\"Sans, 36\" valignment=top halignment=left ! "
                            "videoconvert ! video/x-raw,format=BGR ! ""appsink max-buffers=1 drop=true sync=false";

    cv::VideoCapture cap(pipe, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cout << "\t[ERROR] GStreamer is not working (could not open pipeline).\n";
        std::cout << "\tPipeline: " << pipe << "\n";
        return;
    }

    std::cout << "\t[OK] GStreamer pipeline opened.\n";

    cv::namedWindow("GStreamer test", cv::WINDOW_NORMAL);
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
    std::cout << "\t[ERROR] GStreamer opened but returned empty frame.\n";
    }else{
    std::cout << "\t[OK] GStreamer succesfully passed frame to OpenCV.\n";
    }
    cap.release();
    cv::destroyAllWindows();
}
static void CudaCheck(){
    int cudaCount = 0;
    try {
        cudaCount = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "Device count = " << cudaCount << "\n";
        if (cudaCount <= 0) {
            std::cout << "\t[ERROR] No CUDA devices.\n\n";
        } else {
            cv::cuda::DeviceInfo dev(0);
            printf("[0] %s (%ziMB) - Arq: %i.%i\n", dev.name(), (dev.totalMemory() / (1024 * 1024)), dev.majorVersion(), dev.minorVersion());
            cv::Mat a(256, 256, CV_32F, cv::Scalar(1.0f));
            cv::Mat b(256, 256, CV_32F, cv::Scalar(2.0f));
            cv::cuda::GpuMat ga, gb, gc;
            ga.upload(a);
            gb.upload(b);
            cv::cuda::add(ga, gb, gc);
            cv::Mat c;
            gc.download(c);
            const float v = c.at<float>(0, 0);
            std::cout << (std::abs(v - 3.0f) < 1e-6 ? "\t[OK] CUDA compute works.\n" : "\t[ERROR] CUDA compute mismatch.\n");
        }
    } catch (const cv::Exception& e) {
        std::cout << "\t[ERROR] CUDA runtime :\n" << e.what() << "\n\n";
    }
}


static std::string findVersionAfter(const std::string& info,const std::string& key,const std::string& token) {
    const size_t kpos = info.find(key);
    if (kpos == std::string::npos) return "";
    size_t lineEnd = info.find('\n', kpos);
    if (lineEnd == std::string::npos) lineEnd = info.size();
    std::string line = info.substr(kpos, lineEnd - kpos);
    size_t tpos = line.find(token);
    if (tpos == std::string::npos) return "";

    tpos += token.size();
    while (tpos < line.size() && (line[tpos] == ' ' || line[tpos] == ':' )) ++tpos;

    std::string ver;
    while (tpos < line.size()) {
        char c = line[tpos];
        if (c == ',' || c == ')' || c == ' ' || c == '\r' || c == '\t') break;
        ver.push_back(c);
        ++tpos;
    }
    return ver;
}