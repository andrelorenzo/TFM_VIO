#include <torch/torch.h>
#include <torch/version.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "onnxruntime_cxx_api.h"
#include <cuda_runtime.h>
#include <chrono>
#include <memory>
#include <ctime>
#include <filesystem>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h> // For GetModuleFileNameW
#elif __APPLE__
#include <limits.h>      // For PATH_MAX or similar
#include <mach-o/dyld.h> // For _NSGetExecutablePath
#elif __linux__
#include <limits.h> // For PATH_MAX
#include <unistd.h> // For readlink
#endif

#undef ERROR

#define PARSER_IMP
#include "arg_parser.h"
#include "file_parser.h"

#define LOGGER_IMP
#include "logger.h"
#include "profiler.hpp"

#define PROF_NFRAMES 500
static bool profile = false;

#define PROF_CONCAT_INTERNAL(x,y) x##y
#define PROF_CONCAT(x,y) PROF_CONCAT_INTERNAL(x,y)
#define PROF_UNIQUE_NAME(base) PROF_CONCAT(base, __COUNTER__)

#if 1
    #define PROF_BEGIN_SESSION(name, filepath) do { if (profile) Instrumentor::Get().BeginSession((name), (filepath)); } while(0)
    #define PROF_END_SESSION()                 do { if (profile) Instrumentor::Get().EndSession(); } while(0)
    #define PROF_SCOPE(name) InstrumentationTimer PROF_UNIQUE_NAME(_timer_){ (name) }
    #define PROF_SCOPE_IF(name) auto PROF_UNIQUE_NAME(_timer_ptr_) = (profile ? std::make_unique<InstrumentationTimer>((name)) : nullptr)
#else
    #define PROF_BEGIN_SESSION(name, filepath)
    #define PROF_END_SESSION()
    #define PROF_SCOPE(name)
    #define PROF_SCOPE_IF(name)
#endif


void OpencvInfer(const char * filename, const char * model_file, uint16_t w, uint16_t h, bool gpu, bool show);
void TorchInfer (const char * filename, const char * model_file, uint16_t w, uint16_t h, bool gpu, bool show);
void OnnxInfer  (const char * filename, const char * model_file, int64_t w, int64_t h, bool gpu, bool show);
void OnnxInferDA3  (const char * filename, const char * model_file, int64_t w, int64_t h, bool gpu, bool show);
static void PreprocessToNCHWFloat(const cv::Mat& bgr, int W, int H, std::vector<float>& out_nchw);
void printOut(verb_e verbosity, const char * msg, size_t size);


static uint64_t getTick(){
    using clock = std::chrono::high_resolution_clock;
    return clock::now().time_since_epoch().count();
}
static double NowUs() {
    using clock = std::chrono::high_resolution_clock;
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(
        clock::now().time_since_epoch()).count();
}


static void PrintPerf(const char* tag, const std::vector<double>& ms_samples) {
    if (ms_samples.empty()) return;
    double sum = 0.0;
    double mn = ms_samples[0], mx = ms_samples[0];
    for (double v : ms_samples) { sum += v; mn = min(mn, v); mx = max(mx, v); }
    double avg = sum / ms_samples.size();
    double fps = 1000.0 / avg;
    printf( "[%s] frames=%zu | avg=%.3f ms | min=%.3f ms | max=%.3f ms | FPS=%.2f",
           tag, ms_samples.size(), avg, mn, mx, fps);

}
static bool TensorToDisplayMat_F32(const float* data, const std::vector<int64_t>& shape, cv::Mat& out_u8_color) {
    // Soporta [H,W], [1,H,W], [1,1,H,W]
    int H = 0, W = 0;
    if (shape.size() == 2) { H = (int)shape[0]; W = (int)shape[1]; }
    else if (shape.size() == 3) { H = (int)shape[1]; W = (int)shape[2]; }
    else if (shape.size() == 4) { H = (int)shape[2]; W = (int)shape[3]; }
    else return false;

    if (H <= 0 || W <= 0) return false;

    cv::Mat f32(H, W, CV_32F, (void*)data);
    cv::Mat norm_u8;
    // Normaliza a 0..255 para visualización (evita depender de rangos reales)
    cv::normalize(f32, norm_u8, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Colormap para ver “profundidad” mejor
    cv::applyColorMap(norm_u8, out_u8_color, cv::COLORMAP_TURBO);
    return true;
}



int main(int argc, char ** argv){

    const char * video_test = "../../models/tests/test_video.mp4";
    LoggerSetVerbsity(DEBUG);
    char ** config_file = FlagStr("filename", false, (char*)"../../config/perf_check.txt", "Config file for performance test a model");

    if(!FlagParse(argc,argv)){
        FlagPrintError(stdout);
        FlagPrintHelp(stdout);
        return -1;
    }

    char lib[256];
    char model[256];
    bool gpu=false;
    bool show_video=false;
    uint32_t width=0;
    uint32_t height=0;


    ParamStr((char**)&lib, "LIB", true, (char*)"opencv", "Choose the library to run the performance test (opencv, torch, ort)");
    ParamStr((char**)&model, "MODEL", true, (char*)"../../models/midas/model-small.onnx", "Choose the model to run");
    ParamBool(&gpu, "GPU", false, false, "Infer on GPU");
    ParamUint(&width, "WIDTH", true, 324, "Width of the depth model");
    ParamUint(&height, "HEIGHT", true, 324, "Width of the depth model");
    ParamBool(&show_video, "SHOW", false, false, "Show video windwos for debugging");
    ParamBool(&profile, "PROFI", false, false, "Save profiling test on .txt");
    
    if(!ParamParse(*config_file, FILE_TYPE_TXT)){
        ParamPrintError(stdout);
        return -1;
    }
    if(profile){
        // auto now = std::chrono::system_clock::now();
        // std::time_t t = std::chrono::system_clock::to_time_t(now);
        // std::tm tm_buf{};
        // localtime_s(&tm_buf, &t);
        // char ts[32];
        // std::snprintf(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d",tm_buf.tm_year + 1900,tm_buf.tm_mon + 1,tm_buf.tm_mday,tm_buf.tm_hour,tm_buf.tm_min,tm_buf.tm_sec);
        // std::string prof_path = std::string("../../profiling_results/prof_") + ts + ".json";
        // // const char * name = prof_path.c_str();
        
        PROF_BEGIN_SESSION("perf_check", "../../profiling_results/prof.json");
        // Logger(INFO, "Profiling enabled -> %s", prof_path.c_str());
    }
    // cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
    // cv::setNumThreads(1);

    if(strcmp(lib, "opencv") == 0){
        OpencvInfer(video_test, model, (uint16_t)width, (uint16_t)height, gpu, show_video);
    }else if(strcmp(lib, "torch") == 0){
        TorchInfer(video_test, model, (uint16_t)width, (uint16_t)height, gpu, show_video);
    }else if(strcmp(lib, "ort") == 0){
        OnnxInferDA3(video_test, model, (uint16_t)width, (uint16_t)height, gpu, show_video);
    }

    if (profile) {
        PROF_END_SESSION();
    }
    return 0;
}

void OpencvInfer(const char * filename, const char * model_file, uint16_t w, uint16_t h, bool gpu, bool show){
    try {
    cv::dnn::Net model = cv::dnn::readNetFromONNX(model_file);

    if(gpu){
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }else{
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        Logger(ERROR, "Could not open video: %s", filename);
        return;
    }

    if (show) {
        cv::namedWindow("Input", cv::WINDOW_NORMAL);
        cv::namedWindow("Output", cv::WINDOW_NORMAL);
    }

    const uint16_t nframes = PROF_NFRAMES;
    std::vector<double> samples;
    samples.reserve(nframes);

    cv::Mat frame;
    int idx = 0;

    while (cap.read(frame)) {
PROF_SCOPE_IF("OpenCV::Frame");

{
PROF_SCOPE_IF("OpenCV::Preprocess");
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0/255.0, cv::Size(w, h), cv::Scalar(), true, false);
        model.setInput(blob);

PROF_SCOPE_IF("OpenCV::Forward");
        uint64_t t0 = getTick();
        cv::Mat out;
        try {
            out = model.forward();
        } catch (const cv::Exception& e) {
            Logger(ERROR, "OpenCV forward failed: %s", e.what());
            Logger(ERROR, "Debug: frame=%dx%d ch=%d, blob size=%d dims=%d",
                frame.cols, frame.rows, frame.channels(), (int)blob.total(), blob.dims);
            for (int i = 0; i < blob.dims; ++i) Logger(ERROR, "  blob.size[%d]=%d", i, blob.size[i]);
            return; // aborta esa prueba limpiamente
        }
        uint64_t t1 = getTick();

        if (idx >= 10 && (int)samples.size() < nframes) {
            // getTick() son "ticks"; si quieres ms reales usa chrono duration_ms
            samples.push_back((double)(t1 - t0) / 1e6); // aproximación si count ~ ns (depende del reloj)
        }

        if (show) {
PROF_SCOPE_IF("OpenCV::Display");
            cv::imshow("Input", frame);

            // out puede ser 4D en Mat: a veces OpenCV lo guarda como 4D blob.
            // Lo más robusto: interpretarlo como float y shape conocida.
            cv::Mat out_vis;
            if (out.total() == (size_t)(w * h)) {
                cv::Mat f32(h, w, CV_32F, out.ptr<float>());
                cv::Mat u8, color;
                cv::normalize(f32, u8, 0, 255, cv::NORM_MINMAX, CV_8U);
                cv::applyColorMap(u8, color, cv::COLORMAP_TURBO);
                out_vis = color;
            } else {
                // fallback: intenta reshapes comunes
                cv::Mat flat = out.reshape(1, (int)out.total());
                // si no cuadra, solo muestra algo
                cv::Mat u8;
                cv::normalize(flat, u8, 0, 255, cv::NORM_MINMAX, CV_8U);
                cv::applyColorMap(u8, out_vis, cv::COLORMAP_TURBO);
            }
            cv::imshow("Output", out_vis);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
        }

}
        idx++;
        if ((int)samples.size() >= nframes) break;
    }

    PrintPerf("OpenCV", samples);

    if (show) cv::destroyAllWindows();
    } catch (const cv::Exception& e) {
    Logger(ERROR, "OpenCV exception:\n%s", e.what());
    return;
}
}

void TorchInfer(const char * filename, const char * model_file, uint16_t w, uint16_t h, bool gpu, bool show){
    std::string path(model_file);
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".onnx") {
        Logger(ERROR, "TorchInfer expects a TorchScript model (.pt/.ts), not .onnx");
        return;
    }
    torch::jit::Module module;  
    try { module = torch::jit::load(model_file); }
    catch (const c10::Error& e) { Logger(ERROR, "Torch load failed: %s", e.what()); return; }

    bool use_cuda = gpu && torch::cuda::is_available();
    if (gpu && !torch::cuda::is_available()) {
        Logger(WARN, "GPU requested but torch::cuda::is_available() == false. Falling back to CPU.");
    }

    if (use_cuda) module.to(torch::kCUDA);
    module.eval();

    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        Logger(ERROR, "Could not open video: %s", filename);
        return;
    }
    const uint16_t nframes = PROF_NFRAMES;
    std::vector<double> samples;
    samples.reserve(nframes);

    cv::Mat frame;
    std::vector<float> nchw;
    int idx = 0;
    if (show) {
        cv::namedWindow("Input", cv::WINDOW_NORMAL);
        cv::namedWindow("Output", cv::WINDOW_NORMAL);
    }

    while (cap.read(frame)) {
PROF_SCOPE_IF("Torch::Frame");
{
PROF_SCOPE_IF("Torch::Preprocess");
        PreprocessToNCHWFloat(frame, w, h, nchw);
}
        torch::Tensor input = torch::from_blob(nchw.data(), {1, 3, h, w}, torch::kFloat32).clone();
        if (use_cuda) input = input.to(torch::kCUDA);
PROF_SCOPE_IF("Torch::Forward");
        uint64_t t0 = getTick();
        torch::IValue out_iv;
        try {
            out_iv = module.forward({input});
        } catch (const c10::Error& e) {
            Logger(ERROR, "Torch forward failed: %s", e.what());
            Logger(ERROR, "Input tensor sizes: [1,3,%d,%d]", (int)h, (int)w);
            return;
        }
        if (out_iv.isTensor()) {
            auto t = out_iv.toTensor();
            Logger(INFO, "Torch output dtype=%d dim=%d", (int)t.scalar_type(), (int)t.dim());
            for (int i=0;i<t.dim();++i) Logger(INFO, "  out.size[%d]=%lld", i, (long long)t.size(i));
        }
        if (use_cuda) torch::cuda::synchronize();
        uint64_t t1 = getTick();

        if (idx >= 10 && (int)samples.size() < nframes) samples.push_back((double)(t1 - t0) / 1e6);

        if (show) {
PROF_SCOPE_IF("Torch::Display");
            cv::imshow("Input", frame);

            cv::Mat out_vis;
            if (out_iv.isTensor()) {
                auto out_t = out_iv.toTensor();
                if (use_cuda) out_t = out_t.to(torch::kCPU);
                out_t = out_t.contiguous();

                std::vector<int64_t> shape;
                for (auto s : out_t.sizes()) shape.push_back((int64_t)s);

                float* ptr = out_t.data_ptr<float>();
                if (!TensorToDisplayMat_F32(ptr, shape, out_vis)) {
                    out_vis = frame.clone();
                    cv::putText(out_vis, "Unsupported output shape", {20,40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
                }
            } else {
                out_vis = frame.clone();
                cv::putText(out_vis, "Output is not a Tensor", {20,40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
            }

            cv::imshow("Output", out_vis);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
        }

        idx++;
        if ((int)samples.size() >= nframes) break;
    }

    PrintPerf("Torch", samples);
    if (show) cv::destroyAllWindows();

}

#ifdef _WIN32
static std::wstring ToWString(const char* s) {
    if (!s) return L"";
    std::string a(s);
    return std::wstring(a.begin(), a.end()); // ASCII simple; si hay acentos, usa MultiByteToWideChar
}
#endif

static inline double NowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

std::string ElementTypeToString(ONNXTensorElementDataType type){
    switch(type){
case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED: return std::string("UNDEFINED");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return std::string("FLOAT32");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return std::string("UINT8");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return std::string("INT8");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return std::string("UINT16");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return std::string("INT16");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return std::string("INT32");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return std::string("INT64");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return std::string("STRING");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return std::string("BOOL");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return std::string("FLOAT16");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return std::string("DOUBLE");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return std::string("UINT32");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return std::string("UINT64");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return std::string("COMPLEX64");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return std::string("COMPLEX128");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return std::string("BFLOAT16");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN: return std::string("FLOAT8E4M3FN");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ: return std::string("FLOAT8E4M3FNUZ");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2: return std::string("FLOAT8E5M2");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ: return std::string("FLOAT8E5M2FNUZ");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4: return std::string("UINT4");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4: return std::string("INT4");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1: return std::string("FLOAT4E2M1");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2: return std::string("UINT2");
case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2: return std::string("INT2");
default: return std::string("");
    }

}

#include "string.h"

typedef struct {
    ONNXTensorElementDataType type;
    std::vector<int64_t> size;
    char input_name[64];
    char output_name[64];
}model_config_t;

inline std::string ShapeToString(const std::vector<int64_t>& s) {
    std::string out = "[ ";
    for (auto v : s) out += std::to_string(v) + " ";
    out += "]";
    return out;
}

// Decide si el input es “imagen-like”: rank 4 (N,C,H,W) o rank 5 (B,Nimg,C,H,W)
inline bool IsImageLikeRank(size_t rank) {
    return rank == 4 || rank == 5;
}

// Heurística: H y W suelen ser los dos últimos ejes
inline void FillDynamicDims(std::vector<int64_t>& shape,
                            int64_t batch_default,
                            int64_t num_images_default,
                            int64_t h_default,
                            int64_t w_default)
{
    if (shape.empty()) return;

    // Rellenar H/W si son dinámicos (últimos 2 ejes)
    if (shape.size() >= 2) {
        const size_t h_idx = shape.size() - 2;
        const size_t w_idx = shape.size() - 1;
        if (shape[h_idx] <= 0) shape[h_idx] = h_default;
        if (shape[w_idx] <= 0) shape[w_idx] = w_default;
    }

    // Rank 4: [N,C,H,W]
    if (shape.size() == 4) {
        if (shape[0] <= 0) shape[0] = batch_default; // N
        // C normalmente 3; si viene dinámico lo dejamos a 3 por defecto
        if (shape[1] <= 0) shape[1] = 3;
    }
    // Rank 5: [B, Nimg, C, H, W]
    else if (shape.size() == 5) {
        if (shape[0] <= 0) shape[0] = batch_default;       // batch
        if (shape[1] <= 0) shape[1] = num_images_default;  // num_images
        if (shape[2] <= 0) shape[2] = 3;                   // channels
    }
}

inline void PrintModelInputInfo(Ort::Session& session,
                               Ort::AllocatorWithDefaultOptions& alloc,
                               model_config_t* model_config)
{
    if (!model_config) return;

    // --- Inputs: nombres + tipos + shapes ---
    const size_t n_in  = session.GetInputCount();
    const size_t n_out = session.GetOutputCount();

    std::string inputs_info;
    inputs_info += "[ORT] Model IO:\n";
    inputs_info += "\tInputs (" + std::to_string(n_in) + "):\n";

    // Vamos a elegir un “input principal” automáticamente
    size_t chosen_input = 0;
    bool found = false;

    for (size_t i = 0; i < n_in; ++i) {
        auto in_name = session.GetInputNameAllocated(i, alloc);

        Ort::TypeInfo ti = session.GetInputTypeInfo(i);
        auto tkind = ti.GetONNXType();

        inputs_info += "\t  - #" + std::to_string(i) + " name=" + std::string(in_name.get());

        if (tkind != ONNXType::ONNX_TYPE_TENSOR) {
            inputs_info += " (NON-TENSOR)\n";
            continue;
        }

        auto tensor_info = ti.GetTensorTypeAndShapeInfo();
        auto elem_type   = tensor_info.GetElementType();
        auto shape       = tensor_info.GetShape();

        inputs_info += " type=" + ElementTypeToString(elem_type);
        inputs_info += " shape=" + ShapeToString(shape) + "\n";

        // Selección del input “imagen”:
        // - tensor
        // - rank 4 o 5
        // - tipo float32/uint8/float16 (si quieres)
        if (!found && IsImageLikeRank(shape.size())) {
            if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
                elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            {
                chosen_input = i;
                found = true;
            }
        }
    }

    // --- Output: coge el 0 (como haces ahora) ---
    auto out0_name = session.GetOutputNameAllocated(0, alloc);

    // --- Si no encontramos input “imagen-like”, cae al 0 igualmente ---
    auto chosen_name = session.GetInputNameAllocated(chosen_input, alloc);

    // Copia nombres al config
    std::strncpy(model_config->input_name,  chosen_name.get(),  sizeof(model_config->input_name) - 1);
    model_config->input_name[sizeof(model_config->input_name) - 1] = '\0';

    std::strncpy(model_config->output_name, out0_name.get(), sizeof(model_config->output_name) - 1);
    model_config->output_name[sizeof(model_config->output_name) - 1] = '\0';

    // --- Tipo y shape del input elegido ---
    Ort::TypeInfo chosen_ti = session.GetInputTypeInfo(chosen_input);
    auto chosen_tensor_info = chosen_ti.GetTensorTypeAndShapeInfo();

    model_config->size = chosen_tensor_info.GetShape();          // puede contener -1
    model_config->type = chosen_tensor_info.GetElementType();

    // Log final
    inputs_info += "\tChosen input: #" + std::to_string(chosen_input) +
                   " name=" + std::string(model_config->input_name) + "\n";
    inputs_info += "\tOutput0 name=" + std::string(model_config->output_name) + "\n";
    inputs_info += "\tChosen input element type=" + ElementTypeToString(model_config->type) +
                   " shape=" + ShapeToString(model_config->size) + "\n";

    Logger(INFO, "%s", inputs_info.c_str());
}
// Construye un Ort::Value float32 para inputs tipo:
//  - [1,3,H,W]
//  - [B,C,H,W]
//  - [B,Nimg,3,H,W]   (aquí Nimg=1 por defecto)
//
// Convierte cv::Mat BGR8 -> RGB float32 [0,1] y layout CHW.
// Reemplaza -1 típicos (B, Nimg, H, W) con valores runtime.
//
// Requisitos:
//  - frame: CV_8UC3 (BGR)
//  - cpu_mem: Ort::MemoryInfo CPU
//  - model_config.size: shape plantilla del input (puede contener -1)
// Devuelve: tensor Ort float32 + rellena out_shape_runtime y out_buffer (para mantener viva la memoria).
static inline Ort::Value MakeInputTensor_F32_ImageLike(
    const cv::Mat& frame_bgr,
    int64_t runtime_w,
    int64_t runtime_h,
    const std::vector<int64_t>& model_shape_template,
    Ort::MemoryInfo& cpu_mem,
    std::vector<int64_t>& out_shape_runtime,
    std::vector<float>& out_buffer,
    int64_t batch_default = 1,
    int64_t num_images_default = 1,
    bool input_is_rgb = true,     // normalmente true
    bool normalize_01 = true      // normalmente true
) {
    if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) {
        throw std::runtime_error("MakeInputTensor_F32_ImageLike: frame must be CV_8UC3");
    }

    // 1) Resize a (runtime_w, runtime_h)
    cv::Mat resized;
    if (frame_bgr.cols != runtime_w || frame_bgr.rows != runtime_h) {
        cv::resize(frame_bgr, resized, cv::Size((int)runtime_w, (int)runtime_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = frame_bgr;
    }

    // 2) BGR -> RGB si toca
    cv::Mat rgb;
    if (input_is_rgb) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = resized; // se queda BGR
    }

    if (!rgb.isContinuous()) rgb = rgb.clone();

    // 3) Construir shape runtime desde plantilla, rellenando -1 típicos
    out_shape_runtime = model_shape_template;
    if (out_shape_runtime.empty()) {
        // fallback típico si el modelo no da shape
        out_shape_runtime = {1, 3, runtime_h, runtime_w};
    }

    const size_t rank = out_shape_runtime.size();
    if (!(rank == 4 || rank == 5)) {
        throw std::runtime_error("MakeInputTensor_F32_ImageLike: only rank 4 or 5 supported");
    }

    // H/W son los 2 últimos ejes normalmente
    if (out_shape_runtime[rank - 2] <= 0) out_shape_runtime[rank - 2] = runtime_h;
    if (out_shape_runtime[rank - 1] <= 0) out_shape_runtime[rank - 1] = runtime_w;

    if (rank == 4) {
        // [N,C,H,W]
        if (out_shape_runtime[0] <= 0) out_shape_runtime[0] = batch_default;
        if (out_shape_runtime[1] <= 0) out_shape_runtime[1] = 3;
    } else {
        // [B,Nimg,C,H,W]
        if (out_shape_runtime[0] <= 0) out_shape_runtime[0] = batch_default;
        if (out_shape_runtime[1] <= 0) out_shape_runtime[1] = num_images_default;
        if (out_shape_runtime[2] <= 0) out_shape_runtime[2] = 3;
    }

    // En este helper SOLO soportamos batch=1 y num_images=1 a nivel de contenido,
    // aunque el shape permita otros valores.
    const int64_t B = (rank == 4) ? out_shape_runtime[0] : out_shape_runtime[0];
    const int64_t Nimg = (rank == 5) ? out_shape_runtime[1] : 1;
    const int64_t C = (rank == 4) ? out_shape_runtime[1] : out_shape_runtime[2];
    const int64_t H = out_shape_runtime[rank - 2];
    const int64_t W = out_shape_runtime[rank - 1];

    if (C != 3) {
        throw std::runtime_error("MakeInputTensor_F32_ImageLike: only C=3 supported in this helper");
    }
    if (B != 1 || Nimg != 1) {
        // Si quieres batch>1 o num_images>1, habría que apilar frames/imágenes.
        throw std::runtime_error("MakeInputTensor_F32_ImageLike: only batch=1 and num_images=1 are filled");
    }
    if (H != runtime_h || W != runtime_w) {
        // por seguridad, pero normalmente ya coincide
        throw std::runtime_error("MakeInputTensor_F32_ImageLike: H/W mismatch after fill");
    }

    // 4) HWC uint8 -> CHW float32
    const size_t hw = (size_t)H * (size_t)W;
    out_buffer.resize(hw * 3);

    const uint8_t* src = rgb.ptr<uint8_t>(0);

    // out_buffer layout: [C][H][W]
    float* dstC0 = out_buffer.data() + 0 * hw;
    float* dstC1 = out_buffer.data() + 1 * hw;
    float* dstC2 = out_buffer.data() + 2 * hw;

    if (normalize_01) {
        const float inv255 = 1.0f / 255.0f;
        for (size_t i = 0; i < hw; ++i) {
            const uint8_t r = src[3 * i + 0];
            const uint8_t g = src[3 * i + 1];
            const uint8_t b = src[3 * i + 2];
            dstC0[i] = (float)r * inv255;
            dstC1[i] = (float)g * inv255;
            dstC2[i] = (float)b * inv255;
        }
    } else {
        for (size_t i = 0; i < hw; ++i) {
            dstC0[i] = (float)src[3 * i + 0];
            dstC1[i] = (float)src[3 * i + 1];
            dstC2[i] = (float)src[3 * i + 2];
        }
    }

    // 5) Crear tensor Ort con shape rank 4 o 5
    // Si rank==5, el contenido es como si fuera Nimg=1: [1,1,3,H,W]
    // El buffer contiene solo 3*H*W, que coincide cuando B=1 y Nimg=1.
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        cpu_mem,
        out_buffer.data(),
        out_buffer.size(),              // número de elementos float
        out_shape_runtime.data(),
        out_shape_runtime.size()
    );

    return input_tensor;
}

static constexpr int IN_H = 280;
static constexpr int IN_W = 504;

// ImageNet mean/std
static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};
static std::vector<float> preprocessToNCHW(const cv::Mat& bgr)
{
    cv::Mat resized, rgb;
    cv::resize(bgr, resized, cv::Size(IN_W, IN_H), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<float> chw(1 * 3 * IN_H * IN_W);

    for (int y = 0; y < IN_H; ++y) {
        const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < IN_W; ++x) {
            const cv::Vec3f& px = row[x]; // RGB
            for (int c = 0; c < 3; ++c) {
                float v = (px[c] - MEAN[c]) / STD[c];
                chw[c * (IN_H * IN_W) + y * IN_W + x] = v;
            }
        }
    }
    return chw;
}

static cv::Mat colorizeDepth(const cv::Mat& depth32f)
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

static int findOutputIndex(const std::vector<std::string>& names, const std::string& target)
{
    auto it = std::find(names.begin(), names.end(), target);
    if (it == names.end()) return -1;
    return int(std::distance(names.begin(), it));
}
#include <cuda_runtime.h>

// ------------------------------------------------------------
// CUDA helpers
// ------------------------------------------------------------
static inline void CudaCheck(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(e) << " at " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}
#define CUDA_CHECK(x) CudaCheck((x), __FILE__, __LINE__)

// ------------------------------------------------------------
// Preprocess CPU -> escribe DIRECTO en CHW (float) ya normalizado
// (usa el buffer que le pasas; aquí lo usaremos como pinned host)
// ------------------------------------------------------------
static inline void preprocessToNCHW_into(
    const cv::Mat& bgr,
    float* dstCHW,          // size = 3*IN_H*IN_W
    int H, int W,
    const float mean[3],
    const float stdv[3]
) {
    cv::Mat resized, rgb, f32;
    cv::resize(bgr, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    const int plane = H * W;

    // (Más rápido que triple bucle con Vec3f + c)
    // Seguimos con doble bucle, pero sin vector temporario y escribiendo a dst directamente.
    for (int y = 0; y < H; ++y) {
        const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
        const int base = y * W;
        for (int x = 0; x < W; ++x) {
            const cv::Vec3f& px = row[x]; // RGB float [0..1]
            const int idx = base + x;
            dstCHW[0 * plane + idx] = (px[0] - mean[0]) / stdv[0];
            dstCHW[1 * plane + idx] = (px[1] - mean[1]) / stdv[1];
            dstCHW[2 * plane + idx] = (px[2] - mean[2]) / stdv[2];
        }
    }
}

void OnnxInferDA3(const char * filename, const char * model_file, int64_t w, int64_t h, bool gpu, bool show){
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "da3");
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (gpu) {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        cuda_opts.arena_extend_strategy = 0;
        cuda_opts.gpu_mem_limit = SIZE_MAX;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_opts.do_copy_in_default_stream = 1;
        so.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "[ORT] CUDAExecutionProvider activo\n";
    }

    Ort::Session session(env, ToWString(model_file).c_str(), so);
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

    const int depthIdx = (findOutputIndex(outputNames, "depth") >= 0)
        ? findOutputIndex(outputNames, "depth") : 0;
    const int skyIdx = findOutputIndex(outputNames, "sky");

    // Video
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        Logger(ERROR, "Could not open video: %s", filename);
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
    samples.reserve(PROF_NFRAMES);

    // ------------------------------------------------------------
    // 2.C) MEJORAS: pinned host + input en GPU
    // ------------------------------------------------------------
    float* hInputPinned = nullptr;   // pinned host
    float* dInput = nullptr;         // device
    bool haveGpuInput = false;

    if (gpu) {
        // Pinned host para preprocess (más rápido H2D)
        CUDA_CHECK(cudaMallocHost((void**)&hInputPinned, inputBytes)); // page-locked
        // Device buffer persistente
        CUDA_CHECK(cudaMalloc((void**)&dInput, inputBytes));
        haveGpuInput = true;
        Logger(INFO, "[ORT] Input buffers: pinned host + device allocated (%zu bytes)", inputBytes);
    } else {
        // fallback: usamos vector normal en CPU
        hInputPinned = (float*)std::malloc(inputBytes);
        if (!hInputPinned) throw std::bad_alloc();
    }

    Ort::Value inputTensor{nullptr};     // se recrea pero SIN alloc grande (solo wrapper)
    Ort::Value inputCpuTensor{nullptr};  // path CPU
    Ort::Value inputGpuTensor{nullptr};  // path GPU

    cv::Mat frame;
    int idx = 0;

    try {
        while (cap.read(frame)) {
            if (frame.empty()) break;

            // -------------------------
            // PREPROCESS (CPU) -> escribe en hInputPinned
            // -------------------------
            preprocessToNCHW_into(frame, hInputPinned, IN_H, IN_W, MEAN, STD);

            // -------------------------
            // INFERENCE
            // -------------------------
            binding.ClearBoundInputs();
            binding.ClearBoundOutputs();

            if (gpu) {
                // Copia H2D explícita (evita que ORT meta MemcpyFromHost por su cuenta)
                // Usamos copia síncrona en stream por defecto (0) para no liarla con streams.
                CUDA_CHECK(cudaMemcpy(dInput, hInputPinned, inputBytes, cudaMemcpyHostToDevice));

                // Input tensor EN GPU (device ptr)
                inputGpuTensor = Ort::Value::CreateTensor<float>(
                    cuda_mem,
                    dInput,
                    inputElems,                 // número de floats
                    inputShape.data(),
                    inputShape.size()
                );
                binding.BindInput(inputName.c_str(), inputGpuTensor);

                if (!show) {
                    // PERF: outputs en GPU y NO los lees
                    binding.BindOutput(outputNames[depthIdx].c_str(), cuda_mem);
                    if (skyIdx >= 0) binding.BindOutput(outputNames[skyIdx].c_str(), cuda_mem);

                    Ort::RunOptions run_options;
                    run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");

                    double t0 = NowMs();
                    session.Run(run_options, binding);
                    double t1 = NowMs();

                    if (idx >= warmup && (int)samples.size() < PROF_NFRAMES)
                        samples.push_back(t1 - t0);

                    // Nada de SynchronizeOutputs / GetOutputValues aquí

                } else {
                    // SHOW: output a CPU para visualizar (habrá D2H, inevitable si quieres cv::Mat)
                    binding.BindOutput(outputNames[depthIdx].c_str(), cpu_mem);
                    if (skyIdx >= 0) binding.BindOutput(outputNames[skyIdx].c_str(), cpu_mem);

                    double t0 = NowMs();
                    session.Run(Ort::RunOptions{nullptr}, binding);
                    binding.SynchronizeOutputs();
                    double t1 = NowMs();

                    if (idx >= warmup && (int)samples.size() < PROF_NFRAMES)
                        samples.push_back(t1 - t0);

                    auto outs = binding.GetOutputValues();

                    // Depth visualize (primer output bindeado)
                    auto& depthVal = outs[0];
                    const float* dptr = depthVal.GetTensorData<float>();
                    auto dshape = depthVal.GetTensorTypeAndShapeInfo().GetShape();

                    int outH = IN_H, outW = IN_W;
                    if (dshape.size() == 4) { outH = int(dshape[2]); outW = int(dshape[3]); }
                    else if (dshape.size() == 3) { outH = int(dshape[1]); outW = int(dshape[2]); }

                    cv::Mat depth(outH, outW, CV_32F, (void*)dptr);
                    cv::Mat depthVis = colorizeDepth(depth);

                    cv::Mat frameRS;
                    cv::resize(frame, frameRS, cv::Size(outW, outH), 0, 0, cv::INTER_LINEAR);

                    cv::imshow("Input", frameRS);
                    cv::imshow("Output", depthVis);
                    int key = cv::waitKey(1);
                    if (key == 27 || key == 'q') break;
                }

            } else {
                // CPU-only path
                inputCpuTensor = Ort::Value::CreateTensor<float>(
                    cpu_mem,
                    hInputPinned,                // aquí es malloc normal
                    inputElems,
                    inputShape.data(),
                    inputShape.size()
                );
                binding.BindInput(inputName.c_str(), inputCpuTensor);

                binding.BindOutput(outputNames[depthIdx].c_str(), cpu_mem);
                if (skyIdx >= 0) binding.BindOutput(outputNames[skyIdx].c_str(), cpu_mem);

                double t0 = NowMs();
                session.Run(Ort::RunOptions{nullptr}, binding);
                binding.SynchronizeOutputs();
                double t1 = NowMs();

                if (idx >= warmup && (int)samples.size() < PROF_NFRAMES)
                    samples.push_back(t1 - t0);

                if (show) {
                    auto outs = binding.GetOutputValues();
                    auto& depthVal = outs[0];
                    const float* dptr = depthVal.GetTensorData<float>();
                    auto dshape = depthVal.GetTensorTypeAndShapeInfo().GetShape();

                    int outH = IN_H, outW = IN_W;
                    if (dshape.size() == 4) { outH = int(dshape[2]); outW = int(dshape[3]); }
                    else if (dshape.size() == 3) { outH = int(dshape[1]); outW = int(dshape[2]); }

                    cv::Mat depth(outH, outW, CV_32F, (void*)dptr);
                    cv::Mat depthVis = colorizeDepth(depth);

                    cv::Mat frameRS;
                    cv::resize(frame, frameRS, cv::Size(outW, outH), 0, 0, cv::INTER_LINEAR);

                    cv::imshow("Input", frameRS);
                    cv::imshow("Output", depthVis);
                    int key = cv::waitKey(1);
                    if (key == 27 || key == 'q') break;
                }
            }

            idx++;
            if ((int)samples.size() >= PROF_NFRAMES) break;
        }

    } catch (const std::exception& e) {
        Logger(ERROR, "Exception: %s", e.what());
    }

    // Cleanup
    if (gpu) {
        if (dInput) CUDA_CHECK(cudaFree(dInput));
        if (hInputPinned) CUDA_CHECK(cudaFreeHost(hInputPinned));
    } else {
        if (hInputPinned) std::free(hInputPinned);
    }

    PrintPerf("ORT-DA3", samples);
    if (show) cv::destroyAllWindows();
}


/*
void OnnxInferDA3(const char * filename, const char * model_file, int64_t w, int64_t h, bool gpu, bool show){
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "da3");
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ---- CUDA EP ----
    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    cuda_opts.arena_extend_strategy = 0;
    cuda_opts.gpu_mem_limit = _CRT_SIZE_MAX;
    cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_opts.do_copy_in_default_stream = 1;
    so.AppendExecutionProvider_CUDA(cuda_opts);
    std::cout << "[ORT] CUDAExecutionProvider activo\n";

    Ort::Session session(env, ToWString(model_file).c_str(), so);
    Ort::AllocatorWithDefaultOptions allocator;

    {
        auto inNameAlloc = session.GetInputNameAllocated(0, allocator);
        std::string inputName = inNameAlloc.get();
        std::vector<std::string> outputNames;
        const size_t nOut = session.GetOutputCount();
        outputNames.reserve(nOut);
        for (size_t i = 0; i < nOut; ++i) {
            auto outNameAlloc = session.GetOutputNameAllocated(i, allocator);
            outputNames.emplace_back(outNameAlloc.get());
        }

        std::cout << "Input: " << inputName << "\nOutputs: ";
        for (auto& s : outputNames) std::cout << s << " ";
        std::cout << "\n";

        const int depthIdx = (findOutputIndex(outputNames, "depth") >= 0)
            ? findOutputIndex(outputNames, "depth") : 0;
        const int skyIdx = findOutputIndex(outputNames, "sky");
        cv::VideoCapture cap(filename);
        if (!cap.isOpened()) {
            Logger(ERROR, "Could not open video: %s", filename);
            return;
        }
        if (show) {
            cv::namedWindow("Input", cv::WINDOW_NORMAL);
            cv::namedWindow("Output", cv::WINDOW_NORMAL);
        }

        std::array<int64_t, 4> inputShape{1, 3, IN_H, IN_W};
        const size_t inputSize = 1ull * 3 * IN_H * IN_W;

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const char* input_names[] = { inputName.c_str() };

        std::vector<const char*> output_names;
        output_names.reserve(outputNames.size());
        for (auto& s : outputNames) output_names.push_back(s.c_str());
        cv::Mat frame;
        int idx = 0;
        std::vector<double> samples;
        int nframes = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        samples.reserve(nframes);

        while (cap.read(frame)) {
            if (frame.empty()) break;
            auto inputCHW = preprocessToNCHW(frame);

            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memInfo, inputCHW.data(), inputSize, inputShape.data(), inputShape.size()
            );

            double t0 = NowMs();
            auto outs = session.Run(
                Ort::RunOptions{nullptr},
                input_names, &inputTensor, 1,
                output_names.data(), output_names.size()
            );
            double t1 = NowMs();

            if (idx >= 10){
                samples.push_back(t1 - t0);
            }

            // Depth
            Ort::Value& depthVal = outs[depthIdx];
            float* dptr = depthVal.GetTensorMutableData<float>();
            auto dshape = depthVal.GetTensorTypeAndShapeInfo().GetShape();

            int outH = IN_H, outW = IN_W;
            if (dshape.size() == 4) { outH = int(dshape[2]); outW = int(dshape[3]); }
            else if (dshape.size() == 3) { outH = int(dshape[1]); outW = int(dshape[2]); }

            cv::Mat depth(outH, outW, CV_32F, dptr);
            cv::Mat depthVis = colorizeDepth(depth);

            cv::Mat frameRS;
            cv::resize(frame, frameRS, cv::Size(outW, outH), 0, 0, cv::INTER_LINEAR);

            // Sky (si existe)
            if (skyIdx >= 0) {
                Ort::Value& skyVal = outs[skyIdx];
                float* sptr = skyVal.GetTensorMutableData<float>();
                auto sshape = skyVal.GetTensorTypeAndShapeInfo().GetShape();

                int sH = outH, sW = outW;
                if (sshape.size() == 4) { sH = int(sshape[2]); sW = int(sshape[3]); }
                else if (sshape.size() == 3) { sH = int(sshape[1]); sW = int(sshape[2]); }

                cv::Mat sky(sH, sW, CV_32F, sptr);
                cv::Mat mask = (sky > 0.5f);
                mask.convertTo(mask, CV_8U, 255.0);

                cv::Mat maskBgr;
                cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
                cv::Mat smaller;
                cv::resize(maskBgr, smaller, cv::Size(outW / 3, outH / 3), 0, 0, cv::INTER_NEAREST);
                smaller.copyTo(depthVis(cv::Rect(0, 0, smaller.cols, smaller.rows)));
            }
            if (show) {
                cv::imshow("Input", frameRS);
                cv::imshow("Output", depthVis);
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q') break;
            }
            idx++;
        }
        PrintPerf("ORT", samples);
        if (show) cv::destroyAllWindows();
    }
}
*/
void OnnxInfer(const char * filename, const char * model_file, int64_t w, int64_t h, bool gpu, bool show){
    auto api = Ort::GetApi();
    model_config_t model_config;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_infer");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.EnableProfiling(L"ort_profile.json");
    session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    // session_options.SetLogSeverityLevel(1);

    if (gpu) {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_opts.gpu_mem_limit = SIZE_MAX;
        cuda_opts.arena_extend_strategy = 0;
        cuda_opts.do_copy_in_default_stream = 1;

        session_options.AppendExecutionProvider_CUDA(cuda_opts);
        Logger(INFO, "[ORT] CUDA => OK");
    }

    std::wstring wmodel = ToWString(model_file);
    Ort::Session session(env, wmodel.c_str(), session_options);
    Logger(INFO, "[ORT] sesion from (%s) => OK", model_file);

    Ort::AllocatorWithDefaultOptions allocator;
    PrintModelInputInfo(session, allocator, &model_config);

    // Video
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        Logger(ERROR, "Could not open video: %s", filename);
        return;
    }
    if (show) {
        cv::namedWindow("Input", cv::WINDOW_NORMAL);
        cv::namedWindow("Output", cv::WINDOW_NORMAL);
    }


    const int warmup = 10;
    // int nframes = PROF_NFRAMES;
    std::vector<double> samples;
    int nframes = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    samples.reserve(nframes);

    cv::Mat frame;
    int idx = 0;
    std::vector<uint8_t> input_u8;
    std::vector<Ort::Float16_t> input_f16;
    std::vector<float> input_f32;
    Ort::MemoryInfo cpu_mem  = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo cuda_mem{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

    Ort::IoBinding binding(session);
    while (cap.read(frame)) {
        if (frame.empty()) break;


        bool fixed_hw = true;
        int64_t mh = h, mw = w;
        if (model_config.size.size() >= 2) {
            int64_t sh = model_config.size[model_config.size.size() - 2];
            int64_t sw = model_config.size[model_config.size.size() - 1];
            if (sh > 0) mh = sh; else fixed_hw = false;
            if (sw > 0) mw = sw; else fixed_hw = false;
        }

        std::vector<int64_t> input_shape_runtime;

        // Crea tensor float32 NCHW o B,Nimg,C,H,W (si rank 5) con batch=1, num_images=1
        Ort::Value input_value = MakeInputTensor_F32_ImageLike(
            frame,
            /*runtime_w=*/mw,
            /*runtime_h=*/mh,
            /*model_shape_template=*/model_config.size,
            /*cpu_mem=*/cpu_mem,
            /*out_shape_runtime=*/input_shape_runtime,
            /*out_buffer=*/input_f32,
            /*batch_default=*/1,
            /*num_images_default=*/1,
            /*input_is_rgb=*/true,
            /*normalize_01=*/true
        );

        
        binding.ClearBoundInputs();
        binding.ClearBoundOutputs();

        binding.BindInput(model_config.input_name, input_value);

        if (gpu && !show) {
            // Para PERF: evita copia a CPU del output
            binding.BindOutput(model_config.output_name, cuda_mem);
        } else {
            // Para VISUALIZAR: necesitas output en CPU (o copiar tú)
            binding.BindOutput(model_config.output_name, cpu_mem);
        }

        Ort::RunOptions run_options;
        run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");

        double t0 = NowMs();
        session.Run(run_options, binding);
        binding.SynchronizeOutputs();
        double t1 = NowMs();

        if (idx >= warmup && (int)samples.size() < nframes)
            samples.push_back(t1 - t0);
        

        // Recuperar output
        auto outputs = binding.GetOutputValues();
        if (!outputs.empty() && outputs[0].IsTensor()) {
            auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
            auto out_shape = out_info.GetShape();
            auto out_type  = out_info.GetElementType();

            if (show) {
                cv::imshow("Input", frame);

                cv::Mat out_vis;
                if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    const float* out_ptr = outputs[0].GetTensorData<float>();
                    if (!TensorToDisplayMat_F32(out_ptr, out_shape, out_vis)) {
                        out_vis = frame.clone();
                        cv::putText(out_vis, "Unsupported output shape", {20,40},
                                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
                    }
                } else {
                    out_vis = frame.clone();
                    cv::putText(out_vis, "Output is not float32", {20,40},
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
                    if (idx == warmup) {
                        Logger(WARN, "ORT: output0 no es float32 (type=%d).", (int)out_type);
                    }
                }

                cv::imshow("Output", out_vis);
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') break;
            } else {
                // log 1 vez por si quieres ver shape real
                if (idx == 0) {
                    Logger(INFO, "ORT: output0 type=%d shape=[%lld,%lld,%lld,%lld] (si 4D)",
                            (int)out_type,
                            (long long)(out_shape.size()>0?out_shape[0]:-1),
                            (long long)(out_shape.size()>1?out_shape[1]:-1),
                            (long long)(out_shape.size()>2?out_shape[2]:-1),
                            (long long)(out_shape.size()>3?out_shape[3]:-1));
                }
            }
        } else {
            Logger(WARN, "ORT: no hay output tensor en outputs[0].");
        }

        idx++;
    }

    PrintPerf("ORT", samples);

    if (show) cv::destroyAllWindows();

}


static void PreprocessToNCHWFloat(const cv::Mat& bgr, int W, int H, std::vector<float>& out_nchw) {
    cv::Mat resized, f32;
    cv::resize(bgr, resized, cv::Size(W, H));
    resized.convertTo(f32, CV_32F, 1.0 / 255.0);

    // BGR -> RGB
    cv::cvtColor(f32, f32, cv::COLOR_BGR2RGB);

    // HWC -> NCHW
    out_nchw.resize(1ull * 3 * H * W);
    std::vector<cv::Mat> ch(3);
    cv::split(f32, ch);
    const int plane = H * W;
    std::memcpy(out_nchw.data() + 0 * plane, ch[0].ptr<float>(), plane * sizeof(float));
    std::memcpy(out_nchw.data() + 1 * plane, ch[1].ptr<float>(), plane * sizeof(float));
    std::memcpy(out_nchw.data() + 2 * plane, ch[2].ptr<float>(), plane * sizeof(float));
}
