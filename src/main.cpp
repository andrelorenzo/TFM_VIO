#include <windows.h>
#undef ERROR
#define LOGGER_IMP
#define NOMINMAX
#include "plotter.hpp"
#include "source_man2.hpp"
#include "csv_logger.hpp"
#include "da3.h"
#include "gt_est.hpp"
#include "vio_est.hpp"
#include "controller.hpp"
#include "commander.hpp"
#include "global_planner.hpp"
#include "local_planner.hpp"
#include "perf_monitor.hpp"
#include "runtime_control.hpp"
#include <csignal>

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <thread>


extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

static Config config;
static SourceIn source;
static StateOut state;
static Command cmd;
static Waypoints path;

static double elapsedMs(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static BOOL WINAPI handleConsoleSignal(DWORD signal){
    switch(signal){
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            runtimeRequestStop();
            return TRUE;
        default:
            return FALSE;
    }
}

void projectInit(int argc, char ** argv){
    LoggerSetVerbsity(DEBUG);

    if(argc < 2){
        Logger(ERROR, "Usage: %s <path-to-config.yaml>", argv[0]);
        exit(-1);
    }
    bool ok = config.parseYAML(argv[1]);

    

    if(!ok)exit(-1);

    config.print();
}

void signalHandler(int sig)
{
    printf("Exiting\n");

    runtimeRequestStop();
    runtimeSetPaused(false);
    commanderClose();
    closeSource2();
    closePlotters();
    da3Close();
    vioClose();
    SetConsoleCtrlHandler(handleConsoleSignal, FALSE);

    exit(sig);
}


int main(int argc, char ** argv){
    projectInit(argc, argv);
    runtimeResetControl();
    SetConsoleCtrlHandler(handleConsoleSignal, TRUE);

    signal(SIGINT, signalHandler);
    const bool csv_imu_only = config.gen.type == SOURCE_CSV;

    // Init Logger
    std::vector<std::string> header(std::begin(DEBUG_HEADER), std::end(DEBUG_HEADER));
    CSVLogger logger(config.gen.output.c_str(), &header);

    // Init Modules
    if(!initSource2(&config)){
        Logger(ERROR, "Source initialization failed");
        return -1;
    }
    gtInit(&config);
    vioInit(config);
    da3Init(&config);
    globalPlanInit(&config);
    localPlannerInit(&config);
    controllerInit(&config);
    commanderInit(&config);
    initPlotters(&config);

    if(config.gen.show && config.gen.color_on){
        cv::namedWindow("vio", cv::WINDOW_NORMAL);
        cv::resizeWindow("vio", 1280, 720);
    }
    if(config.gen.show && config.da3.enabled && config.da3.show_window){
        cv::namedWindow("da3", cv::WINDOW_NORMAL);
        cv::resizeWindow("da3", config.da3.input_width * 2, config.da3.input_height * 2);
    }

    cv::Mat last_vio_debug;
    cv::Mat last_da3_debug;
    PerfMonitor perf_monitor(config.gen.fps_stats_on, config.gen.fps_stats_period_s);

    auto handleUiKey = [&](int raw_key) {
        const int key = raw_key & 0xff;
        if (key == 27 || key == 'q' || key == 'Q') {
            runtimeRequestStop();
            return;
        }

        if (key == 'p' || key == 'P' || key == ' ') {
            const bool next_paused = !runtimeIsPaused();
            runtimeSetPaused(next_paused);
            Logger(INFO,
                   next_paused ? "Execution paused. Press P or SPACE to resume."
                               : "Execution resumed.");
        }
    };

    auto makePausedOverlay = [](const cv::Mat& input) {
        if (input.empty()) {
            return cv::Mat{};
        }

        cv::Mat output = input.clone();
        const cv::Rect banner(0, 0, output.cols, std::min(56, output.rows));
        cv::rectangle(output, banner, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(output, "PAUSADO", cv::Point(16, 22), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::putText(output, "P/SPACE reanuda  Q/ESC sale", cv::Point(16, 46), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        return output;
    };

    auto addPerfOverlay = [&](cv::Mat& image) {
        const std::vector<std::string> lines = perf_monitor.overlayLines();
        if (image.empty() || lines.empty()) {
            return;
        }

        const int margin = 12;
        const int line_gap = 6;
        const double font_scale = 0.55;
        const int thickness = 1;
        const int baseline_pad = 6;
        int max_width = 0;
        int text_height = 0;

        for (const std::string& line : lines) {
            int baseline = 0;
            const cv::Size sz = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
            max_width = std::max(max_width, sz.width);
            text_height = std::max(text_height, sz.height + baseline);
        }

        const int box_width = std::min(image.cols - 2 * margin, max_width + 2 * margin);
        const int box_height = static_cast<int>(lines.size()) * text_height +
                               static_cast<int>(lines.size() - 1) * line_gap +
                               2 * baseline_pad;
        const int x0 = margin;
        const int y0 = std::max(margin, image.rows - box_height - margin);

        cv::rectangle(image,
                      cv::Rect(x0, y0, std::max(1, box_width), std::max(1, box_height)),
                      cv::Scalar(0, 0, 0),
                      cv::FILLED);

        int y = y0 + baseline_pad + text_height - 4;
        for (const std::string& line : lines) {
            cv::putText(image,
                        line,
                        cv::Point(x0 + baseline_pad, y),
                        cv::FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        cv::Scalar(0, 255, 255),
                        thickness,
                        cv::LINE_AA);
            y += text_height + line_gap;
        }
    };

    while(!runtimeStopRequested()){
        if (runtimeIsPaused()) {
            if (config.gen.show && !csv_imu_only && !last_vio_debug.empty()) {
                cv::imshow("vio", makePausedOverlay(last_vio_debug));
            }
            if (config.gen.show && config.da3.enabled && config.da3.show_window && !last_da3_debug.empty()) {
                cv::imshow("da3", makePausedOverlay(last_da3_debug));
            }

            handleUiKey(cv::waitKey(30));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        const auto frame_begin = std::chrono::steady_clock::now();
        int ret = getSource2(&source);
        const auto after_source = std::chrono::steady_clock::now();

        if (ret < 0) {
            break;
        }
        if (ret == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if(!csv_imu_only && source.frame.empty()) continue;
        perf_monitor.recordStage("source", elapsedMs(frame_begin, after_source));

        if(!csv_imu_only) {
            const auto da3_begin = std::chrono::steady_clock::now();
            da3Update(&source);
            perf_monitor.recordStage("da3", elapsedMs(da3_begin, std::chrono::steady_clock::now()));
        }

        const auto gt_begin = std::chrono::steady_clock::now();
        state.da3 = da3Get();
        gtUpdate(&source, &state);
        perf_monitor.recordStage("gt", elapsedMs(gt_begin, std::chrono::steady_clock::now()));

        const auto vio_begin = std::chrono::steady_clock::now();
        if(!vioUpdate(&source, &state)){
            perf_monitor.recordStage("vio", elapsedMs(vio_begin, std::chrono::steady_clock::now()));
            perf_monitor.finishFrame(elapsedMs(frame_begin, std::chrono::steady_clock::now()));
            perf_monitor.printLiveIfDue();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        perf_monitor.recordStage("vio", elapsedMs(vio_begin, std::chrono::steady_clock::now()));

        if(!csv_imu_only){
            const auto control_begin = std::chrono::steady_clock::now();
            state.da3 = da3Get();
            globalPlanUpdate(state, path);
            localPlannerUpdate(state.da3, state, path, &cmd);
            controllerUpdate(state, &cmd);
            commanderSend(cmd);
            perf_monitor.recordStage("control", elapsedMs(control_begin, std::chrono::steady_clock::now()));
        }

        const auto render_begin = std::chrono::steady_clock::now();
        if(config.gen.show && !csv_imu_only){
            cv::Mat vio_debug = getDebugImage();
            if (vio_debug.empty()) {
                vio_debug = source.frame.clone();
            }
            if(!vio_debug.empty() && vio_debug.cols > 0 && vio_debug.rows > 0){
                addPerfOverlay(vio_debug);
                last_vio_debug = vio_debug.clone();
                cv::imshow("vio", vio_debug);
            }
        }
        if(config.gen.show && config.da3.enabled && config.da3.show_window){
            cv::Mat da3_debug = da3GetDebugImage();
            if(!da3_debug.empty() && da3_debug.cols > 0 && da3_debug.rows > 0){
                addPerfOverlay(da3_debug);
                last_da3_debug = da3_debug.clone();
                cv::imshow("da3", da3_debug);
            }
        }
        handleUiKey(cv::waitKey(1));
        perf_monitor.recordStage("render", elapsedMs(render_begin, std::chrono::steady_clock::now()));


        const auto plotlog_begin = std::chrono::steady_clock::now();
        updatePlots(&state);
        if(!config.gen.output.empty())logger.addRow(state.toVector(config.gen.debug));    // Log
        perf_monitor.recordStage("plots_log", elapsedMs(plotlog_begin, std::chrono::steady_clock::now()));
        perf_monitor.finishFrame(elapsedMs(frame_begin, std::chrono::steady_clock::now()));
        perf_monitor.printLiveIfDue();
    }

    runtimeSetPaused(false);
    commanderClose();
    closeSource2();
    closePlotters();
    da3Close();
    vioClose();
    SetConsoleCtrlHandler(handleConsoleSignal, FALSE);
    perf_monitor.printSummary();
    Logger(INFO, "Exiting succesfully, bye..");
    return 0;
}
