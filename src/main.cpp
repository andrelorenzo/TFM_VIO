#define LOGGER_IMP
#include "plotter.hpp"
#include "source_man2.hpp"
#include "csv_logger.hpp"
#include "da3.h"
#include "gt_est.hpp"
#include "vio_est.hpp"


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


void exitProgram(){
    int c = getc(stdin);
    exit(-1);
}

int main(int argc, char ** argv){
    projectInit(argc, argv);
    

    std::thread exitTh(exitProgram);
    exitTh.detach();

    // Init Logger
    std::vector<std::string> header(std::begin(DEBUG_HEADER), std::end(DEBUG_HEADER));
    CSVLogger logger(config.gen.output.c_str(), &header);

    // Init Modules
    initSource2(&config);
    gtInit(&config);
    vioInit(config);
    da3Init(&config);
    // globalPlanInit(&config);
    // localPlanInit(&config);
    // controllerInit(&config);
    // commanderInit(&config);
    initPlotters(&config);

    if(config.gen.show && config.gen.color_on){
        cv::namedWindow("vio", cv::WINDOW_NORMAL);
        cv::resizeWindow("vio", 1280, 720);
    }
    if(config.gen.show && config.da3.enabled && config.da3.show_window){
        cv::namedWindow("da3", cv::WINDOW_NORMAL);
        cv::resizeWindow("da3", config.da3.input_width * 2, config.da3.input_height * 2);
    }
    
    while(1){
        int ret = getSource2(&source);

        if (ret <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if(source.frame.empty())continue;

        da3Update(&source);
        state.da3 = da3Get();
        gtUpdate(&source, &state);
        if(!vioUpdate(&source, &state)){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        state.da3 = da3Get();
        // da3Update(source, state);            // pensar si meter en un thread a parte
        // // globalPlanUpdate(state, tray);
        // // localPlanUpdate(state, waypoints, tray);
        // // controllerUpdate(state, tray, cmd);

        // // commanderSend(cmd);              // Send command to drone

        if(config.gen.show){
            cv::Mat vio_debug = source.frame.clone();
            if(!vio_debug.empty() && vio_debug.cols > 0 && vio_debug.rows > 0){
                cv::imshow("vio", vio_debug);
            }
        }
        if(config.gen.show && config.da3.enabled && config.da3.show_window){
            cv::Mat da3_debug = da3GetDebugImage();
            if(!da3_debug.empty() && da3_debug.cols > 0 && da3_debug.rows > 0){
                cv::imshow("da3", da3_debug);
            }
        }
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;


        updatePlots(&state);
        if(!config.gen.output.empty())logger.addRow(state.toVector(config.gen.debug));    // Log
    }

    closePlotters();
    da3Close();
    vioClose();
    Logger(INFO, "Exiting succesfully, bye..");
    return 0;
}
