#define LOGGER_IMP
#include "plotter.hpp"
#include "source_man2.hpp"
#include "csv_logger.hpp"
#include "gt_est.hpp"
#include "vio_est.hpp"


#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <thread>

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
    if(config.gen.debug) {
        std::vector<std::string> deb_h(std::begin(DEBUG_HEADER), std::end(DEBUG_HEADER));
        header.insert(header.end(), deb_h.begin(), deb_h.end());
    }
    CSVLogger logger(config.gen.output.c_str(), &header);

    // Init Modules
    initSource2(&config);
    gtInit(&config);
    vioInit(config);
    // da3Init(&config);
    // globalPlanInit(&config);
    // localPlanInit(&config);
    // controllerInit(&config);
    // commanderInit(&config);
    initPlotters(&config);

    if(config.gen.show && config.gen.color_on){
        cv::namedWindow("vio", cv::WINDOW_NORMAL);
        cv::resizeWindow("vio", 1280, 720);
    }
    
    while(1){
        int ret = getSource2(&source);

        if (ret <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if(source.frame.empty())continue;


        gtUpdate(&source, &state);
        if(!vioUpdate(&source, &state)){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        // // da3Update(source, state);            // pensar si meter en un thread a parte
        // // globalPlanUpdate(state, tray);
        // // localPlanUpdate(state, waypoints, tray);
        // // controllerUpdate(state, tray, cmd);

        // // commanderSend(cmd);              // Send command to drone

        if(config.gen.show && !source.frame.empty())cv::imshow("vio", source.frame);
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;


        updatePlots(&state);
        if(!config.gen.output.empty())logger.addRow(state.toVector(config.gen.debug));    // Log
        // // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // // printf("\x1B[2J\x1B[H");
        // if(ret != SOURCEMAN_OK){
        //     Logger(ERROR, "Ret number: %i", ret);
        //     break; // Capture Sources
        // }

    }

    closePlotters();
    vioClose();
    Logger(INFO, "Exiting succesfully, bye..");
    return 0;
}
