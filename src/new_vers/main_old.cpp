#define LOGGER_IMP
#include "plotter.hpp"
#include "source_man2.hpp"
#include "csv_logger.hpp"
// #include "gt_est.hpp"
#include "pre_int.hpp"
#include "vio_update.hpp"
// #include "lie_math.hpp"

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
        Logger(ERROR, "Usage: %s <path-to-yaml>", argv[0]);
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
    std::vector<std::string> header(std::begin(STATEOUT_HEADER), std::end(STATEOUT_HEADER));
    if(config.gen.debug) {
        std::vector<std::string> deb_h(std::begin(DEBUGSTATE_HEADER), std::end(DEBUGSTATE_HEADER));
        header.insert(header.end(), deb_h.begin(), deb_h.end());
    }
    CSVLogger logger(config.gen.output.c_str(), &header);

    // Init Modules
    // initSourceMan(&config);
    initSource2(config);
    // gtInit(&config);
    imuPreInit(&config);
    vioInit(&config);
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
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // SourceManType ret = getSourceMan(&source);
        // printf("\x1B[2J\x1B[H");
        int ret = getSource2(&source);
        // gtUpdate(&source, &state);
        imuPreUpdate(&source, &state);
        vioUpdate(&source, &state);


        
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
    Logger(INFO, "Exiting succesfully, bye..");
    return 0;
}
