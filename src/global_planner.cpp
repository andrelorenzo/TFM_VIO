#include "global_planner.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

struct GlobalPlannerState{
    GlobalPlannerConfig config;
    Waypoints raw_waypoints;
    Waypoints generated_path;
    bool initialized = false;
};

static GlobalPlannerState gplan;

static vec3 interpolateVec3(const vec3& p0, const vec3& p1, double alpha){
    return (1.0 - alpha)*p0 + alpha*p1;
}

static bool parseWaypointLine(const std::string& line, vec3 * point){
    if(point == nullptr){
        return false;
    }

    std::string clean = line;
    const std::size_t comment_pos = clean.find('#');
    if(comment_pos != std::string::npos){
        clean = clean.substr(0, comment_pos);
    }

    for(char& ch : clean){
        if(ch == ',' || ch == ';' || ch == '\t'){
            ch = ' ';
        }
    }

    std::istringstream iss(clean);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if(!(iss >> x >> y >> z)){
        return false;
    }

    *point = vec3(x, y, z);
    return true;
}

static Waypoints loadWaypointsFromFile(const std::string& path){
    Waypoints waypoints;

    if(path.empty()){
        Logger(WARN, "globalPlanInit: empty waypoint file path");
        return waypoints;
    }

    std::ifstream file(path);
    if(!file.is_open()){
        Logger(ERROR, "globalPlanInit: could not open waypoint file %s", path.c_str());
        return waypoints;
    }

    std::string line;
    int line_idx = 0;
    while(std::getline(file, line)){
        ++line_idx;

        vec3 point;
        if(parseWaypointLine(line, &point)){
            waypoints.p.push_back(point);
            continue;
        }

        const bool blank = std::all_of(line.begin(), line.end(), [](unsigned char c){
            return std::isspace(c) != 0;
        });
        const bool comment = !line.empty() && line.find_first_not_of(" \t") != std::string::npos &&
                             line[line.find_first_not_of(" \t")] == '#';
        if(!blank && !comment){
            Logger(WARN, "globalPlanInit: ignoring invalid waypoint line %d => %s", line_idx, line.c_str());
        }
    }

    return waypoints;
}

static Waypoints pathFromWaypoints3D(const Waypoints& waypoints, double resolution){
    Waypoints path;

    if(waypoints.p.empty()){
        return path;
    }

    if(waypoints.p.size() == 1){
        path = waypoints;
        return path;
    }

    const double safe_resolution = resolution > 1e-6 ? resolution : 1.0;

    for(std::size_t i = 0; i + 1 < waypoints.p.size(); ++i){
        const vec3& p0 = waypoints.p[i];
        const vec3& p1 = waypoints.p[i + 1];
        const double length = (p1 - p0).norm();
        const std::size_t n = std::max<std::size_t>(2, static_cast<std::size_t>(std::ceil(length / safe_resolution)));

        for(std::size_t k = 0; k < n; ++k){
            if(i > 0 && k == 0){
                continue;
            }

            const double alpha = (n <= 1) ? 0.0 : static_cast<double>(k) / static_cast<double>(n - 1);
            path.p.push_back(interpolateVec3(p0, p1, alpha));
        }
    }

    return path;
}

void globalPlanInit(const Config * config){
    gplan = GlobalPlannerState{};

    if(config == nullptr){
        Logger(ERROR, "globalPlanInit: null config");
        return;
    }

    gplan.config = config->gplan;
    gplan.raw_waypoints = loadWaypointsFromFile(gplan.config.waypoints_file);
    gplan.generated_path = pathFromWaypoints3D(gplan.raw_waypoints, gplan.config.resolution);
    gplan.initialized = true;

    Logger(INFO,    "globalPlanInit: waypoint_file=%s raw_waypoints=%d path_points=%d resolution=%.3f", gplan.config.waypoints_file.c_str(), static_cast<int>(gplan.raw_waypoints.p.size()),  static_cast<int>(gplan.generated_path.p.size()),  gplan.config.resolution);
}

void globalPlanUpdate(StateOut& state, Waypoints& path){
    if(!gplan.initialized){
        state.deb.gplan_waypoints.clear();
        state.deb.gplan_path.clear();
        return;
    }

    if(path.p.size() != gplan.generated_path.p.size()){
        path = gplan.generated_path;
    }

    state.deb.gplan_waypoints = gplan.raw_waypoints.p;
    state.deb.gplan_path = path.p;
}
