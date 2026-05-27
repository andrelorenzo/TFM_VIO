#pragma once

#include "plot_core.hpp"
#include "config.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace vioplot {

enum class TrajectoryPlane {
    XY,
    XZ,
    YZ
};

struct VioPlotterOptions {
    std::size_t max_points = 20000;
    TrajectoryPlane trajectory_plane = TrajectoryPlane::XZ;
    bool equal_trajectory_axes = false;
};

class VioPlotter {
public:
    void configure(const Config& config, VioPlotterOptions options = {});
    void update(const StateOut& state);
    void render();
    void clearSamples();
    void reset();

    bool enabled() const { return enabled_; }
    std::size_t plotCount() const { return plots_.plotCount(); }
    std::size_t sampleCount() const { return plots_.sampleCount(); }

private:
    struct Binding {
        plotlib::SeriesHandle series;
        std::function<plotlib::Point(const StateOut&)> sample;
    };

    struct CollectionBinding {
        plotlib::SeriesHandle series;
        std::function<std::vector<plotlib::Point>(const StateOut&)> sample;
    };

    plotlib::PlotHandle addTimePlot(const char* id, const char* title, const char* y_label);
    plotlib::PlotHandle addTrajectoryPlot(const char* id, const char* title);

    void bind(const plotlib::PlotHandle& plot,
              const char* label,
              std::function<plotlib::Point(const StateOut&)> sample,
              float line_weight = 1.0f,
              bool scatter = false);

    void bindCollection(const plotlib::PlotHandle& plot,
                        const char* label,
                        std::function<std::vector<plotlib::Point>(const StateOut&)> sample,
                        float line_weight = 1.0f,
                        bool scatter = false);

    void bindVec3Time(const plotlib::PlotHandle& plot,
                      const char* prefix,
                      std::function<vec3(const StateOut&)> value);

    void bindRpyTime(const plotlib::PlotHandle& plot,
                     const char* prefix,
                     std::function<vec3(const StateOut&)> value);

    void bindTrajectory(const plotlib::PlotHandle& plot,
                        const char* label,
                        std::function<vec3(const StateOut&)> position);

    void bindTrajectoryCollection(const plotlib::PlotHandle& plot,
                                  const char* label,
                                  std::function<std::vector<vec3>(const StateOut&)> positions,
                                  float line_weight = 1.0f,
                                  bool scatter = false);

    static double timeSeconds(const StateOut& state);
    static vec3 poseRpy(Pose pose);
    static double preimuAt(const StateOut& state, int index);

    plotlib::Point projectTrajectory(const vec3& p) const;
    const char* trajectoryXLabel() const;
    const char* trajectoryYLabel() const;
    void renderDashboard();

    plotlib::PlotRegistry plots_;
    std::vector<Binding> bindings_;
    std::vector<CollectionBinding> collection_bindings_;
    VioPlotterOptions options_;
    bool enabled_ = false;
    int dashboard_columns_ = 2;
    std::size_t focused_plot_ = plotlib::npos;
};

} // namespace vioplot
