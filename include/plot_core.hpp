#pragma once

#include "imgui.h"
#include "implot.h"

#include <cstddef>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace plotlib {

constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

struct PlotHandle {
    std::size_t index = npos;
    bool valid() const { return index != npos; }
};

struct SeriesHandle {
    std::size_t plot = npos;
    std::size_t series = npos;
    bool valid() const { return plot != npos && series != npos; }
};

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct PlotConfig {
    std::string id;
    std::string window_title;
    std::string title;
    std::string x_label = "x";
    std::string y_label = "y";

    bool auto_fit_x = true;
    bool auto_fit_y = true;
    bool equal = false;
    bool show_legend = true;
    bool skip_nan_points = false;

    ImVec2 size = ImVec2(-1.0f, 0.0f);
    ImPlotFlags plot_flags = ImPlotFlags_None;
    ImPlotAxisFlags x_axis_flags = ImPlotAxisFlags_None;
    ImPlotAxisFlags y_axis_flags = ImPlotAxisFlags_None;

    std::size_t max_points = 20000;
};

struct SeriesConfig {
    std::string label;
    float line_weight = 1.0f;
    bool enabled = true;
};

struct SeriesData {
    SeriesConfig config;
    std::vector<double> x;
    std::vector<double> y;
};

struct PlotData {
    PlotConfig config;
    std::vector<SeriesData> series;
};

class PlotRegistry {
public:
    PlotHandle createPlot(PlotConfig config);
    SeriesHandle addSeries(PlotHandle plot, SeriesConfig config);

    void push(SeriesHandle series, double x, double y);
    void push(SeriesHandle series, Point p);

    void clearSamples();
    void reset();

    bool empty() const;
    std::size_t plotCount() const;
    std::size_t sampleCount() const;

    std::vector<PlotData> snapshot() const;
    void renderAll() const;
    static void renderPlotWidget(const PlotData& plot, const ImVec2& size = ImVec2(-1.0f, 0.0f));

private:
    static void trim(SeriesData& series, std::size_t max_points);
    static void renderPlotWindow(const PlotData& plot);

    mutable std::mutex mutex_;
    std::vector<PlotData> plots_;
};

} // namespace plotlib
