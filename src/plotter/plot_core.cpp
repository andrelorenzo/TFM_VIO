#include "plot_core.hpp"

#include <algorithm>
#include <cmath>

namespace plotlib {

PlotHandle PlotRegistry::createPlot(PlotConfig config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (config.id.empty()) {
        config.id = "plot_" + std::to_string(plots_.size());
    }
    if (config.title.empty()) {
        config.title = config.id;
    }
    if (config.window_title.empty()) {
        config.window_title = config.title;
    }

    plots_.push_back(PlotData{std::move(config), {}});
    return PlotHandle{plots_.size() - 1};
}

SeriesHandle PlotRegistry::addSeries(PlotHandle plot, SeriesConfig config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!plot.valid() || plot.index >= plots_.size()) {
        return {};
    }
    if (config.label.empty()) {
        config.label = "series_" + std::to_string(plots_[plot.index].series.size());
    }

    auto& series = plots_[plot.index].series;
    series.push_back(SeriesData{std::move(config), {}, {}});
    return SeriesHandle{plot.index, series.size() - 1};
}

void PlotRegistry::push(SeriesHandle series, double x, double y) {
    if (!series.valid()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (series.plot >= plots_.size()) {
        return;
    }

    auto& plot = plots_[series.plot];
    if (series.series >= plot.series.size()) {
        return;
    }

    if (!std::isfinite(x)) {
        return;
    }
    if (plot.config.skip_nan_points && !std::isfinite(y)) {
        return;
    }

    auto& s = plot.series[series.series];
    s.x.push_back(x);
    s.y.push_back(y);
    trim(s, plot.config.max_points);
}

void PlotRegistry::push(SeriesHandle series, Point p) {
    push(series, p.x, p.y);
}

void PlotRegistry::clearSamples() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& plot : plots_) {
        for (auto& series : plot.series) {
            series.x.clear();
            series.y.clear();
        }
    }
}

void PlotRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    plots_.clear();
}

bool PlotRegistry::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return plots_.empty();
}

std::size_t PlotRegistry::plotCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return plots_.size();
}

std::size_t PlotRegistry::sampleCount() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::size_t count = 0;
    for (const auto& plot : plots_) {
        for (const auto& series : plot.series) {
            count = std::max(count, series.x.size());
        }
    }
    return count;
}

std::vector<PlotData> PlotRegistry::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return plots_;
}

void PlotRegistry::renderAll() const {
    const auto plots = snapshot();
    for (const auto& plot : plots) {
        renderPlotWindow(plot);
    }
}

void PlotRegistry::trim(SeriesData& series, std::size_t max_points) {
    if (max_points == 0 || series.x.size() <= max_points) {
        return;
    }

    const auto extra = series.x.size() - max_points;
    series.x.erase(series.x.begin(), series.x.begin() + static_cast<std::ptrdiff_t>(extra));
    series.y.erase(series.y.begin(), series.y.begin() + static_cast<std::ptrdiff_t>(extra));
}

void PlotRegistry::renderPlotWidget(const PlotData& plot, const ImVec2& requested_size) {
    const auto& cfg = plot.config;

    ImVec2 size = requested_size;
    if (size.x == 0.0f || size.y == 0.0f) {
        size = cfg.size;
    }
    if (size.x == 0.0f || size.y == 0.0f) {
        size = ImGui::GetContentRegionAvail();
    }

    ImPlotFlags flags = cfg.plot_flags;
    if (cfg.equal) {
        flags |= ImPlotFlags_Equal;
    }
    if (!cfg.show_legend) {
        flags |= ImPlotFlags_NoLegend;
    }

    const std::string plot_id = cfg.title + "###" + cfg.id + "_plot";
    if (ImPlot::BeginPlot(plot_id.c_str(), size, flags)) {
        ImPlotAxisFlags x_flags = cfg.x_axis_flags;
        ImPlotAxisFlags y_flags = cfg.y_axis_flags;
        if (cfg.auto_fit_x) {
            x_flags |= ImPlotAxisFlags_AutoFit;
        }
        if (cfg.auto_fit_y) {
            y_flags |= ImPlotAxisFlags_AutoFit;
        }

        ImPlot::SetupAxes(cfg.x_label.c_str(), cfg.y_label.c_str(), x_flags, y_flags);

        for (const auto& series : plot.series) {
            if (!series.config.enabled || series.x.empty() || series.x.size() != series.y.size()) {
                continue;
            }

            if (series.config.line_weight > 0.0f && series.config.line_weight != 1.0f) {
                ImPlot::PlotLine(
                    series.config.label.c_str(),
                    series.x.data(),
                    series.y.data(),
                    static_cast<int>(series.x.size()),
                    ImPlotSpec(ImPlotProp_LineWeight, series.config.line_weight)
                );
            } else {
                ImPlot::PlotLine(
                    series.config.label.c_str(),
                    series.x.data(),
                    series.y.data(),
                    static_cast<int>(series.x.size())
                );
            }
        }

        ImPlot::EndPlot();
    }
}

void PlotRegistry::renderPlotWindow(const PlotData& plot) {
    const auto& cfg = plot.config;

    const std::string window_id = cfg.window_title + "###" + cfg.id + "_window";
    ImGui::Begin(window_id.c_str());

    renderPlotWidget(plot);

    ImGui::End();
}

} // namespace plotlib
