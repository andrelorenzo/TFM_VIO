#include "perf_monitor.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
double fpsFromMilliseconds(double ms) {
    return (std::isfinite(ms) && ms > 1e-9) ? (1000.0 / ms) : 0.0;
}
}

void PerfMonitor::MetricStats::add(double ms) {
    current_ms = ms;
    sum_ms += ms;
    sum_sq_ms += ms * ms;
    if (samples == 0) {
        min_ms = ms;
        max_ms = ms;
    } else {
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }
    ++samples;
}

double PerfMonitor::MetricStats::avgMs() const {
    return samples > 0 ? (sum_ms / static_cast<double>(samples)) : 0.0;
}

double PerfMonitor::MetricStats::stdMs() const {
    if (samples < 2) {
        return 0.0;
    }
    const double mean = avgMs();
    const double variance = std::max(0.0, (sum_sq_ms / static_cast<double>(samples)) - (mean * mean));
    return std::sqrt(variance);
}

double PerfMonitor::MetricStats::avgFps() const {
    return fpsFromMilliseconds(avgMs());
}

double PerfMonitor::MetricStats::minFps() const {
    return fpsFromMilliseconds(max_ms);
}

double PerfMonitor::MetricStats::maxFps() const {
    return fpsFromMilliseconds(min_ms);
}

bool PerfMonitor::MetricStats::hasSamples() const {
    return samples > 0;
}

PerfMonitor::PerfMonitor(bool enabled, double live_print_period_s)
    : enabled_(enabled),
      live_print_period_s_(std::max(0.1, live_print_period_s)),
      session_start_(std::chrono::steady_clock::now()),
      last_live_print_(session_start_) {}

bool PerfMonitor::enabled() const {
    return enabled_;
}

void PerfMonitor::recordStage(const std::string& name, double ms) {
    if (!enabled_) {
        return;
    }
    accessMetric(name).add(ms);
}

void PerfMonitor::finishFrame(double total_ms) {
    if (!enabled_) {
        return;
    }
    recordStage("total", total_ms);
    ++frame_count_;
}

void PerfMonitor::printLiveIfDue() {
    if (!enabled_) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>(now - last_live_print_).count();
    if (elapsed_s < live_print_period_s_) {
        return;
    }
    last_live_print_ = now;

    const MetricStats* total = findMetric("total");
    if (total == nullptr || !total->hasSamples()) {
        return;
    }

    std::ostringstream oss;
    oss << "[FPS] total cur=" << formatDouble(fpsFromMs(total->current_ms))
        << " avg=" << formatDouble(total->avgFps())
        << " min=" << formatDouble(total->minFps())
        << " max=" << formatDouble(total->maxFps());

    const auto appendStage = [&](const char* name, const char* label) {
        const MetricStats* metric = findMetric(name);
        if (metric != nullptr && metric->hasSamples()) {
            oss << " | " << label << "=" << formatDouble(metric->avgMs()) << "ms";
        }
    };

    appendStage("source", "src");
    appendStage("da3", "da3");
    appendStage("vio", "vio");
    appendStage("control", "ctrl");

    std::cout << oss.str() << std::endl;
}

std::vector<std::string> PerfMonitor::overlayLines() const {
    if (!enabled_) {
        return {};
    }

    const MetricStats* total = findMetric("total");
    if (total == nullptr || !total->hasSamples()) {
        return {};
    }

    std::vector<std::string> lines;
    std::ostringstream line1;
    line1 << "FPS total " << formatDouble(fpsFromMs(total->current_ms))
          << " | avg " << formatDouble(total->avgFps())
          << " | min " << formatDouble(total->minFps())
          << " | max " << formatDouble(total->maxFps());
    lines.push_back(line1.str());

    std::ostringstream line2;
    line2 << "ms";
    const auto appendCurrentStage = [&](const char* name, const char* label) {
        const MetricStats* metric = findMetric(name);
        if (metric != nullptr && metric->hasSamples()) {
            line2 << "  " << label << " " << formatDouble(metric->current_ms, 1);
        }
    };

    appendCurrentStage("source", "src");
    appendCurrentStage("da3", "da3");
    appendCurrentStage("gt", "gt");
    appendCurrentStage("vio", "vio");
    appendCurrentStage("control", "ctrl");
    appendCurrentStage("render", "ui");
    lines.push_back(line2.str());

    return lines;
}

void PerfMonitor::printSummary(std::ostream& os) const {
    if (!enabled_) {
        return;
    }

    const MetricStats* total = findMetric("total");
    if (total == nullptr || !total->hasSamples()) {
        os << "\n========== FPS SUMMARY ==========\n";
        os << "No se han recogido muestras.\n";
        os << "=================================\n";
        return;
    }

    os << "\n================ FPS SUMMARY ================\n";
    os << "Frames procesados: " << frame_count_ << "\n";
    os << std::left
       << std::setw(12) << "Metric"
       << std::setw(10) << "Samples"
       << std::setw(12) << "Avg ms"
       << std::setw(12) << "Min ms"
       << std::setw(12) << "Max ms"
       << std::setw(12) << "Std ms"
       << std::setw(12) << "Avg FPS"
       << std::setw(12) << "Min FPS"
       << std::setw(12) << "Max FPS"
       << "\n";

    for (const std::string& name : metric_order_) {
        const auto it = metrics_.find(name);
        if (it == metrics_.end() || !it->second.hasSamples()) {
            continue;
        }

        const MetricStats& metric = it->second;
        os << std::left
           << std::setw(12) << name
           << std::setw(10) << metric.samples
           << std::setw(12) << formatDouble(metric.avgMs())
           << std::setw(12) << formatDouble(metric.min_ms)
           << std::setw(12) << formatDouble(metric.max_ms)
           << std::setw(12) << formatDouble(metric.stdMs())
           << std::setw(12) << formatDouble(metric.avgFps())
           << std::setw(12) << formatDouble(metric.minFps())
           << std::setw(12) << formatDouble(metric.maxFps())
           << "\n";
    }

    os << "=============================================\n";
}

PerfMonitor::MetricStats& PerfMonitor::accessMetric(const std::string& name) {
    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
        metric_order_.push_back(name);
        it = metrics_.emplace(name, MetricStats{}).first;
    }
    return it->second;
}

const PerfMonitor::MetricStats* PerfMonitor::findMetric(const std::string& name) const {
    const auto it = metrics_.find(name);
    return it != metrics_.end() ? &it->second : nullptr;
}

double PerfMonitor::fpsFromMs(double ms) {
    return (std::isfinite(ms) && ms > 1e-9) ? (1000.0 / ms) : 0.0;
}

std::string PerfMonitor::formatDouble(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}
