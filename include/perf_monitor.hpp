#pragma once

#include <chrono>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

class PerfMonitor {
public:
    PerfMonitor(bool enabled, double live_print_period_s);

    bool enabled() const;
    void recordStage(const std::string& name, double ms);
    void finishFrame(double total_ms);
    void printLiveIfDue();
    std::vector<std::string> overlayLines() const;
    void printSummary(std::ostream& os = std::cout) const;

private:
    struct MetricStats {
        std::size_t samples = 0;
        double current_ms = 0.0;
        double sum_ms = 0.0;
        double sum_sq_ms = 0.0;
        double min_ms = 0.0;
        double max_ms = 0.0;

        void add(double ms);
        double avgMs() const;
        double stdMs() const;
        double avgFps() const;
        double minFps() const;
        double maxFps() const;
        bool hasSamples() const;
    };

    MetricStats& accessMetric(const std::string& name);
    const MetricStats* findMetric(const std::string& name) const;
    static double fpsFromMs(double ms);
    static std::string formatDouble(double value, int precision = 2);

    bool enabled_ = false;
    double live_print_period_s_ = 1.0;
    std::uint64_t frame_count_ = 0;
    std::chrono::steady_clock::time_point session_start_;
    std::chrono::steady_clock::time_point last_live_print_;
    std::vector<std::string> metric_order_;
    std::map<std::string, MetricStats> metrics_;
};
