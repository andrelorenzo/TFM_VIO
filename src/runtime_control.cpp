#include "runtime_control.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace {

std::atomic<bool> g_stop_requested{false};
std::atomic<bool> g_pause_requested{false};
std::mutex g_pause_mutex;
std::condition_variable g_pause_cv;

} // namespace

bool runtimeStopRequested() {
    return g_stop_requested.load();
}

void runtimeRequestStop() {
    g_stop_requested.store(true);
    g_pause_cv.notify_all();
}

void runtimeResetControl() {
    g_stop_requested.store(false);
    g_pause_requested.store(false);
    g_pause_cv.notify_all();
}

bool runtimeIsPaused() {
    return g_pause_requested.load();
}

void runtimeSetPaused(bool paused) {
    g_pause_requested.store(paused);
    if (!paused) {
        g_pause_cv.notify_all();
    }
}

void runtimeTogglePaused() {
    runtimeSetPaused(!runtimeIsPaused());
}

bool runtimeWaitIfPaused() {
    if (!runtimeIsPaused()) {
        return runtimeStopRequested();
    }

    std::unique_lock<std::mutex> lock(g_pause_mutex);
    g_pause_cv.wait(lock, [] {
        return runtimeStopRequested() || !runtimeIsPaused();
    });
    return runtimeStopRequested();
}
