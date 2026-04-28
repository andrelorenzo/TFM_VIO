#include "plotter.hpp"

#include "seconds/logger.h"

#include <windows.h>
#undef ERROR
#include <d3d11.h>
#include <tchar.h>

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include "implot.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "d3d11.lib")
extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

constexpr size_t PLOT_MAX_POINTS = 20000;
constexpr ImPlotFlags PLOT_FLAGS_DEFAULT = ImPlotFlags_Equal;
constexpr float TRAJECTORY_LINE_WEIGHT = 2.0f;

// -------------------------
// Globals
// -------------------------
static Config* g_plot_cfg = nullptr;
static std::atomic<bool> g_plot_on{false};
static std::atomic<bool> g_plot_running{false};
static std::atomic<bool> g_plot_ready{false};

static std::thread g_plot_thread;
static std::mutex g_plot_mutex;
static HWND g_plot_hwnd = nullptr;

// D3D
static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

// -------------------------
// Data buffers
// -------------------------
static std::vector<double> g_t;

static std::vector<double> g_vio_x, g_vio_y, g_vio_z;
static std::vector<double> g_vis_x, g_vis_y, g_vis_z;
static std::vector<double> g_imu_x, g_imu_y, g_imu_z;
static std::vector<double> g_gt_x, g_gt_y, g_gt_z;

static std::vector<double> g_gx_raw, g_gy_raw, g_gz_raw;
static std::vector<double> g_ax_raw, g_ay_raw, g_az_raw;
static std::vector<double> g_gx_cal, g_gy_cal, g_gz_cal;
static std::vector<double> g_ax_cal, g_ay_cal, g_az_cal;

static std::vector<double> g_rpy_r, g_rpy_p, g_rpy_yaw;
static std::vector<double> g_vis_rpy_r, g_vis_rpy_p, g_vis_rpy_yaw;
static std::vector<double> g_gt_r, g_gt_p, g_gt_yaw;

// -------------------------
// Snapshot for rendering
// -------------------------
struct PlotSnapshot {
    std::vector<double> t;

    std::vector<double> vio_x, vio_y, vio_z;
    std::vector<double> vis_x, vis_y, vis_z;
    std::vector<double> imu_x, imu_y, imu_z;
    std::vector<double> gt_x, gt_y, gt_z;

    std::vector<double> gx_raw, gy_raw, gz_raw;
    std::vector<double> ax_raw, ay_raw, az_raw;
    std::vector<double> gx_cal, gy_cal, gz_cal;
    std::vector<double> ax_cal, ay_cal, az_cal;

    std::vector<double> rpy_r, rpy_p, rpy_yaw;
    std::vector<double> vis_rpy_r, vis_rpy_p, vis_rpy_yaw;
    std::vector<double> gt_r, gt_p, gt_yaw;
};

// -------------------------
// Helpers
// -------------------------
template <typename T>
static void trimVector(std::vector<T>& v, size_t max_points = PLOT_MAX_POINTS) {
    if (v.size() > max_points) {
        const size_t extra = v.size() - max_points;
        v.erase(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(extra));
    }
}

template <typename T>
static void pushTrim(std::vector<T>& v, const T& value, size_t max_points = PLOT_MAX_POINTS) {
    v.emplace_back(value);
    trimVector(v, max_points);
}

static void clearAllPlotBuffers() {
    std::lock_guard<std::mutex> lock(g_plot_mutex);

    g_t.clear();

    g_vio_x.clear(); g_vio_y.clear(); g_vio_z.clear();
    g_vis_x.clear(); g_vis_y.clear(); g_vis_z.clear();
    g_imu_x.clear(); g_imu_y.clear(); g_imu_z.clear();
    g_gt_x.clear();  g_gt_y.clear();  g_gt_z.clear();

    g_gx_raw.clear(); g_gy_raw.clear(); g_gz_raw.clear();
    g_ax_raw.clear(); g_ay_raw.clear(); g_az_raw.clear();
    g_gx_cal.clear(); g_gy_cal.clear(); g_gz_cal.clear();
    g_ax_cal.clear(); g_ay_cal.clear(); g_az_cal.clear();

    g_rpy_r.clear(); g_rpy_p.clear(); g_rpy_yaw.clear();
    g_vis_rpy_r.clear(); g_vis_rpy_p.clear(); g_vis_rpy_yaw.clear();
    g_gt_r.clear();  g_gt_p.clear();  g_gt_yaw.clear();
}

static PlotSnapshot makeSnapshot() {
    std::lock_guard<std::mutex> lock(g_plot_mutex);

    PlotSnapshot s;
    s.t = g_t;

    s.vio_x = g_vio_x; s.vio_y = g_vio_y; s.vio_z = g_vio_z;
    s.vis_x = g_vis_x; s.vis_y = g_vis_y; s.vis_z = g_vis_z;
    s.imu_x = g_imu_x; s.imu_y = g_imu_y; s.imu_z = g_imu_z;
    s.gt_x  = g_gt_x;  s.gt_y  = g_gt_y;  s.gt_z  = g_gt_z;

    s.gx_raw = g_gx_raw; s.gy_raw = g_gy_raw; s.gz_raw = g_gz_raw;
    s.ax_raw = g_ax_raw; s.ay_raw = g_ay_raw; s.az_raw = g_az_raw;
    s.gx_cal = g_gx_cal; s.gy_cal = g_gy_cal; s.gz_cal = g_gz_cal;
    s.ax_cal = g_ax_cal; s.ay_cal = g_ay_cal; s.az_cal = g_az_cal;

    s.rpy_r = g_rpy_r; s.rpy_p = g_rpy_p; s.rpy_yaw = g_rpy_yaw;
    s.vis_rpy_r = g_vis_rpy_r; s.vis_rpy_p = g_vis_rpy_p; s.vis_rpy_yaw = g_vis_rpy_yaw;
    s.gt_r  = g_gt_r;  s.gt_p  = g_gt_p;  s.gt_yaw = g_gt_yaw;

    return s;
}

static bool wantAnyPlots(const Config* cfg) {
    if (cfg == nullptr) return false;
    return
        cfg->gen.plot_tray ||
        cfg->gen.plot_vis_tray ||
        cfg->gen.plot_imu_tray ||
        cfg->gen.plot_gt_with_tray ||
        cfg->gen.plot_gt_with_vis_tray ||
        cfg->gen.plot_gt_with_imu_tray ||
        cfg->gen.plot_imu ||
        cfg->gen.plot_rpy ||
        cfg->gen.plot_vis_rpy ||
        cfg->gen.plot_gt_with_rpy ||
        cfg->gen.plot_gt_with_vis_rpy;
}

static void plotLine2D(const char* label,
                       const std::vector<double>& x,
                       const std::vector<double>& y) {
    if (!x.empty() && x.size() == y.size()) {
        ImPlot::PlotLine(label, x.data(), y.data(), static_cast<int>(x.size()));
    }
}

static void setupTrajectoryAxes(bool plot_2d) {
    if (plot_2d) {
        ImPlot::SetupAxes("x [m]", "y [m]", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

    } else {
        // ImPlot no soporta 3D real. Usamos proyeccion X-Z.
        ImPlot::SetupAxes("x [m]", "z [m]", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
    }
}

static void plotTrajectoryProjected(const char* label,
                                    const std::vector<double>& x,
                                    const std::vector<double>& y,
                                    const std::vector<double>& z,
                                    bool plot_2d) {
    if (x.empty() || x.size() != y.size() || x.size() != z.size()) return;

    if (plot_2d) {
        ImPlot::PlotLine(label, x.data(), y.data(), static_cast<int>(x.size()), {
            ImPlotProp_LineWeight, TRAJECTORY_LINE_WEIGHT
        });
    } else {
        ImPlot::PlotLine(label, x.data(), z.data(), static_cast<int>(x.size()), {
            ImPlotProp_LineWeight, TRAJECTORY_LINE_WEIGHT
        });
    }
}

// -------------------------
// D3D helpers
// -------------------------
bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_0
    };

    const HRESULT res = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        createDeviceFlags,
        featureLevelArray,
        2,
        D3D11_SDK_VERSION,
        &sd,
        &g_pSwapChain,
        &g_pd3dDevice,
        &featureLevel,
        &g_pd3dDeviceContext
    );

    return SUCCEEDED(res);
}

void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer = nullptr;
    if (FAILED(g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer)))) return;
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget() {
    if (g_mainRenderTargetView) {
        g_mainRenderTargetView->Release();
        g_mainRenderTargetView = nullptr;
    }
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

// -------------------------
// Win32
// -------------------------

LRESULT WINAPI PlotWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) {
        return true;
    }

    switch (msg) {
    case WM_SIZE:
        if (g_pd3dDevice != nullptr && wParam != SIZE_MINIMIZED) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;

    case WM_CLOSE:
        g_plot_running.store(false, std::memory_order_relaxed);
        DestroyWindow(hWnd);
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }
}

// -------------------------
// ImGui / ImPlot rendering
// -------------------------
static void renderPlotsUi(const PlotSnapshot& s) {
    if (g_plot_cfg == nullptr) return;

    if (g_plot_cfg->gen.plot_tray) {
        ImGui::Begin("VIO tray");
        if (ImPlot::BeginPlot("VIO tray", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            setupTrajectoryAxes(g_plot_cfg->gen.plot_2d);
            plotTrajectoryProjected("vio", s.vio_x, s.vio_y, s.vio_z, g_plot_cfg->gen.plot_2d);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_vis_tray) {
        ImGui::Begin("VIS tray");
        if (ImPlot::BeginPlot("VIS tray", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            setupTrajectoryAxes(g_plot_cfg->gen.plot_2d);
            plotTrajectoryProjected("vis", s.vis_x, s.vis_y, s.vis_z, g_plot_cfg->gen.plot_2d);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_imu_tray) {
        ImGui::Begin("IMU tray");
        if (ImPlot::BeginPlot("IMU tray", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            setupTrajectoryAxes(g_plot_cfg->gen.plot_2d);
            plotTrajectoryProjected("imu", s.imu_x, s.imu_y, s.imu_z, g_plot_cfg->gen.plot_2d);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_gt_with_tray) {
        ImGui::Begin("GT + VIO");
        if (ImPlot::BeginPlot("GT + VIO", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            setupTrajectoryAxes(g_plot_cfg->gen.plot_2d);
            plotTrajectoryProjected("gt",  s.gt_x,  s.gt_y,  s.gt_z,  g_plot_cfg->gen.plot_2d);
            plotTrajectoryProjected("vio", s.vio_x, s.vio_y, s.vio_z, g_plot_cfg->gen.plot_2d);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_gt_with_vis_tray) {
        ImGui::Begin("GT + VIS");
        if (ImPlot::BeginPlot("GT + VIS", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            setupTrajectoryAxes(g_plot_cfg->gen.plot_2d);
            plotTrajectoryProjected("gt",  s.gt_x,  s.gt_y,  s.gt_z,  g_plot_cfg->gen.plot_2d);
            plotTrajectoryProjected("vis", s.vis_x, s.vis_y, s.vis_z, g_plot_cfg->gen.plot_2d);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_gt_with_imu_tray) {
        ImGui::Begin("GT + IMU");
        if (ImPlot::BeginPlot("GT + IMU", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            setupTrajectoryAxes(g_plot_cfg->gen.plot_2d);
            plotTrajectoryProjected("gt",  s.gt_x,  s.gt_y,  s.gt_z,  g_plot_cfg->gen.plot_2d);
            plotTrajectoryProjected("imu", s.imu_x, s.imu_y, s.imu_z, g_plot_cfg->gen.plot_2d);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_imu) {
        ImGui::Begin("Gyro");
        if (ImPlot::BeginPlot("Gyro", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            ImPlot::SetupAxes("t [s]", "rad/s", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            plotLine2D("gx raw", s.t, s.gx_raw);
            plotLine2D("gy raw", s.t, s.gy_raw);
            plotLine2D("gz raw", s.t, s.gz_raw);
            plotLine2D("gx cal", s.t, s.gx_cal);
            plotLine2D("gy cal", s.t, s.gy_cal);
            plotLine2D("gz cal", s.t, s.gz_cal);
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Accel");
        if (ImPlot::BeginPlot("Accel", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            ImPlot::SetupAxes("t [s]", "m/s^2", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            plotLine2D("ax raw", s.t, s.ax_raw);
            plotLine2D("ay raw", s.t, s.ay_raw);
            plotLine2D("az raw", s.t, s.az_raw);
            plotLine2D("ax cal", s.t, s.ax_cal);
            plotLine2D("ay cal", s.t, s.ay_cal);
            plotLine2D("az cal", s.t, s.az_cal);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_rpy) {
        ImGui::Begin("IMU RPY");
        if (ImPlot::BeginPlot("IMU RPY", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            ImPlot::SetupAxes("t [s]", "rad", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            plotLine2D("roll",  s.t, s.rpy_r);
            plotLine2D("pitch", s.t, s.rpy_p);
            plotLine2D("yaw",   s.t, s.rpy_yaw);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_vis_rpy) {
        ImGui::Begin("VIS RPY");
        if (ImPlot::BeginPlot("VIS RPY", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            ImPlot::SetupAxes("t [s]", "rad", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            plotLine2D("vis roll",  s.t, s.vis_rpy_r);
            plotLine2D("vis pitch", s.t, s.vis_rpy_p);
            plotLine2D("vis yaw",   s.t, s.vis_rpy_yaw);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_gt_with_rpy) {
        ImGui::Begin("GT + IMU RPY");
        if (ImPlot::BeginPlot("GT + IMU RPY", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            ImPlot::SetupAxes("t [s]", "rad", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            plotLine2D("imu roll",  s.t, s.rpy_r);
            plotLine2D("imu pitch", s.t, s.rpy_p);
            plotLine2D("imu yaw",   s.t, s.rpy_yaw);
            plotLine2D("gt roll",   s.t, s.gt_r);
            plotLine2D("gt pitch",  s.t, s.gt_p);
            plotLine2D("gt yaw",    s.t, s.gt_yaw);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    if (g_plot_cfg->gen.plot_gt_with_vis_rpy) {
        ImGui::Begin("GT + VIS RPY");
        if (ImPlot::BeginPlot("GT + VIS RPY", ImGui::GetContentRegionAvail(), PLOT_FLAGS_DEFAULT)) {
            ImPlot::SetupAxes("t [s]", "rad", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            plotLine2D("vis roll",  s.t, s.vis_rpy_r);
            plotLine2D("vis pitch", s.t, s.vis_rpy_p);
            plotLine2D("vis yaw",   s.t, s.vis_rpy_yaw);
            plotLine2D("gt roll",   s.t, s.gt_r);
            plotLine2D("gt pitch",  s.t, s.gt_p);
            plotLine2D("gt yaw",    s.t, s.gt_yaw);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    ImGui::Begin("Plotter status");
    ImGui::Text("Samples: %d", static_cast<int>(s.t.size()));
    ImGui::Text("Plot 3D mode: %s", g_plot_cfg->gen.plot_2d ? "2D XY" : "Projected XZ");
    ImGui::End();
}

static void plotThreadMain() {
    WNDCLASSEXW wc = {
        sizeof(WNDCLASSEXW), CS_CLASSDC, PlotWndProc, 0L, 0L,
        GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
        L"ImPlotStandaloneWindow", nullptr
    };

    RegisterClassExW(&wc);

    HWND hwnd = CreateWindowW(
        wc.lpszClassName,
        L"Plots - ImPlot",
        WS_OVERLAPPEDWINDOW,
        100, 100, 1400, 900,
        nullptr, nullptr, wc.hInstance, nullptr
    );

    if (hwnd == nullptr) {
        Logger(ERROR, "Plotter: failed to create window");
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        g_plot_ready.store(false, std::memory_order_relaxed);
        g_plot_running.store(false, std::memory_order_relaxed);
        return;
    }

    g_plot_hwnd = hwnd;

    if (!CreateDeviceD3D(hwnd)) {
        Logger(ERROR, "Plotter: failed to create D3D11 device");
        DestroyWindow(hwnd);
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        g_plot_hwnd = nullptr;
        g_plot_ready.store(false, std::memory_order_relaxed);
        g_plot_running.store(false, std::memory_order_relaxed);
        return;
    }

    CreateRenderTarget();
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();



    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    g_plot_ready.store(true, std::memory_order_relaxed);

    while (g_plot_running.load(std::memory_order_relaxed)) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) {
                g_plot_running.store(false, std::memory_order_relaxed);
            }
        }

        if (!g_plot_running.load(std::memory_order_relaxed)) {
            break;
        }

        PlotSnapshot snapshot = makeSnapshot();

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        renderPlotsUi(snapshot);

        ImGui::Render();
        const float clear_color[4] = {0.08f, 0.08f, 0.10f, 1.0f};
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        g_pSwapChain->Present(1, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    CleanupDeviceD3D();

    if (g_plot_hwnd != nullptr) {
        DestroyWindow(g_plot_hwnd);
        g_plot_hwnd = nullptr;
    }

    UnregisterClassW(wc.lpszClassName, wc.hInstance);

    g_plot_ready.store(false, std::memory_order_relaxed);
}

} // namespace

void initPlotters(Config* config) {
    g_plot_cfg = config;
    const bool enabled = wantAnyPlots(config);
    g_plot_on.store(enabled, std::memory_order_relaxed);

    if (!enabled) {
        return;
    }

    clearAllPlotBuffers();

    if (g_plot_running.load(std::memory_order_relaxed)) {
        return;
    }

    g_plot_running.store(true, std::memory_order_relaxed);
    g_plot_ready.store(false, std::memory_order_relaxed);
    g_plot_thread = std::thread(plotThreadMain);

    for (int i = 0; i < 100; ++i) {
        if (g_plot_ready.load(std::memory_order_relaxed)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void updatePlots(StateOut* state) {
    if (!g_plot_on.load(std::memory_order_relaxed) || state == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_plot_mutex);

    pushTrim(g_t, state->ts_ms * 1e-3);

    pushTrim(g_vio_x, state->pos_m.x());
    pushTrim(g_vio_y, state->pos_m.y());
    pushTrim(g_vio_z, state->pos_m.z());

    pushTrim(g_vis_x, state->deb.vis_xyz.x());
    pushTrim(g_vis_y, state->deb.vis_xyz.y());
    pushTrim(g_vis_z, state->deb.vis_xyz.z());

    pushTrim(g_imu_x, state->deb.imu_xyz.x());
    pushTrim(g_imu_y, state->deb.imu_xyz.y());
    pushTrim(g_imu_z, state->deb.imu_xyz.z());

    pushTrim(g_gt_x, state->posgt_m.x());
    pushTrim(g_gt_y, state->posgt_m.y());
    pushTrim(g_gt_z, state->posgt_m.z());

    pushTrim(g_gx_raw, state->deb.gyr_rads.x());
    pushTrim(g_gy_raw, state->deb.gyr_rads.y());
    pushTrim(g_gz_raw, state->deb.gyr_rads.z());
    pushTrim(g_ax_raw, state->deb.acc_ms2.x());
    pushTrim(g_ay_raw, state->deb.acc_ms2.y());
    pushTrim(g_az_raw, state->deb.acc_ms2.z());

    pushTrim(g_gx_cal, state->gyr_cal_rads.x());
    pushTrim(g_gy_cal, state->gyr_cal_rads.y());
    pushTrim(g_gz_cal, state->gyr_cal_rads.z());
    pushTrim(g_ax_cal, state->acc_cal_ms2.x());
    pushTrim(g_ay_cal, state->acc_cal_ms2.y());
    pushTrim(g_az_cal, state->acc_cal_ms2.z());

    pushTrim(g_rpy_r, state->deb.imu_rpy.x());
    pushTrim(g_rpy_p, state->deb.imu_rpy.y());
    pushTrim(g_rpy_yaw, state->deb.imu_rpy.z());

    pushTrim(g_vis_rpy_r, state->deb.vis_rpy.x());
    pushTrim(g_vis_rpy_p, state->deb.vis_rpy.y());
    pushTrim(g_vis_rpy_yaw, state->deb.vis_rpy.z());

    pushTrim(g_gt_r, state->origt_rad.x());
    pushTrim(g_gt_p, state->origt_rad.y());
    pushTrim(g_gt_yaw, state->origt_rad.z());
}

void closePlotters() {
    if (!g_plot_running.load(std::memory_order_relaxed) && !g_plot_thread.joinable()) {
        g_plot_on.store(false, std::memory_order_relaxed);
        return;
    }

    g_plot_running.store(false, std::memory_order_relaxed);

    if (g_plot_hwnd != nullptr) {
        PostMessage(g_plot_hwnd, WM_CLOSE, 0, 0);
    }

    if (g_plot_thread.joinable()) {
        g_plot_thread.join();
    }

    g_plot_on.store(false, std::memory_order_relaxed);
    g_plot_ready.store(false, std::memory_order_relaxed);
    g_plot_hwnd = nullptr;
}
