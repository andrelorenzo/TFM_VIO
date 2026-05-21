#include "plotter.hpp"

#include "vio_plots.hpp"
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
#include <memory>
#include <thread>

#pragma comment(lib, "d3d11.lib")
extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

std::atomic<bool> g_plot_on{false};
std::atomic<bool> g_plot_running{false};
std::atomic<bool> g_plot_ready{false};

std::thread g_plot_thread;
HWND g_plot_hwnd = nullptr;

ID3D11Device* g_pd3dDevice = nullptr;
ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
IDXGISwapChain* g_pSwapChain = nullptr;
ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

std::unique_ptr<vioplot::VioPlotter> g_vio_plotter;

bool wantAnyPlots(const Config* cfg) {
    if (cfg == nullptr) {
        return false;
    }

    return cfg->gen.plot_imu ||
           cfg->gen.plot_tray ||
           cfg->gen.plot_vis_tray ||
           cfg->gen.plot_imu_tray ||
           cfg->gen.plot_height ||
           cfg->gen.plot_rpy ||
           cfg->gen.plot_vis_rpy ||
           cfg->gen.plot_imu_rpy ||
           cfg->gen.plot_dpos ||
           cfg->gen.plot_dvel ||
           cfg->gen.plot_da3;
}

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
    if (g_pSwapChain == nullptr) {
        return;
    }
    if (FAILED(g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer)))) {
        return;
    }
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget() {
    if (g_mainRenderTargetView != nullptr) {
        g_mainRenderTargetView->Release();
        g_mainRenderTargetView = nullptr;
    }
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain != nullptr) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext != nullptr) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice != nullptr) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

LRESULT WINAPI PlotWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) {
        return true;
    }

    switch (msg) {
    case WM_SIZE:
        if (g_pd3dDevice != nullptr && wParam != SIZE_MINIMIZED) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, static_cast<UINT>(LOWORD(lParam)), static_cast<UINT>(HIWORD(lParam)), DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;

    case WM_CLOSE:
        g_plot_running.store(false, std::memory_order_relaxed);
        DestroyWindow(hWnd);
        return 0;

    case WM_DESTROY:
        if (g_plot_hwnd == hWnd) {
            g_plot_hwnd = nullptr;
        }
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }
}

void renderPlotsUi() {
    if (g_vio_plotter != nullptr) {
        g_vio_plotter->render();
    }
}

void plotThreadMain() {
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

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        renderPlotsUi();

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

    if (g_plot_hwnd != nullptr && IsWindow(g_plot_hwnd)) {
        DestroyWindow(g_plot_hwnd);
    }
    g_plot_hwnd = nullptr;

    UnregisterClassW(wc.lpszClassName, wc.hInstance);

    g_plot_ready.store(false, std::memory_order_relaxed);
}

} // namespace

void initPlotters(Config* config) {
    const bool enabled = wantAnyPlots(config);
    g_plot_on.store(enabled, std::memory_order_relaxed);

    if (!enabled || config == nullptr) {
        return;
    }

    g_vio_plotter = std::make_unique<vioplot::VioPlotter>();
    vioplot::VioPlotterOptions options;
    options.max_points = 20000;
    options.trajectory_plane = vioplot::TrajectoryPlane::XZ;
    options.equal_trajectory_axes = true;
    g_vio_plotter->configure(*config, options);

    if (!g_vio_plotter->enabled()) {
        g_plot_on.store(false, std::memory_order_relaxed);
        return;
    }

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
    if (!g_plot_on.load(std::memory_order_relaxed) || state == nullptr || g_vio_plotter == nullptr) {
        return;
    }
    g_vio_plotter->update(*state);
}

void closePlotters() {
    if (!g_plot_running.load(std::memory_order_relaxed) && !g_plot_thread.joinable()) {
        g_plot_on.store(false, std::memory_order_relaxed);
        g_vio_plotter.reset();
        return;
    }

    g_plot_running.store(false, std::memory_order_relaxed);

    if (g_plot_hwnd != nullptr && IsWindow(g_plot_hwnd)) {
        PostMessage(g_plot_hwnd, WM_CLOSE, 0, 0);
    }

    if (g_plot_thread.joinable()) {
        g_plot_thread.join();
    }

    g_plot_on.store(false, std::memory_order_relaxed);
    g_plot_ready.store(false, std::memory_order_relaxed);
    g_plot_hwnd = nullptr;
    g_vio_plotter.reset();
}
