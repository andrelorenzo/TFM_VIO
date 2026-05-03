#include "vio_plots.hpp"
#include "lie_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace vioplot {
namespace {

bool wantAnyPlots(const Config& cfg) {
    return cfg.gen.plot_imu || cfg.gen.plot_tray || cfg.gen.plot_vis_tray || cfg.gen.plot_imu_tray || cfg.gen.plot_height || cfg.gen.plot_rpy || cfg.gen.plot_vis_rpy || cfg.gen.plot_imu_rpy || cfg.gen.plot_dpos || cfg.gen.plot_dvel;
}

} // namespace

void VioPlotter::configure(const Config& config, VioPlotterOptions options) {
    reset();

    options_ = options;
    enabled_ = wantAnyPlots(config);
    if (!enabled_) {
        return;
    }

    if (config.gen.plot_tray) {
        auto plot = addTrajectoryPlot("vio_traj", "VIO trajectory");
        bindTrajectory(plot, "vio", [](const StateOut& s) { return s.state.pose.pos; });
        if (config.gen.depth_on)
            bindTrajectory(plot, "gt", [](const StateOut& s) { return s.gtpose.pos; });
    }

    if (config.gen.plot_vis_tray) {
        auto plot = addTrajectoryPlot("vis_traj", "Visual-only trajectory");
        bindTrajectory(plot, "visual", [](const StateOut& s) { return s.deb.vis.pos; });
        if (config.gen.depth_on)
            bindTrajectory(plot, "gt", [](const StateOut& s) { return s.gtpose.pos; });
    }

    if (config.gen.plot_imu_tray) {
        auto plot = addTrajectoryPlot("imu_traj", "Inertial-only trajectory");
        bindTrajectory(plot, "imu", [](const StateOut& s) { return s.deb.imu.pos; });
        if (config.gen.depth_on) 
            bindTrajectory(plot, "gt", [](const StateOut& s) { return s.gtpose.pos; });
    }

    if (config.gen.plot_height) {
        auto plot = addTimePlot("height", "Height", "height y [m]");
        bind(plot, "vio y", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), s.state.pose.pos.y()}; });
        bind(plot, "visual y", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), s.deb.vis.pos.y()}; });
        bind(plot, "imu y", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), s.deb.imu.pos.y()}; });
        if (config.gen.depth_on) {
            bind(plot, "gt y", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), s.gtpose.pos.y()}; });
        }
    }

    if (config.gen.plot_imu) {
        auto gyro = addTimePlot("imu_gyro", "Gyroscope", "rad/s");
        bindVec3Time(gyro, "raw gyro", [](const StateOut& s) { return s.deb.rawimu.vgyr; });
        bindVec3Time(gyro, "corrected gyro", [](const StateOut& s) { return s.deb.corimu.vgyr; });

        auto accel = addTimePlot("imu_accel", "Accelerometer", "m/s^2");
        bindVec3Time(accel, "raw accel", [](const StateOut& s) { return s.deb.rawimu.vacc; });
        bindVec3Time(accel, "corrected accel", [](const StateOut& s) { return s.deb.corimu.vacc; });
    }

    if (config.gen.plot_rpy) {
        auto plot = addTimePlot("vio_rpy", "VIO RPY", "rad");
        bindRpyTime(plot, "vio", [](const StateOut& s) { return poseRpy(s.state.pose); });
        if (config.gen.depth_on) {
            bindRpyTime(plot, "gt", [](const StateOut& s) { return poseRpy(s.gtpose); });
        }
    }

    if (config.gen.plot_vis_rpy) {
        auto plot = addTimePlot("vis_rpy", "Visual-only RPY", "rad");
        bindRpyTime(plot, "visual", [](const StateOut& s) { return poseRpy(s.deb.vis); });
        if (config.gen.depth_on) {
            bindRpyTime(plot, "gt", [](const StateOut& s) { return poseRpy(s.gtpose); });
        }
    }

    if (config.gen.plot_imu_rpy) {
        auto plot = addTimePlot("imu_rpy", "Inertial-only RPY", "rad");
        bindRpyTime(plot, "imu", [](const StateOut& s) { return poseRpy(s.deb.imu); });
        if (config.gen.depth_on) {
            bindRpyTime(plot, "gt", [](const StateOut& s) { return poseRpy(s.gtpose); });
        }
    }

    if (config.gen.plot_dpos) {
        auto plot = addTimePlot("preint_dpos", "Preintegrated dPos", "m");
        bind(plot, "dp x", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), preimuAt(s, 0)}; });
        bind(plot, "dp y", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), preimuAt(s, 1)}; });
        bind(plot, "dp z", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), preimuAt(s, 2)}; });
    }

    if (config.gen.plot_dvel) {
        auto dvel = addTimePlot("preint_dvel", "Preintegrated dVel", "m/s");
        bind(dvel, "dv x", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), preimuAt(s, 3)}; });
        bind(dvel, "dv y", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), preimuAt(s, 4)}; });
        bind(dvel, "dv z", [](const StateOut& s) { return plotlib::Point{timeSeconds(s), preimuAt(s, 5)}; });

        auto vel = addTimePlot("state_velocity", "State velocity", "m/s");
        bindVec3Time(vel, "vel", [](const StateOut& s) { return s.state.vel; });
    }
}

void VioPlotter::update(const StateOut& state) {
    if (!enabled_) {
        return;
    }

    for (const auto& binding : bindings_) {
        plots_.push(binding.series, binding.sample(state));
    }
}

void VioPlotter::render() {
    if (!enabled_) {
        return;
    }
    renderDashboard();
}

void VioPlotter::clearSamples() {
    plots_.clearSamples();
}

void VioPlotter::reset() {
    bindings_.clear();
    plots_.reset();
    enabled_ = false;
}

plotlib::PlotHandle VioPlotter::addTimePlot(const char* id, const char* title, const char* y_label) {
    plotlib::PlotConfig cfg;
    cfg.id = id;
    cfg.window_title = title;
    cfg.title = title;
    cfg.x_label = "t [s]";
    cfg.y_label = y_label;
    cfg.auto_fit_x = true;
    cfg.auto_fit_y = true;
    cfg.equal = false;
    cfg.max_points = options_.max_points;
    return plots_.createPlot(std::move(cfg));
}

plotlib::PlotHandle VioPlotter::addTrajectoryPlot(const char* id, const char* title) {
    plotlib::PlotConfig cfg;
    cfg.id = id;
    cfg.window_title = title;
    cfg.title = title;
    cfg.x_label = trajectoryXLabel();
    cfg.y_label = trajectoryYLabel();
    cfg.auto_fit_x = true;
    cfg.auto_fit_y = true;
    cfg.equal = options_.equal_trajectory_axes;
    cfg.max_points = options_.max_points;
    return plots_.createPlot(std::move(cfg));
}

void VioPlotter::bind(const plotlib::PlotHandle& plot,
                      const char* label,
                      std::function<plotlib::Point(const StateOut&)> sample,
                      float line_weight) {
    plotlib::SeriesConfig series;
    series.label = label;
    series.line_weight = line_weight;

    auto handle = plots_.addSeries(plot, std::move(series));
    bindings_.push_back(Binding{handle, std::move(sample)});
}

void VioPlotter::bindVec3Time(const plotlib::PlotHandle& plot,
                              const char* prefix,
                              std::function<vec3(const StateOut&)> value) {
    const std::string p(prefix);

    bind(plot, (p + " x").c_str(), [value](const StateOut& s) {
        const auto v = value(s);
        return plotlib::Point{timeSeconds(s), v.x()};
    });
    bind(plot, (p + " y").c_str(), [value](const StateOut& s) {
        const auto v = value(s);
        return plotlib::Point{timeSeconds(s), v.y()};
    });
    bind(plot, (p + " z").c_str(), [value](const StateOut& s) {
        const auto v = value(s);
        return plotlib::Point{timeSeconds(s), v.z()};
    });
}

void VioPlotter::bindRpyTime(const plotlib::PlotHandle& plot,
                             const char* prefix,
                             std::function<vec3(const StateOut&)> value) {
    const std::string p(prefix);

    bind(plot, (p + " roll").c_str(), [value](const StateOut& s) {
        const auto rpy = value(s);
        return plotlib::Point{timeSeconds(s), rpy.x()};
    });
    bind(plot, (p + " pitch").c_str(), [value](const StateOut& s) {
        const auto rpy = value(s);
        return plotlib::Point{timeSeconds(s), rpy.y()};
    });
    bind(plot, (p + " yaw").c_str(), [value](const StateOut& s) {
        const auto rpy = value(s);
        return plotlib::Point{timeSeconds(s), rpy.z()};
    });
}

void VioPlotter::bindTrajectory(const plotlib::PlotHandle& plot,
                                const char* label,
                                std::function<vec3(const StateOut&)> position) {
    bind(plot, label, [this, position](const StateOut& s) {
        return projectTrajectory(position(s));
    }, 2.0f);
}

double VioPlotter::timeSeconds(const StateOut& state) {
    return state.ts_ms * 1e-3;
}

vec3 VioPlotter::poseRpy(Pose pose) {
    return quatToCameraRpyRad(pose.rot);
}

double VioPlotter::preimuAt(const StateOut& state, int index) {
    if (index < 0 || index >= state.deb.preimu.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return state.deb.preimu(index);
}

plotlib::Point VioPlotter::projectTrajectory(const vec3& p) const {
    switch (options_.trajectory_plane) {
    case TrajectoryPlane::XY:
        return {p.x(), p.y()};
    case TrajectoryPlane::XZ:
        return {p.x(), p.z()};
    case TrajectoryPlane::YZ:
        return {p.y(), p.z()};
    default:
        return {p.x(), p.z()};
    }
}

const char* VioPlotter::trajectoryXLabel() const {
    switch (options_.trajectory_plane) {
    case TrajectoryPlane::XY:
    case TrajectoryPlane::XZ:
        return "x [m]";
    case TrajectoryPlane::YZ:
        return "y [m]";
    default:
        return "x [m]";
    }
}

const char* VioPlotter::trajectoryYLabel() const {
    switch (options_.trajectory_plane) {
    case TrajectoryPlane::XY:
        return "y [m]";
    case TrajectoryPlane::XZ:
    case TrajectoryPlane::YZ:
        return "z [m]";
    default:
        return "z [m]";
    }
}

void VioPlotter::renderDashboard() {
    const auto plots = plots_.snapshot();

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(viewport->WorkSize, ImGuiCond_Always);
    ImGui::Begin("Plots Dashboard", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);

    if (plots.empty()) {
        ImGui::TextUnformatted("No hay plots activos.");
        ImGui::End();
        return;
    }

    if (focused_plot_ != plotlib::npos && focused_plot_ >= plots.size()) {
        focused_plot_ = plotlib::npos;
    }

    const int max_columns = static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(4, plots.size())));
    dashboard_columns_ = std::clamp(dashboard_columns_, 1, max_columns);

    ImGui::Text("Plots: %d", static_cast<int>(plots.size()));
    ImGui::SameLine();
    ImGui::Text("Samples: %d", static_cast<int>(plots_.sampleCount()));

    if (focused_plot_ == plotlib::npos) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160.0f);
        ImGui::SliderInt("Columns", &dashboard_columns_, 1, max_columns);
    } else {
        ImGui::SameLine();
        if (ImGui::Button("Back to grid")) {
            focused_plot_ = plotlib::npos;
        }
    }

    ImGui::Separator();

    if (focused_plot_ != plotlib::npos) {
        const auto& plot = plots[focused_plot_];
        ImGui::TextUnformatted(plot.config.title.c_str());
        ImGui::Separator();
        plotlib::PlotRegistry::renderPlotWidget(plot, ImGui::GetContentRegionAvail());
        ImGui::End();
        return;
    }

    const int columns = std::max(1, dashboard_columns_);
    const int rows = static_cast<int>((plots.size() + static_cast<std::size_t>(columns) - 1) / static_cast<std::size_t>(columns));
    const float spacing_y = ImGui::GetStyle().ItemSpacing.y;
    const float total_height = ImGui::GetContentRegionAvail().y;
    const float cell_height = rows > 0
        ? std::max(220.0f, (total_height - spacing_y * static_cast<float>(rows - 1)) / static_cast<float>(rows))
        : total_height;

    if (ImGui::BeginTable("plot_dashboard_grid", columns, ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_BordersInnerV)) {
        for (std::size_t i = 0; i < plots.size(); ++i) {
            if ((i % static_cast<std::size_t>(columns)) == 0) {
                ImGui::TableNextRow();
            }
            ImGui::TableSetColumnIndex(static_cast<int>(i % static_cast<std::size_t>(columns)));

            const auto& plot = plots[i];
            const std::string child_id = "plot_cell_" + plot.config.id;
            ImGui::BeginChild(child_id.c_str(), ImVec2(0.0f, cell_height), true);

            ImGui::TextUnformatted(plot.config.title.c_str());
            ImGui::SameLine();
            const std::string button_id = "Max##" + plot.config.id;
            if (ImGui::Button(button_id.c_str())) {
                focused_plot_ = i;
            }
            ImGui::Separator();

            plotlib::PlotRegistry::renderPlotWidget(plot, ImGui::GetContentRegionAvail());
            ImGui::EndChild();
        }
        ImGui::EndTable();
    }

    ImGui::End();
}

} // namespace vioplot
