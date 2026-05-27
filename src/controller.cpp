#include "controller.hpp"
#include "lie_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace {

struct PID{
    double kp = 0.0;
    double ki = 0.0;
    double kd = 0.0;

    double b = 1.0;
    double c = 1.0;
    double u0 = 0.0;

    double Dt = 0.01;
    double Tf = 0.0;
    double Tt = 0.1;

    double minlim = -std::numeric_limits<double>::infinity();
    double maxlim = std::numeric_limits<double>::infinity();
    double dumin = -std::numeric_limits<double>::infinity();
    double dumax = std::numeric_limits<double>::infinity();

    double xu = 0.0;
    double xus = 0.0;
    double xr = 0.0;
    double xdr = 0.0;
    double xy = 0.0;
    double xdy = 0.0;
    double xuff = 0.0;
    double xf1 = 0.0;
    double xf2 = 0.0;
    bool is_initialized = false;

    void configure(const PIDAxisConfig& cfg){
        kp = cfg.kp;
        ki = cfg.ki;
        kd = cfg.kd;
        b = cfg.b;
        c = cfg.c;
        u0 = cfg.u0;
        Dt = cfg.Dt;
        Tf = cfg.Tf;
        Tt = std::max(cfg.Tt, 1e-6);
        minlim = cfg.minlim;
        maxlim = cfg.maxlim;
        dumin = cfg.dumin;
        dumax = cfg.dumax;
        is_initialized = false;
    }

    void setDt(double dt){
        if(std::isfinite(dt) && dt > 1e-6){
            Dt = dt;
        }
    }

    void init(double r0 = 0.0, double y0 = 0.0, double u0_in = 0.0, double uff0 = 0.0){
        xr = r0;
        xy = y0;
        xdr = 0.0;
        xdy = 0.0;
        xu = u0_in;
        xus = u0_in;
        xuff = uff0;
        xf1 = y0;
        xf2 = y0;
        is_initialized = true;
    }

    double filter(double y){
        if(!is_initialized){
            init(0.0, y, u0, 0.0);
        }

        if(Tf <= 1e-9){
            xf1 = y;
            xf2 = y;
            return y;
        }

        const double a = Dt / (Tf + 0.5*Dt);
        xf1 = xf1 + a*(y - xf1);
        xf2 = xf2 + a*(xf1 - xf2);
        return xf2;
    }

    double control(double r, double y, double uff = 0.0, double uman = 0.0, double utrack = 0.0, const std::string& mode = "AUTO"){
        if(!is_initialized){
            init(r, y, u0, uff);
        }

        const double umin = std::max(minlim, xus + Dt*dumin);
        const double umax = std::min(maxlim, xus + Dt*dumax);

        const double dr = (r - xr) / Dt;
        const double dy = (y - xy) / Dt;

        if(mode == "TRACK"){
            xu = utrack;
        }

        double u = 0.0;
        if(mode == "MAN"){
            u = uman;
        }else if(std::abs(ki) <= 1e-12){
            u = u0 + kp*(r - y) + kd*(c*dr - dy) + uff;
        }else{
            const double d_r = r - xr;
            const double d_y = y - xy;
            const double d_dr = dr - xdr;
            const double d_dy = dy - xdy;

            const double du_p = kp*(b*d_r - d_y);
            const double du_i = ki*Dt*(r - y);
            const double du_d = kd*(c*d_dr - d_dy);
            const double du_ff = uff - xuff;

            u = xu + du_p + du_i + du_d + du_ff;

            const double us = std::clamp(u, umin, umax);
            u = u - Dt/Tt*(u - us);
        }

        xu = u;
        u = std::clamp(u, umin, umax);
        xus = u;

        xr = r;
        xdr = dr;
        xy = y;
        xdy = dy;
        xuff = uff;

        return u;
    }
};

struct ControllerState{
    ControllerConfig config;
    PID vx;
    PID vy;
    PID vz;
    PID wz;
    bool initialized = false;
};

ControllerState gctrl;

static double safeDt(double dt, double fallback){
    if(std::isfinite(dt) && dt > 1e-6){
        return dt;
    }
    return std::max(fallback, 1e-3);
}

}

void controllerInit(const Config * config){
    gctrl = ControllerState{};

    if(config == nullptr){
        Logger(ERROR, "controllerInit: null config");
        return;
    }

    gctrl.config = config->ctrl;
    gctrl.vx.configure(gctrl.config.vx);
    gctrl.vy.configure(gctrl.config.vy);
    gctrl.vz.configure(gctrl.config.vz);
    gctrl.wz.configure(gctrl.config.wz);
    gctrl.initialized = true;

    Logger(INFO,"controllerInit: enabled=%d dt=%.4f gains_vx=[%.3f %.3f %.3f] gains_vy=[%.3f %.3f %.3f] gains_vz=[%.3f %.3f %.3f] gains_wz=[%.3f %.3f %.3f]",gctrl.config.enabled ? 1 : 0,gctrl.config.default_dt,gctrl.config.vx.kp, gctrl.config.vx.ki, gctrl.config.vx.kd,gctrl.config.vy.kp, gctrl.config.vy.ki, gctrl.config.vy.kd,gctrl.config.vz.kp, gctrl.config.vz.ki, gctrl.config.vz.kd,gctrl.config.wz.kp, gctrl.config.wz.ki, gctrl.config.wz.kd);
}

void controllerUpdate(StateOut& state, Command * cmd){
    if(cmd == nullptr){
        return;
    }

    state.deb.post_pid_lin_cmd = cmd->lenvel_ms;
    state.deb.post_pid_ang_cmd = cmd->angvel_rads;

    if(!gctrl.initialized || !gctrl.config.enabled){
        return;
    }

    const double dt = safeDt(state.dt, gctrl.config.default_dt);
    gctrl.vx.setDt(dt);
    gctrl.vy.setDt(dt);
    gctrl.vz.setDt(dt);
    gctrl.wz.setDt(dt);

    const vec3 v_meas = state.state.vel;
    vec3 w_meas_cam = state.state.dpose.vgyr;
    if (state.Localx.size() >= 17) {
        const mat3 Rci = QuatToRot(state.Localx.segment(10,4));
        w_meas_cam = Rci * w_meas_cam;
    }

    const vec3 v_ref = cmd->lenvel_ms;
    const vec3 w_ref = cmd->angvel_rads;

    const double vx_f = gctrl.vx.filter(v_meas.x());
    const double vy_f = gctrl.vy.filter(v_meas.y());
    const double vz_f = gctrl.vz.filter(v_meas.z());
    // Camera yaw is rotation about -y in the visual frame.
    const double yaw_rate_f = gctrl.wz.filter(-w_meas_cam.y());

    cmd->ts_ms = state.ts_ms;
    cmd->lenvel_ms.x() = gctrl.vx.control(v_ref.x(), vx_f, 0.0, 0.0, v_ref.x(), "AUTO");
    cmd->lenvel_ms.y() = gctrl.vy.control(v_ref.y(), vy_f, 0.0, 0.0, v_ref.y(), "AUTO");
    cmd->lenvel_ms.z() = gctrl.vz.control(v_ref.z(), vz_f, 0.0, 0.0, v_ref.z(), "AUTO");
    cmd->angvel_rads.x() = 0.0;
    cmd->angvel_rads.y() = 0.0;
    cmd->angvel_rads.z() = gctrl.wz.control(w_ref.z(), yaw_rate_f, 0.0, 0.0, w_ref.z(), "AUTO");

    state.deb.post_pid_lin_cmd = cmd->lenvel_ms;
    state.deb.post_pid_ang_cmd = cmd->angvel_rads;
}
