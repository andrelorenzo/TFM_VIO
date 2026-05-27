#include "local_planner.hpp"
#include "config.hpp"
#include "lie_math.hpp"
#include <cmath>
#include <algorithm>
#include <cfloat>

struct PPConfig{
    float la_m;
    size_t searc_win;
    size_t last_idx;
    bool finished;
    float v_max_xy;
    float v_max_z;
    float w_max_z;
    float k_xy;
    float k_z;
    float k_yaw;
    float goal_tol_m;
    float evade_pp_blend;
};

PPConfig ppconfig;

static float clampf(float x, float min_val, float max_val){
    return std::max(min_val, std::min(x, max_val));
}

static float wrapPi(float angle){
    while(angle > M_PI){ angle -= 2.0f*M_PI; }
    while(angle < -M_PI){ angle += 2.0f*M_PI; }
    return angle;
}

static float vec3Dist(vec3 a, vec3 b){
    float dx = a.x() - b.x();
    float dy = a.y() - b.y();
    float dz = a.z() - b.z();
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

static vec3 vec3Interp(vec3 a, vec3 b, float alpha){
    float x = (1.0f - alpha)*a.x() + alpha*b.x();
    float y = (1.0f - alpha)*a.y() + alpha*b.y();
    float z = (1.0f - alpha)*a.z() + alpha*b.z();
    return vec3(x, y, z);
}

static float yawFromQuat(quat q){
    return static_cast<float>(quatToCameraRpyRad(q).z());
}

static vec3 clampVelocity(vec3 vel){
    // VIO/world frame uses camera convention: x right, y down, z forward.
    // The planar motion for navigation lives on the x-z plane.
    const float xz_norm = static_cast<float>(std::hypot(vel.x(), vel.z()));
    if(xz_norm > ppconfig.v_max_xy && xz_norm > 1e-6f){
        const float scale = ppconfig.v_max_xy / xz_norm;
        vel.x() *= scale;
        vel.z() *= scale;
    }

    // The legacy v_max_z parameter now limits the vertical camera axis (y-down).
    vel.y() = clampf(static_cast<float>(vel.y()), -ppconfig.v_max_z, ppconfig.v_max_z);
    return vel;
}

static bool buildEvadeVelocity(const EvitationDir& dir, float yaw, vec3 * evade_vel){
    if(evade_vel == nullptr){
        return false;
    }

    // dir.obstacle_score;
    // dir.must_evade;
    // dir.norm_vec;
    
    const vec3 raw_dir = dir.norm_vec;
    const float lateral = static_cast<float>(raw_dir.x());
    const float vertical = static_cast<float>(-raw_dir.y());
    const float planar_norm = std::sqrt(lateral*lateral + vertical*vertical);
    if(planar_norm <= 1e-6f){
        return false;
    }

    const vec3 right_w(std::cos(yaw), 0.0, std::sin(yaw));
    const vec3 up_w(0.0, -1.0, 0.0);

    vec3 evade_dir = lateral*right_w + vertical*up_w;
    const double evade_dir_norm = evade_dir.norm();
    if(evade_dir_norm <= 1e-6){
        return false;
    }

    evade_dir /= evade_dir_norm;

    const float evade_gain = clampf(std::max(0.5f, dir.magnitude), 0.0f, 1.0f);
    const float evade_speed = evade_gain*ppconfig.v_max_xy;
    *evade_vel = clampVelocity(evade_speed*evade_dir);
    return true;
}

void localPlannerInit(Config * config){
    ppconfig.la_m       = config->ppr3d.la_m;
    ppconfig.searc_win  = config->ppr3d.searc_win;
    ppconfig.last_idx   = 0;
    ppconfig.finished   = false;
    ppconfig.v_max_xy   = config->ppr3d.v_max_xy;
    ppconfig.v_max_z    = config->ppr3d.v_max_z;
    ppconfig.w_max_z    = config->ppr3d.w_max_z;
    ppconfig.k_xy       = config->ppr3d.k_xy;
    ppconfig.k_z        = config->ppr3d.k_z;
    ppconfig.k_yaw      = config->ppr3d.k_yaw;
    ppconfig.goal_tol_m = config->ppr3d.goal_tol_m;
    ppconfig.evade_pp_blend = 0.1f;
}

static bool getPurePursuitTarget3D(vec3 pos, Waypoints path, vec3 *p_ref, bool *target_is_end){
    size_t n = path.p.size();

    if(n == 0){ return false; }

    if(n == 1){
        *p_ref = path.p[0];
        *target_is_end = true;
        ppconfig.last_idx = 0;
        return true;
    }

    size_t idx0 = 0;

    if(ppconfig.last_idx > 10){
        idx0 = ppconfig.last_idx - 10;
    }

    size_t idx1 = std::min(ppconfig.last_idx + ppconfig.searc_win, n - 1);

    float min_dist = FLT_MAX;
    size_t idx_closest = ppconfig.last_idx;

    for(size_t i = idx0; i <= idx1; i++){
        float d = vec3Dist(path.p[i], pos);

        if(d < min_dist){
            min_dist = d;
            idx_closest = i;
        }
    }

    ppconfig.last_idx = idx_closest;

    float remaining_la = ppconfig.la_m;

    for(size_t i = idx_closest; i < n - 1; i++){
        float seg_len = vec3Dist(path.p[i], path.p[i + 1]);

        if(seg_len < 1e-6f){
            continue;
        }

        if(remaining_la <= seg_len){
            float alpha = remaining_la / seg_len;
            *p_ref = vec3Interp(path.p[i], path.p[i + 1], alpha);
            *target_is_end = false;
            return true;
        }

        remaining_la -= seg_len;
    }

    *p_ref = path.p[n - 1];
    *target_is_end = true;
    return true;
}
void localPlannerUpdate(const EvitationDir& dir, StateOut& state, const Waypoints& path, Command * cmd){
    vec3 pos = state.state.pose.pos;
    quat ori = state.state.pose.rot;
    double dt = state.dt;
    (void)dt;

    cmd->lenvel_ms = vec3(0.0f, 0.0f, 0.0f);
    cmd->angvel_rads = vec3(0.0f, 0.0f, 0.0f);
    state.deb.lplan_target_valid = false;
    state.deb.pre_pid_lin_cmd = vec3::Zero();
    state.deb.pre_pid_ang_cmd = vec3::Zero();

    if(ppconfig.finished){
        return;
    }

    if(path.p.size() == 0){
        ppconfig.finished = true;
        return;
    }

    vec3 p_ref;
    bool target_is_end = false;

    bool valid_target = getPurePursuitTarget3D(pos, path, &p_ref, &target_is_end);

    if(!valid_target){
        ppconfig.finished = true;
        return;
    }

    state.deb.lplan_target_pos = p_ref;
    state.deb.lplan_target_valid = true;

    float ex = p_ref.x() - pos.x();
    float ey = p_ref.y() - pos.y();
    float ez = p_ref.z() - pos.z();

    const float dist_xz = sqrtf(ex*ex + ez*ez);
    float dist_goal = vec3Dist(pos, path.p[path.p.size() - 1]);

    if(target_is_end && dist_goal < ppconfig.goal_tol_m){
        ppconfig.finished = true;
        cmd->lenvel_ms = vec3(0.0f, 0.0f, 0.0f);
        cmd->angvel_rads = vec3(0.0f, 0.0f, 0.0f);
        return;
    }

    float vx = 0.0f;
    float vz = 0.0f;

    if(dist_xz > 1e-6f){
        float v_xy = ppconfig.k_xy*dist_xz;
        v_xy = clampf(v_xy, 0.0f, ppconfig.v_max_xy);

        vx = v_xy*ex/dist_xz;
        vz = v_xy*ez/dist_xz;
    }

    float vy = ppconfig.k_z*ey;
    vy = clampf(vy, -ppconfig.v_max_z, ppconfig.v_max_z);

    float yaw = yawFromQuat(ori);
    float wz = 0.0f;

    if(dist_xz > 1e-6f){
        float yaw_ref = atan2f(ex, ez);
        float yaw_error = wrapPi(yaw_ref - yaw);

        wz = ppconfig.k_yaw*yaw_error;
        wz = clampf(wz, -ppconfig.w_max_z, ppconfig.w_max_z);
    }

    vec3 pp_vel(vx, vy, vz);
    vec3 cmd_vel = pp_vel;
    float cmd_wz = wz;

    vec3 evade_vel = vec3::Zero();
    if(dir.must_evade && buildEvadeVelocity(dir, yaw, &evade_vel)){
        cmd_vel = clampVelocity(evade_vel + ppconfig.evade_pp_blend*pp_vel);
        cmd_wz = ppconfig.evade_pp_blend*wz;
    }

    cmd->lenvel_ms = cmd_vel;
    cmd->angvel_rads = vec3(0.0f, 0.0f, cmd_wz);
    state.deb.pre_pid_lin_cmd = cmd_vel;
    state.deb.pre_pid_ang_cmd = cmd->angvel_rads;
}
