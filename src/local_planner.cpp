#include "local_planner.hpp"
#include "config.hpp"
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
    float siny_cosp = 2.0f*(q.w()*q.z() + q.x()*q.y());
    float cosy_cosp = 1.0f - 2.0f*(q.y()*q.y() + q.z()*q.z());
    return atan2f(siny_cosp, cosy_cosp);
}

void localPlannerInit(Config * config){
    (void)config; // Evita warning de compilación porque el parámetro config todavía no se usa.

    ppconfig.la_m = 0.75f;        // Distancia lookahead en metros. El dron no persigue el punto más cercano, sino un punto situado 0.75 m por delante en la trayectoria.
    ppconfig.searc_win = 50;      // Ventana de búsqueda. Número máximo de puntos del path que se analizan hacia delante desde last_idx.
    ppconfig.last_idx = 0;        // Último índice conocido del path donde estaba el dron. Se usa para no buscar desde el principio en cada iteración.
    ppconfig.finished = false;    // Indica si el planificador local ya ha terminado de seguir el path actual.

    ppconfig.v_max_xy = 0.75f;    // Velocidad lineal máxima en el plano XY, en m/s. Limita sqrt(vx^2 + vy^2).
    ppconfig.v_max_z = 0.45f;     // Velocidad lineal máxima vertical, en m/s. Limita la velocidad de subida y bajada.
    ppconfig.w_max_z = 1.0f;      // Velocidad angular máxima en yaw, en rad/s. Limita la rotación alrededor del eje Z.

    ppconfig.k_xy = 1.0f;         // Ganancia proporcional para convertir el error horizontal en velocidad XY.
    ppconfig.k_z = 1.0f;          // Ganancia proporcional para convertir el error vertical en velocidad Z.
    ppconfig.k_yaw = 2.0f;        // Ganancia proporcional para convertir el error de yaw en velocidad angular wz.

    ppconfig.goal_tol_m = 0.12f;  // Tolerancia de llegada al objetivo final, en metros. Si el dron está a menos de 0.12 m del final, se considera terminado.
}
float vec3Dist(vec3 a, vec3 b){ return sqrt( powf(a.x()-b.x(), 2) + powf(a.y()-b.y(), 2) + powf(a.z()-b.z(), 2));}

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
void localPlannerUpdate(StateOut state, Waypoints path, Command * cmd){
    vec3 pos = state.state.pose.pos;
    quat ori = state.state.pose.rot;
    double dt = state.dt;
    (void)dt;

    cmd->lenvel_ms = vec3(0.0f, 0.0f, 0.0f);
    cmd->angvel_rads = vec3(0.0f, 0.0f, 0.0f);

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

    float ex = p_ref.x() - pos.x();
    float ey = p_ref.y() - pos.y();
    float ez = p_ref.z() - pos.z();

    float dist_xy = sqrtf(ex*ex + ey*ey);
    float dist_goal = vec3Dist(pos, path.p[path.p.size() - 1]);

    if(target_is_end && dist_goal < ppconfig.goal_tol_m){
        ppconfig.finished = true;
        cmd->lenvel_ms = vec3(0.0f, 0.0f, 0.0f);
        cmd->angvel_rads = vec3(0.0f, 0.0f, 0.0f);
        return;
    }

    float vx = 0.0f;
    float vy = 0.0f;

    if(dist_xy > 1e-6f){
        float v_xy = ppconfig.k_xy*dist_xy;
        v_xy = clampf(v_xy, 0.0f, ppconfig.v_max_xy);

        vx = v_xy*ex/dist_xy;
        vy = v_xy*ey/dist_xy;
    }

    float vz = ppconfig.k_z*ez;
    vz = clampf(vz, -ppconfig.v_max_z, ppconfig.v_max_z);

    float yaw = yawFromQuat(ori);
    float wz = 0.0f;

    if(dist_xy > 1e-6f){
        float yaw_ref = atan2f(ey, ex);
        float yaw_error = wrapPi(yaw_ref - yaw);

        wz = ppconfig.k_yaw*yaw_error;
        wz = clampf(wz, -ppconfig.w_max_z, ppconfig.w_max_z);
    }

    cmd->lenvel_ms = vec3(vx, vy, vz);
    cmd->angvel_rads = vec3(0.0f, 0.0f, wz);
}