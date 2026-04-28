#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include "seconds/logger.h"

inline constexpr const char * DEBUGSTATE_HEADER[] = {
    "vis_xyz_x","vis_xyz_y","vis_xyz_z","vis_rpy_r","vis_rpy_p","vis_rpy_y","scale",
    "vio_valid","vio_rel_valid","vio_trans_valid","vio_pnp_valid","vio_refined","vio_local_ba","vio_clone_factor","vio_fused",
    "vio_matches","vio_tracked","vio_inliers","vio_pnp_corr",
    "vio_kf_age","vio_window_kfs","vio_ref_age","vio_kf_parallax_deg","vio_init_tracks","vio_pose_only_tracks","vio_explo_tracks",
    "vio_triangulated","vio_pnp_inliers","vio_refine_inliers","vio_ba_landmarks","vio_ba_obs","vio_ba_inliers","vio_clone_features","vio_clone_obs","vio_clone_inliers","vio_landmarks","vio_reproj_px","vio_refine_reproj_px","vio_ba_reproj_px","vio_clone_reproj_px","vio_fuse_pos_res_m","vio_fuse_vel_res_ms","vio_fuse_ori_res_deg",
    "imu_rpy_r","imu_rpy_p","imu_rpy_y","imu_xyz_x","imu_xyz_y","imu_xyz_z","imu_stationary",
    "imu_dp_x","imu_dp_y","imu_dp_z","imu_dv_x","imu_dv_y","imu_dv_z",
    "imu_drpy_r","imu_drpy_p","imu_drpy_y","imu_dq_w","imu_dq_x","imu_dq_y","imu_dq_z",
    "acc_ms2_x","acc_ms2_y","acc_ms2_z","gyr_rads_x","gyr_rads_y","gyr_rads_z"
};
static constexpr size_t DEBUGSTATE_HEADER_SIZE = sizeof(DEBUGSTATE_HEADER) / sizeof(DEBUGSTATE_HEADER[0]);

inline constexpr const char * STATEOUT_HEADER[] = {
    "ts_ms","dt","pos_m_x","pos_m_y","pos_m_z","vel_ms_x","vel_ms_y","vel_ms_z","acc_cal_ms2_x","acc_cal_ms2_y","acc_cal_ms2_z","rpy_rad_r","rpy_rad_p",
    "rpy_rad_y","quat_rad_w","quat_rad_x","quat_rad_y","quat_rad_z","gyr_cal_rads_x",
    "gyr_cal_rads_y","gyr_cal_rads_z","posgt_m_x","posgt_m_y","posgt_m_z","origt_rad_x","origt_rad_y","origt_rad_z"
};
static constexpr size_t STATEOUT_HEADER_SIZE = sizeof(STATEOUT_HEADER) / sizeof(STATEOUT_HEADER[0]);
class CSVLogger {
public:
    CSVLogger() = default;

    CSVLogger(const char* filename, std::vector<std::string>* header) {
        // if(strstr(filename, "") == 0)return;    // If empty dont log anything
        init(filename, header);
    }

    ~CSVLogger() {
        if (_file.is_open()) {
            _file.flush();
            _file.close();
        }
    }

    bool init(const char* filename, std::vector<std::string>* header) {
        if (header == nullptr) {
            Logger(ERROR, "CSVLogger::init => header is null");
            return false;
        }

        if (_file.is_open()) {
            _file.flush();
            _file.close();
        }

        if (!header->empty()) {     // Writer: borrar contenido anterior
            _file.open(filename, std::ios::out | std::ios::trunc);
            if (!_file.is_open()) {
                Logger(ERROR, "Unable to open logging file for writing");
                return false;
            }

            for (size_t i = 0; i < header->size(); ++i) {
                _file << header->at(i);
                if (i + 1 < header->size()) {
                    _file << ",";
                }
            }
            _file << "\n";
            _file.flush();
            return true;
        } else {                    // Reader: no borrar contenido
            _file.open(filename, std::ios::in);
            if (!_file.is_open()) {
                Logger(ERROR, "Unable to open logging file for reading");
                return false;
            }

            std::string line;
            if (!std::getline(_file, line)) {
                Logger(ERROR, "Unable to read CSV header");
                return false;
            }

            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ',')) {
                header->push_back(cell);
            }

            return true;
        }
    }

    void addRow(const std::vector<double>& row) {
        if (!_file.is_open()) {
            return;
        }

        for (size_t i = 0; i < row.size(); ++i) {
            _file << row[i];
            if (i + 1 < row.size()) {
                _file << ",";
            }
        }
        _file << "\n";
    }

    bool readRow(std::vector<double>* row) {
        if (!_file.is_open() || row == nullptr) {
            return false;
        }

        row->clear();

        std::string line;
        if (!std::getline(_file, line)) {
            return false;
        }

        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row->push_back(std::stod(cell));
        }

        return true;
    }

    bool isOpen() const {
        return _file.is_open();
    }

private:
    std::fstream _file;
};
