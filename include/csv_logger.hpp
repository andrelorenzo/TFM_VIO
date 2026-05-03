#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include "seconds/logger.h"


inline constexpr const char * DEBUG_HEADER[] = {
    "ts", "dt"
    "visx", "visy", "visz", "visroll", "vispitch", "visyaw", "imux", "imuy", "imuz", "imuroll", "imupitch", "imuyaw",   // Visual pose | inertial part
    "gtx", "gty", "gtz", "gtroll", "gtpitch", "gtyaw", "x", "y", "z", "roll", "pitch", "yaw",                           // GT pose  | Solved pose
    "accx", "accy", "accz", "gyrx", "gyry", "gyrz",                                          // Raw data
    "vio_in", "imu_stat", "vio_valid",                                                                                  // Flags

    "dpx", "dpy", "dpz", "dvx", "dvy", "dvz", "dqw", "dqx", "dqy", "dqz",                                              // Pre integration dvalues
};
static constexpr size_t DEBUGHEADER_SIZE = sizeof(DEBUG_HEADER) / sizeof(DEBUG_HEADER[0]);


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
