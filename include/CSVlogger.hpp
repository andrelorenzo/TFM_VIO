#pragma once

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

class CsvLogger {
public:
    CsvLogger() = default;
    ~CsvLogger() { close(); }
    // const char * header[] = {   "x","y","z","imu_x", "imu_y", "imu_z", "vis_x", "vis_y", "vis_z"
    //                             "vx", "vy", "vz", "imu_vx", "imu_vy", "imu_vz", "vis_vx", "vis_vy", "vis_vz",
    //                             "ax", "ay", "az", "imu_ax", "imu_ay", "imu_az", "vis_ax", "vis_ay", "vis_az",
    //                             };

    bool init(const std::string& filename,const std::vector<std::string>& header,bool append = false){
        close();

        std::ios::openmode mode = std::ios::out;
        if (append) {
            mode |= std::ios::app;
        } else {
            mode |= std::ios::trunc;
        }

        file_.open(filename, mode);
        if (!file_.is_open()) {
            return false;
        }

        filename_ = filename;

        if (!append) {
            writeRow(header);
        }

        return true;
    }

    bool addRow(const std::vector<std::string>& row){
        if (!file_.is_open()) {
            return false;
        }

        writeRow(row);
        return file_.good();
    }

    template <typename T>
    bool addRow(const std::vector<T>& row)
    {
        if (!file_.is_open()) {
            return false;
        }

        std::vector<std::string> fields;
        fields.reserve(row.size());

        for (const auto& v : row) {
            std::ostringstream oss;

            if constexpr (std::is_floating_point_v<T>) {
                oss << std::fixed << std::setprecision(9) << v;
            } else {
                oss << v;
            }

            fields.push_back(oss.str());
        }

        writeRow(fields);
        return file_.good();
    }

    bool isOpen() const
    {
        return file_.is_open();
    }

    void flush()
    {
        if (file_.is_open()) {
            file_.flush();
        }
    }

    void close()
    {
        if (file_.is_open()) {
            file_.flush();
            file_.close();
        }
    }

    const std::string& filename() const
    {
        return filename_;
    }

private:
    std::ofstream file_;
    std::string filename_;

    static std::string escapeCsv(const std::string& s)
    {
        bool needs_quotes = false;
        for (char c : s) {
            if (c == ',' || c == '"' || c == '\n' || c == '\r') {
                needs_quotes = true;
                break;
            }
        }

        if (!needs_quotes) {
            return s;
        }

        std::string out;
        out.reserve(s.size() + 4);
        out.push_back('"');

        for (char c : s) {
            if (c == '"') {
                out.push_back('"');
            }
            out.push_back(c);
        }

        out.push_back('"');
        return out;
    }

    void writeRow(const std::vector<std::string>& row)
    {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) {
                file_ << ",";
            }
            file_ << escapeCsv(row[i]);
        }
        file_ << "\n";
    }

};