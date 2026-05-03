#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <librealsense2/rs.hpp>
#include <System.h>

namespace
{

std::atomic<bool> g_run(true);


struct ImagePacket
{
    cv::Mat gray;
    double ts = -1.0;
};

struct SharedBuffer
{
    std::mutex mtx;
    std::condition_variable cv_data;

    std::deque<ImagePacket> image_queue;

    std::vector<rs2_vector> gyro_data;
    std::vector<double> gyro_timestamps;

    std::vector<rs2_vector> accel_sync_data;
    std::vector<double> accel_sync_timestamps;

    rs2_vector prev_accel = {};
    rs2_vector curr_accel = {};
    double prev_accel_ts = 0.0;
    double curr_accel_ts = 0.0;
    bool have_accel = false;

    bool have_first_gyro = false;
    double first_gyro_ts = -1.0;

    bool playback_finished = false;
};
SharedBuffer* g_shared_ptr = nullptr;

void SigIntHandler(int)
{
    g_run = false;
    if (g_shared_ptr)
        g_shared_ptr->cv_data.notify_all();
}
rs2_vector InterpolateMeasure(const double target_time,
                              const rs2_vector current_data,
                              const double current_time,
                              const rs2_vector prev_data,
                              const double prev_time)
{
    if (prev_time == 0.0)
        return current_data;

    if (target_time >= current_time)
        return current_data;

    if (target_time <= prev_time)
        return prev_data;

    const double dt = current_time - prev_time;
    if (dt <= 1e-9)
        return current_data;

    const double alpha = (target_time - prev_time) / dt;

    rs2_vector out;
    out.x = static_cast<float>(prev_data.x + (current_data.x - prev_data.x) * alpha);
    out.y = static_cast<float>(prev_data.y + (current_data.y - prev_data.y) * alpha);
    out.z = static_cast<float>(prev_data.z + (current_data.z - prev_data.z) * alpha);
    return out;
}

void SyncAccelToGyro(SharedBuffer& shared)
{
    while (shared.gyro_timestamps.size() > shared.accel_sync_timestamps.size())
    {
        const std::size_t idx = shared.accel_sync_timestamps.size();
        const double target_time = shared.gyro_timestamps[idx];

        rs2_vector interp = {};
        if (shared.have_accel)
        {
            interp = InterpolateMeasure(target_time,
                                        shared.curr_accel,
                                        shared.curr_accel_ts,
                                        shared.prev_accel,
                                        shared.prev_accel_ts);
        }

        shared.accel_sync_data.push_back(interp);
        shared.accel_sync_timestamps.push_back(target_time);
    }
}

cv::Mat ConvertFrameToGray(const rs2::video_frame& vf)
{
    const int w = vf.get_width();
    const int h = vf.get_height();
    const rs2_format fmt = vf.get_profile().format();

    if (fmt == RS2_FORMAT_Y8)
    {
        cv::Mat gray(h, w, CV_8UC1, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        return gray.clone();
    }
    if (fmt == RS2_FORMAT_BGR8)
    {
        cv::Mat bgr(h, w, CV_8UC3, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    if (fmt == RS2_FORMAT_RGB8)
    {
        cv::Mat rgb(h, w, CV_8UC3, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat gray;
        cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
        return gray;
    }
    if (fmt == RS2_FORMAT_RGBA8)
    {
        cv::Mat rgba(h, w, CV_8UC4, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat gray;
        cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
        return gray;
    }
    if (fmt == RS2_FORMAT_BGRA8)
    {
        cv::Mat bgra(h, w, CV_8UC4, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat gray;
        cv::cvtColor(bgra, gray, cv::COLOR_BGRA2GRAY);
        return gray;
    }
    if (fmt == RS2_FORMAT_YUYV)
    {
        cv::Mat yuyv(h, w, CV_8UC2, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat gray;
        cv::cvtColor(yuyv, gray, cv::COLOR_YUV2GRAY_YUY2);
        return gray;
    }

    throw std::runtime_error("Formato de color no soportado: " +
                             std::to_string(static_cast<int>(fmt)));
}

void PrintCalibrationForYaml(const rs2::pipeline_profile& profile)
{
    try
    {
        const rs2::stream_profile cam_stream = profile.get_stream(RS2_STREAM_COLOR);
        const rs2::stream_profile imu_stream = profile.get_stream(RS2_STREAM_GYRO);

        const rs2_intrinsics intr = cam_stream.as<rs2::video_stream_profile>().get_intrinsics();
        const rs2_extrinsics c_to_b = cam_stream.get_extrinsics_to(imu_stream);

        double R_cb[3][3];
        double R_bc[3][3];
        double t_cb[3];
        double t_bc[3];

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                R_cb[r][c] = static_cast<double>(c_to_b.rotation[c * 3 + r]);
            }
            t_cb[r] = static_cast<double>(c_to_b.translation[r]);
        }

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                R_bc[r][c] = R_cb[c][r];
            }
        }

        for (int r = 0; r < 3; ++r)
        {
            t_bc[r] = -(R_bc[r][0] * t_cb[0] + R_bc[r][1] * t_cb[1] + R_bc[r][2] * t_cb[2]);
        }

        std::cout << "\n========== Perfil COLOR usado ==========" << std::endl;
        std::cout << "width  = " << intr.width << std::endl;
        std::cout << "height = " << intr.height << std::endl;
        std::cout << "fx     = " << intr.fx << std::endl;
        std::cout << "fy     = " << intr.fy << std::endl;
        std::cout << "cx     = " << intr.ppx << std::endl;
        std::cout << "cy     = " << intr.ppy << std::endl;
        std::cout << "model  = " << intr.model << std::endl;
        std::cout << "k1..k5 = "
                  << intr.coeffs[0] << ", "
                  << intr.coeffs[1] << ", "
                  << intr.coeffs[2] << ", "
                  << intr.coeffs[3] << ", "
                  << intr.coeffs[4] << std::endl;

        std::cout << "\n========== Pega esto en tu YAML monocular-inercial ==========" << std::endl;
        std::cout << std::fixed << std::setprecision(9);
        std::cout << "Camera.type: \"PinHole\"\n";
        std::cout << "Camera.fx: " << intr.fx << "\n";
        std::cout << "Camera.fy: " << intr.fy << "\n";
        std::cout << "Camera.cx: " << intr.ppx << "\n";
        std::cout << "Camera.cy: " << intr.ppy << "\n";
        std::cout << "Camera.k1: " << intr.coeffs[0] << "\n";
        std::cout << "Camera.k2: " << intr.coeffs[1] << "\n";
        std::cout << "Camera.p1: " << intr.coeffs[2] << "\n";
        std::cout << "Camera.p2: " << intr.coeffs[3] << "\n";
        std::cout << "Camera.width: " << intr.width << "\n";
        std::cout << "Camera.height: " << intr.height << "\n";
        std::cout << "Tbc: !!opencv-matrix\n";
        std::cout << "   rows: 4\n   cols: 4\n   dt: f\n   data: ["
                  << R_bc[0][0] << ", " << R_bc[0][1] << ", " << R_bc[0][2] << ", " << t_bc[0] << ",\n"
                  << "         " << R_bc[1][0] << ", " << R_bc[1][1] << ", " << R_bc[1][2] << ", " << t_bc[1] << ",\n"
                  << "         " << R_bc[2][0] << ", " << R_bc[2][1] << ", " << R_bc[2][2] << ", " << t_bc[2] << ",\n"
                  << "         0.0, 0.0, 0.0, 1.0]\n";
        std::cout << "IMU.T_b_c1: !!opencv-matrix\n";
        std::cout << "   rows: 4\n   cols: 4\n   dt: f\n   data: ["
                  << R_bc[0][0] << ", " << R_bc[0][1] << ", " << R_bc[0][2] << ", " << t_bc[0] << ",\n"
                  << "         " << R_bc[1][0] << ", " << R_bc[1][1] << ", " << R_bc[1][2] << ", " << t_bc[1] << ",\n"
                  << "         " << R_bc[2][0] << ", " << R_bc[2][1] << ", " << R_bc[2][2] << ", " << t_bc[2] << ",\n"
                  << "         0.0, 0.0, 0.0, 1.0]\n";
        std::cout << "===========================================================\n" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[WARN] No se pudo imprimir intrinsecos/extrinsecos: "
                  << e.what() << std::endl;
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 5)
    {
        std::cerr << "Uso: ./orb_inertial "
                  << "path_to_vocabulary path_to_settings path_to_bag [output_prefix]"
                  << std::endl;
        return 1;
    }

    const std::string voc_file = argv[1];
    const std::string settings_file = argv[2];
    const std::string bag_file = argv[3];
    const std::string output_prefix = (argc == 5) ? argv[4] : std::string();

#ifdef _WIN32
    std::signal(SIGINT, SigIntHandler);
#else
    struct sigaction sigint_handler;
    sigint_handler.sa_handler = SigIntHandler;
    sigemptyset(&sigint_handler.sa_mask);
    sigint_handler.sa_flags = 0;
    sigaction(SIGINT, &sigint_handler, nullptr);
#endif

    SharedBuffer shared;
    g_shared_ptr = &shared;

auto rs_callback = [&shared](const rs2::frame& frame)
{
    try
    {
        if (rs2::frameset fs = frame.as<rs2::frameset>())
        {
            rs2::video_frame color_frame = fs.get_color_frame();
            if (color_frame)
            {
                cv::Mat gray = ConvertFrameToGray(color_frame);
                const double ts = fs.get_timestamp() * 1e-3;

                {
                    std::lock_guard<std::mutex> lock(shared.mtx);
                    ImagePacket pkt;
                    pkt.gray = std::move(gray);
                    pkt.ts = ts;
                    shared.image_queue.push_back(std::move(pkt));
                    std::printf("[COLOR] frameset ts=%.6f q=%zu\n", ts, shared.image_queue.size());
                    std::fflush(stdout);
                }

                shared.cv_data.notify_one();
            }
            return;
        }

        if (rs2::video_frame vf = frame.as<rs2::video_frame>())
        {
            if (vf.get_profile().stream_type() == RS2_STREAM_COLOR)
            {
                cv::Mat gray = ConvertFrameToGray(vf);
                const double ts = vf.get_timestamp() * 1e-3;

                {
                    std::lock_guard<std::mutex> lock(shared.mtx);
                    ImagePacket pkt;
                    pkt.gray = std::move(gray);
                    pkt.ts = ts;
                    shared.image_queue.push_back(std::move(pkt));
                    std::printf("[COLOR] video_frame ts=%.6f q=%zu\n", ts, shared.image_queue.size());
                    std::fflush(stdout);
                }

                shared.cv_data.notify_one();
            }
            return;
        }

        if (rs2::motion_frame mf = frame.as<rs2::motion_frame>())
        {
            const double ts = mf.get_timestamp() * 1e-3;
            const rs2_vector data = mf.get_motion_data();
            const rs2_stream stream_type = mf.get_profile().stream_type();

            {
                std::lock_guard<std::mutex> lock(shared.mtx);

                if (stream_type == RS2_STREAM_GYRO)
                {
                    shared.gyro_data.push_back(data);
                    shared.gyro_timestamps.push_back(ts);

                    if (!shared.have_first_gyro)
                    {
                        shared.have_first_gyro = true;
                        shared.first_gyro_ts = ts;
                    }
                }
                else if (stream_type == RS2_STREAM_ACCEL)
                {
                    shared.prev_accel = shared.curr_accel;
                    shared.prev_accel_ts = shared.curr_accel_ts;
                    shared.curr_accel = data;
                    shared.curr_accel_ts = ts;
                    shared.have_accel = true;
                    SyncAccelToGyro(shared);
                }
            }

            shared.cv_data.notify_one();
            return;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[WARN] Error en callback de RealSense: " << e.what() << std::endl;
    }
};

    ORB_SLAM3::System SLAM(voc_file,
                           settings_file,
                           ORB_SLAM3::System::IMU_MONOCULAR,
                           true,
                           0,
                           bag_file);

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_device_from_file(bag_file, false);
    cfg.enable_stream(RS2_STREAM_COLOR);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

    rs2::pipeline_profile profile;
    std::shared_ptr<rs2::playback> playback_device;

    try
    {
        profile = pipe.start(cfg, rs_callback);
        rs2::device dev = profile.get_device();

        if (dev.is<rs2::playback>())
        {
            playback_device = std::make_shared<rs2::playback>(dev.as<rs2::playback>());
            playback_device->set_real_time(false);
            playback_device->set_status_changed_callback([&shared](rs2_playback_status status)
            {
                if (status == RS2_PLAYBACK_STATUS_STOPPED)
                {
                    {
                        std::lock_guard<std::mutex> lock(shared.mtx);
                        shared.playback_finished = true;
                    }
                    shared.cv_data.notify_all();
                    std::printf("[PLAYBACK] STOPPED\n");
                    std::fflush(stdout);
                }
            });
        }

        PrintCalibrationForYaml(profile);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error iniciando RealSense bag playback: " << e.what() << std::endl;
        return 1;
    }

    const float image_scale = 1.0f;
    int frame_count = 0;
    double last_processed_image_ts = -1.0;
    bool have_initialized_tracking = false;

    while (g_run.load())
    {
        cv::Mat image_gray;
        double image_ts = -1.0;
        std::vector<ORB_SLAM3::IMU::Point> imu_meas;
        std::size_t erase_until = 0;

        {
            std::unique_lock<std::mutex> lock(shared.mtx);

            shared.cv_data.wait_for(lock, std::chrono::milliseconds(500), [&shared]() {
                return !g_run.load() ||
                    shared.playback_finished ||
                    !shared.image_queue.empty();
            });

            if (!g_run.load())
                break;

            if (shared.image_queue.empty() && !shared.playback_finished)
            {
                std::printf("[WAIT ] sin nuevas imagenes aun\n");
                std::fflush(stdout);
                continue;
            }

            if (!g_run.load())
                break;

            if (shared.image_queue.empty())
            {
                if (shared.playback_finished)
                    break;
                continue;
            }

            SyncAccelToGyro(shared);

            // Descarta cualquier imagen anterior al primer gyro.
            while (!shared.image_queue.empty() &&
                   shared.have_first_gyro &&
                   shared.image_queue.front().ts < shared.first_gyro_ts)
            {
                std::printf("[DROP ] imagen anterior al primer gyro: img_ts=%.6f first_gyro=%.6f\n",
                            shared.image_queue.front().ts, shared.first_gyro_ts);
                std::fflush(stdout);
                shared.image_queue.pop_front();
            }

            if (shared.image_queue.empty())
            {
                if (shared.playback_finished)
                    break;
                continue;
            }

            image_gray = shared.image_queue.front().gray.clone();
            image_ts = shared.image_queue.front().ts;
            shared.image_queue.pop_front();

            if (!have_initialized_tracking)
            {
                // Primer frame válido: se envía sin IMU y fija el origen temporal.
                have_initialized_tracking = true;
                last_processed_image_ts = image_ts;
            }
            else
            {
                std::size_t start = 0;
                while (start < shared.gyro_timestamps.size() &&
                       shared.gyro_timestamps[start] <= last_processed_image_ts)
                {
                    ++start;
                }

                std::size_t end = start;
                while (end < shared.gyro_timestamps.size() &&
                       shared.gyro_timestamps[end] <= image_ts)
                {
                    ++end;
                }

                if (end <= start)
                {
                    std::printf("[SKIP ] sin IMU en intervalo (%.6f, %.6f]\n",
                                last_processed_image_ts, image_ts);
                    std::fflush(stdout);

                    // OJO: no actualizamos last_processed_image_ts
                    // OJO: no borramos IMU
                    image_gray.release();
                    image_ts = -1.0;
                }
                else
                {
                    imu_meas.reserve(end - start);
                    for (std::size_t i = start; i < end; ++i)
                    {
                        imu_meas.emplace_back(shared.accel_sync_data[i].x,
                                              shared.accel_sync_data[i].y,
                                              shared.accel_sync_data[i].z,
                                              shared.gyro_data[i].x,
                                              shared.gyro_data[i].y,
                                              shared.gyro_data[i].z,
                                              shared.gyro_timestamps[i]);
                    }

                    erase_until = end;
                }
            }
        }

        if (!g_run.load())
            break;

        if (image_ts < 0.0 || image_gray.empty())
        {
            if (shared.playback_finished)
            {
                std::lock_guard<std::mutex> lock(shared.mtx);
                if (shared.image_queue.empty())
                    break;
            }
            continue;
        }

        if (image_scale != 1.0f)
        {
            const int new_width = static_cast<int>(image_gray.cols * image_scale);
            const int new_height = static_cast<int>(image_gray.rows * image_scale);
            cv::resize(image_gray, image_gray, cv::Size(new_width, new_height));
        }

        ++frame_count;
        if (frame_count <= 20 || (frame_count % 30) == 0)
        {
            std::printf("[TRACK] frame=%d ts=%.6f size=%dx%d imu=%zu\n",
                        frame_count, image_ts, image_gray.cols, image_gray.rows, imu_meas.size());
            if (!imu_meas.empty())
            {
                std::printf("[IMU  ] first=%.6f last=%.6f\n",
                            imu_meas.front().t, imu_meas.back().t);
            }
            std::fflush(stdout);
        }

        SLAM.TrackMonocular(image_gray, image_ts, imu_meas);

        {
            std::lock_guard<std::mutex> lock(shared.mtx);

            // Solo ahora consumimos la IMU realmente usada.
            if (erase_until > 0)
            {
                shared.gyro_data.erase(shared.gyro_data.begin(),
                                       shared.gyro_data.begin() + static_cast<std::ptrdiff_t>(erase_until));
                shared.gyro_timestamps.erase(shared.gyro_timestamps.begin(),
                                             shared.gyro_timestamps.begin() + static_cast<std::ptrdiff_t>(erase_until));
                shared.accel_sync_data.erase(shared.accel_sync_data.begin(),
                                             shared.accel_sync_data.begin() + static_cast<std::ptrdiff_t>(erase_until));
                shared.accel_sync_timestamps.erase(shared.accel_sync_timestamps.begin(),
                                                   shared.accel_sync_timestamps.begin() + static_cast<std::ptrdiff_t>(erase_until));
            }
        }

        // Solo actualizamos el último timestamp al haber trackeado de verdad.
        last_processed_image_ts = image_ts;
    }

    g_run = false;
    shared.cv_data.notify_all();

    try
    {
        pipe.stop();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[WARN] Error al parar el pipeline: " << e.what() << std::endl;
    }

    SLAM.Shutdown();

    if (!output_prefix.empty())
    {
        SLAM.SaveKeyFrameTrajectoryTUM(output_prefix + "_kf_tum.txt");
    }

    return 0;
}