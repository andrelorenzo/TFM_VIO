#include <atomic>
#include <csignal>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

namespace
{
std::atomic<bool> g_run(true);

void SigIntHandler(int)
{
    g_run = false;
}

cv::Mat ConvertToBgr(const rs2::video_frame& vf)
{
    const int w = vf.get_width();
    const int h = vf.get_height();
    const rs2_format fmt = vf.get_profile().format();

    if (fmt == RS2_FORMAT_BGR8)
    {
        cv::Mat bgr(h, w, CV_8UC3, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        return bgr.clone();
    }
    if (fmt == RS2_FORMAT_RGB8)
    {
        cv::Mat rgb(h, w, CV_8UC3, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }
    if (fmt == RS2_FORMAT_RGBA8)
    {
        cv::Mat rgba(h, w, CV_8UC4, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        return bgr;
    }
    if (fmt == RS2_FORMAT_BGRA8)
    {
        cv::Mat bgra(h, w, CV_8UC4, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat bgr;
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    if (fmt == RS2_FORMAT_YUYV)
    {
        cv::Mat yuyv(h, w, CV_8UC2, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat bgr;
        cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
        return bgr;
    }
    if (fmt == RS2_FORMAT_Y8)
    {
        cv::Mat gray(h, w, CV_8UC1, const_cast<void*>(vf.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }

    throw std::runtime_error("Formato no soportado");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Uso: cb_test_queue.exe path_to_bag" << std::endl;
        return 1;
    }

    const std::string bag_file = argv[1];

#ifdef _WIN32
    std::signal(SIGINT, SigIntHandler);
#else
    struct sigaction sigint_handler;
    sigint_handler.sa_handler = SigIntHandler;
    sigemptyset(&sigint_handler.sa_mask);
    sigint_handler.sa_flags = 0;
    sigaction(SIGINT, &sigint_handler, nullptr);
#endif

    rs2::frame_queue q(1000);

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_device_from_file(bag_file, false);
    cfg.enable_stream(RS2_STREAM_COLOR);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

    rs2::pipeline_profile profile;
    std::shared_ptr<rs2::playback> playback;
    bool playback_finished = false;

    try
    {
        profile = pipe.start(cfg, q);
        rs2::device dev = profile.get_device();

        if (dev.is<rs2::playback>())
        {
            playback = std::make_shared<rs2::playback>(dev.as<rs2::playback>());
            playback->set_real_time(false);
            playback->set_status_changed_callback([&playback_finished](rs2_playback_status status)
            {
                if (status == RS2_PLAYBACK_STATUS_STOPPED)
                {
                    playback_finished = true;
                    std::printf("[PLAYBACK] STOPPED\n");
                    std::fflush(stdout);
                }
            });
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error iniciando bag: " << e.what() << std::endl;
        return 1;
    }

    int color_count = 0;
    int gyro_count = 0;
    int accel_count = 0;

    cv::namedWindow("bag_queue_debug", cv::WINDOW_NORMAL);

    while (g_run.load())
    {
        rs2::frame f;
        if (!q.try_wait_for_frame(&f, 500))
        {
            if (playback_finished)
            {
                std::printf("[MAIN ] fin de playback\n");
                std::fflush(stdout);
                break;
            }

            std::printf("[WAIT ] sin frames en queue\n");
            std::fflush(stdout);
            continue;
        }

        try
        {
            if (rs2::frameset fs = f.as<rs2::frameset>())
            {
                rs2::video_frame color = fs.get_color_frame();
                if (color)
                {
                    cv::Mat bgr = ConvertToBgr(color);
                    const double ts = fs.get_timestamp() * 1e-3;

                    ++color_count;
                    if (color_count <= 10 || (color_count % 30) == 0)
                    {
                        std::printf("[COLOR][frameset] n=%d ts=%.6f size=%dx%d\n",
                                    color_count, ts, bgr.cols, bgr.rows);
                        std::fflush(stdout);
                    }

                    cv::imshow("bag_queue_debug", bgr);
                    cv::waitKey(1);
                }
                continue;
            }

            if (rs2::video_frame vf = f.as<rs2::video_frame>())
            {
                if (vf.get_profile().stream_type() == RS2_STREAM_COLOR)
                {
                    cv::Mat bgr = ConvertToBgr(vf);
                    const double ts = vf.get_timestamp() * 1e-3;

                    ++color_count;
                    if (color_count <= 10 || (color_count % 30) == 0)
                    {
                        std::printf("[COLOR][video_frame] n=%d ts=%.6f size=%dx%d\n",
                                    color_count, ts, bgr.cols, bgr.rows);
                        std::fflush(stdout);
                    }

                    cv::imshow("bag_queue_debug", bgr);
                    cv::waitKey(1);
                }
                continue;
            }

            if (rs2::motion_frame mf = f.as<rs2::motion_frame>())
            {
                const double ts = mf.get_timestamp() * 1e-3;
                const rs2_vector d = mf.get_motion_data();

                if (mf.get_profile().stream_type() == RS2_STREAM_GYRO)
                {
                    ++gyro_count;
                    if (gyro_count <= 20 || (gyro_count % 200) == 0)
                    {
                        std::printf("[GYRO ] n=%d ts=%.6f xyz=(%.6f, %.6f, %.6f)\n",
                                    gyro_count, ts, d.x, d.y, d.z);
                        std::fflush(stdout);
                    }
                }
                else if (mf.get_profile().stream_type() == RS2_STREAM_ACCEL)
                {
                    ++accel_count;
                    if (accel_count <= 20 || (accel_count % 100) == 0)
                    {
                        std::printf("[ACCEL] n=%d ts=%.6f xyz=(%.6f, %.6f, %.6f)\n",
                                    accel_count, ts, d.x, d.y, d.z);
                        std::fflush(stdout);
                    }
                }
                continue;
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "[WARN] Error procesando frame: " << e.what() << std::endl;
        }
    }

    try
    {
        pipe.stop();
    }
    catch (...) {}

    cv::destroyAllWindows();
    return 0;
}