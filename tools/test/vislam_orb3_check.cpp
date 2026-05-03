#include <atomic>
#include <csignal>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <librealsense2/rs.hpp>
#include <System.h>

namespace
{

std::atomic<bool> g_run(true);
std::atomic<bool> g_playback_finished(false);

void SigIntHandler(int)
{
    g_run = false;
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

    throw std::runtime_error("Formato de color no soportado por este ejemplo: " +
                             std::to_string(static_cast<int>(fmt)));
}

void PrintColorCalibration(const rs2::pipeline_profile& profile)
{
    try
    {
        const rs2::stream_profile cam_stream = profile.get_stream(RS2_STREAM_COLOR);
        const rs2_intrinsics intr = cam_stream.as<rs2::video_stream_profile>().get_intrinsics();

        std::cout << "\n========== Perfil COLOR usado ==========" << std::endl;
        std::cout << "width  = " << intr.width << std::endl;
        std::cout << "height = " << intr.height << std::endl;
        std::cout << "fx     = " << intr.fx << std::endl;
        std::cout << "fy     = " << intr.fy << std::endl;
        std::cout << "cx     = " << intr.ppx << std::endl;
        std::cout << "cy     = " << intr.ppy << std::endl;
        std::cout << "k1..k5 = "
                  << intr.coeffs[0] << ", "
                  << intr.coeffs[1] << ", "
                  << intr.coeffs[2] << ", "
                  << intr.coeffs[3] << ", "
                  << intr.coeffs[4] << std::endl;

        std::cout << "\n========== Pega esto en tu YAML monocular ==========" << std::endl;
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
        std::cout << "Camera.fps: 30.0\n";
        std::cout << "Camera.RGB: 1\n";
        std::cout << "===================================================\n" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[WARN] No se pudo imprimir la calibracion del stream color: "
                  << e.what() << std::endl;
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 5)
    {
        std::cerr << "Uso: ./mono_realsense_color_bag_sync "
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

    ORB_SLAM3::System SLAM(voc_file,
                           settings_file,
                           ORB_SLAM3::System::MONOCULAR,
                           true,
                           0,
                           bag_file);

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_device_from_file(bag_file, false);
    cfg.enable_stream(RS2_STREAM_COLOR);

    rs2::pipeline_profile profile;
    std::shared_ptr<rs2::playback> playback_device;

    try
    {
        profile = pipe.start(cfg);
        rs2::device dev = profile.get_device();

        if (dev.is<rs2::playback>())
        {
            playback_device = std::make_shared<rs2::playback>(dev.as<rs2::playback>());
            playback_device->set_real_time(false);
            playback_device->set_status_changed_callback([](rs2_playback_status status)
            {
                if (status == RS2_PLAYBACK_STATUS_STOPPED)
                {
                    g_playback_finished = true;
                    std::printf("[PLAYBACK] STOPPED\n");
                    std::fflush(stdout);
                }
            });
        }

        PrintColorCalibration(profile);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error iniciando RealSense bag playback: " << e.what() << std::endl;
        return 1;
    }

    const float image_scale = 1.0f;
    int frame_count = 0;
    double last_ts = -1.0;

    while (g_run.load())
    {
        rs2::frameset frames;
        const bool got_frames = pipe.try_wait_for_frames(&frames, 5000);

        if (!got_frames)
        {
            if (g_playback_finished.load())
            {
                std::printf("[MAIN ] fin de playback\n");
                std::fflush(stdout);
                break;
            }

            std::printf("[WAIT ] sin frames de color todavia\n");
            std::fflush(stdout);
            continue;
        }

        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame)
        {
            std::printf("[SKIP ] frameset sin color\n");
            std::fflush(stdout);
            continue;
        }

        const double image_ts = frames.get_timestamp() * 1e-3;
        if (last_ts > 0.0 && std::abs(image_ts - last_ts) < 1e-6)
        {
            std::printf("[SKIP ] timestamp repetido ts=%.6f\n", image_ts);
            std::fflush(stdout);
            continue;
        }
        last_ts = image_ts;

        cv::Mat image_gray;
        try
        {
            image_gray = ConvertFrameToGray(color_frame);
        }
        catch (const std::exception& e)
        {
            std::cerr << "[WARN] Error convirtiendo imagen a gris: " << e.what() << std::endl;
            continue;
        }

        if (image_gray.empty())
        {
            std::printf("[SKIP ] imagen vacia\n");
            std::fflush(stdout);
            continue;
        }

        if (image_scale != 1.0f)
        {
            const int new_width = static_cast<int>(image_gray.cols * image_scale);
            const int new_height = static_cast<int>(image_gray.rows * image_scale);
            cv::resize(image_gray, image_gray, cv::Size(new_width, new_height));
        }

        ++frame_count;
        if (frame_count <= 10 || (frame_count % 30) == 0)
        {
            std::printf("[TRACK] frame=%d ts=%.6f size=%dx%d\n",
                        frame_count,
                        image_ts,
                        image_gray.cols,
                        image_gray.rows);
            std::fflush(stdout);
        }

        SLAM.TrackMonocular(image_gray, image_ts);
    }

    g_run = false;

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
