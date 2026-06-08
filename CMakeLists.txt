cmake_minimum_required(VERSION 3.26)

project(visalm LANGUAGES CXX CUDA)

if(MSVC)
  cmake_policy(SET CMP0091 NEW)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dynamic Cuda
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/")
find_package(CUDAToolkit REQUIRED)

# Static Opencv 
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/thirds/opencv/x64/vc17/staticlib")
set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed NumPy)


# Dynamic Gstreamer
set(GSTREAMER_DIR "C:/Program Files/gstreamer/1.0/msvc_x86_64" CACHE PATH "GStreamer MSVC root")

set(GSTREAMER_INCLUDE_DIRS "${GSTREAMER_DIR}/include/gstreamer-1.0" "${GSTREAMER_DIR}/include/glib-2.0" "${GSTREAMER_DIR}/lib/glib-2.0/include")
set(GSTREAMER_LIB_DIR "${GSTREAMER_DIR}/lib")
set(GSTREAMER_BIN_DIR "${GSTREAMER_DIR}/bin")
set(GSTREAMER_PLUGIN_DIR "${GSTREAMER_DIR}/lib/gstreamer-1.0")
set(GSTREAMER_PLUGIN_SCANNER "${GSTREAMER_DIR}/libexec/gstreamer-1.0/gst-plugin-scanner.exe")

set(GSTREAMER_LIBRARIES "${GSTREAMER_LIB_DIR}/gstreamer-1.0.lib" "${GSTREAMER_LIB_DIR}/gstapp-1.0.lib" "${GSTREAMER_LIB_DIR}/gstbase-1.0.lib" "${GSTREAMER_LIB_DIR}/gstvideo-1.0.lib" "${GSTREAMER_LIB_DIR}/gobject-2.0.lib" "${GSTREAMER_LIB_DIR}/glib-2.0.lib" "${GSTREAMER_LIB_DIR}/gio-2.0.lib" "${GSTREAMER_LIB_DIR}/gmodule-2.0.lib")

if(EXISTS "${GSTREAMER_LIB_DIR}/intl.lib")
  list(APPEND GSTREAMER_LIBRARIES "${GSTREAMER_LIB_DIR}/intl.lib")
elseif(EXISTS "${GSTREAMER_LIB_DIR}/libintl.lib")
  list(APPEND GSTREAMER_LIBRARIES "${GSTREAMER_LIB_DIR}/libintl.lib")
endif()


file(GLOB VIO_PLOTTER_SRC CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/plotter/*.cpp"
)
file(GLOB VIO_SRC CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)


add_executable(vio 
    ${VIO_PLOTTER_SRC}
    ${VIO_SRC}
    thirds/imgui/imgui.cpp
    thirds/imgui/imgui_draw.cpp
    thirds/imgui/imgui_tables.cpp
    thirds/imgui/imgui_widgets.cpp
    thirds/imgui/imgui_demo.cpp
    thirds/imgui/backends/imgui_impl_win32.cpp
    thirds/imgui/backends/imgui_impl_dx11.cpp
    thirds/implot/implot.cpp
    thirds/implot/implot_items.cpp
    thirds/implot/implot_demo.cpp
    src/preprocess.cu
)
target_include_directories(vio PRIVATE 
  include 
  "${CMAKE_SOURCE_DIR}/src/plotter" 
  "${CMAKE_SOURCE_DIR}/thirds/realsense/include" 
  ${OpenCV_INCLUDE_DIRS} 
  ${GSTREAMER_INCLUDE_DIRS} 
  "${CMAKE_SOURCE_DIR}/thirds/onnxruntime/include" 
  "${CMAKE_SOURCE_DIR}/thirds/Eigen/eigen-3.4.0/" 
  thirds/imgui 
  thirds/imgui/backends 
  thirds/implot
)
target_link_libraries(vio PRIVATE 
    opencv_world 
    "${CMAKE_SOURCE_DIR}/thirds/realsense/lib/x64/realsense2.lib"
    "${CMAKE_SOURCE_DIR}/thirds/seconds/Project_commons.lib"
    "${CMAKE_SOURCE_DIR}/thirds/onnxruntime/lib/onnxruntime.lib"
    ${GSTREAMER_LIBRARIES}
    d3d11
    dxgi
    d3dcompiler
    CUDA::cudart_static
    Ws2_32 
    Mswsock
)
if(MSVC)
    target_compile_options(vio PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj /wd4267>
    )
endif()
target_compile_definitions(vio PRIVATE USE_CUDA)


# Runtime DLLs needed by ONNX Runtime CUDA provider
if(WIN32)
  set(ORT_RUNTIME_DLLS
    "${CMAKE_SOURCE_DIR}/thirds/onnxruntime/lib/onnxruntime.dll"
    "${CMAKE_SOURCE_DIR}/thirds/onnxruntime/lib/onnxruntime_providers_shared.dll"
    "${CMAKE_SOURCE_DIR}/thirds/onnxruntime/lib/onnxruntime_providers_cuda.dll"
  )

  set(CUDA_RUNTIME_DLLS
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/cublas64_12.dll"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/cublasLt64_12.dll"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/cudart64_12.dll"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/cufft64_11.dll"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/curand64_10.dll"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/cusolver64_11.dll"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/cusparse64_12.dll"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvrtc64_120_0.dll"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvrtc-builtins64_129.dll"
  )

  file(GLOB CUDNN_RUNTIME_DLLS
    "C:/Program Files/NVIDIA/CUDNN/v9.19/bin/12.9/x64/*.dll"
  )

  foreach(RUNTIME_DLL IN LISTS ORT_RUNTIME_DLLS CUDA_RUNTIME_DLLS CUDNN_RUNTIME_DLLS)
    if(EXISTS "${RUNTIME_DLL}")
      add_custom_command(TARGET vio POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${RUNTIME_DLL}"
          "$<TARGET_FILE_DIR:vio>"
      )
    endif()
  endforeach()

  file(GLOB GSTREAMER_RUNTIME_DLLS "${GSTREAMER_BIN_DIR}/*.dll")

  foreach(GSTREAMER_DLL IN LISTS GSTREAMER_RUNTIME_DLLS)
    if(EXISTS "${GSTREAMER_DLL}")
      add_custom_command(TARGET vio POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GSTREAMER_DLL}" "$<TARGET_FILE_DIR:vio>")
    endif()
  endforeach()

  if(EXISTS "${GSTREAMER_PLUGIN_DIR}")
    add_custom_command(TARGET vio POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${GSTREAMER_PLUGIN_DIR}" "$<TARGET_FILE_DIR:vio>/gstreamer-1.0")
  endif()

  if(EXISTS "${GSTREAMER_PLUGIN_SCANNER}")
    add_custom_command(TARGET vio POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vio>/libexec/gstreamer-1.0")
    add_custom_command(TARGET vio POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GSTREAMER_PLUGIN_SCANNER}" "$<TARGET_FILE_DIR:vio>/libexec/gstreamer-1.0/gst-plugin-scanner.exe")
  endif()

  set_property(TARGET vio PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=${GSTREAMER_BIN_DIR}$<SEMICOLON>%PATH%\nGST_PLUGIN_PATH=${GSTREAMER_PLUGIN_DIR}\nGST_PLUGIN_SCANNER=${GSTREAMER_PLUGIN_SCANNER}")
endif()

