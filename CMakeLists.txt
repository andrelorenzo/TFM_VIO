cmake_minimum_required(VERSION 3.16)
project(visalm LANGUAGES CXX)

if(MSVC)
  cmake_policy(SET CMP0091 NEW)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/thirds/opencv/x64/vc17/staticlib")
set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed NumPy)


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
)
target_include_directories(vio PRIVATE 
    include
    "${CMAKE_SOURCE_DIR}/src/plotter"
    "${CMAKE_SOURCE_DIR}/thirds/realsense/include" 
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/thirds/Eigen/eigen-3.4.0/"
    thirds/imgui
    thirds/imgui/backends
    thirds/implot
)
target_link_libraries(vio PRIVATE 
    opencv_world 
    "${CMAKE_SOURCE_DIR}/thirds/realsense/lib/x64/realsense2.lib"
    "${CMAKE_SOURCE_DIR}/thirds/seconds/Project_commons.lib"
    d3d11
    dxgi
    d3dcompiler
)
if(MSVC)
    target_compile_options(vio PRIVATE /bigobj)
endif()
