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

# file(GLOB RVIO_SOURCES CONFIGURE_DEPENDS
#     "${CMAKE_SOURCE_DIR}/src/rvio2/*.cc"
# )


file(GLOB RVIO2_CORE_SRC CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/rvio2/*.cc"
)


# MAIN PROGRAMM
add_executable(${CMAKE_PROJECT_NAME}
    src/main_own.cpp
    src/imu.cpp
    src/source_man.cpp
    src/gt_estimator.cpp
    src/visual_odometry.cpp
    src/feature_tracker.cpp
    # ${RVIO_SOURCES}
)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE 
    include
    "${CMAKE_SOURCE_DIR}/thirds/realsense/include" 
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/thirds/Eigen/eigen-3.4.0/"
    "${CMAKE_SOURCE_DIR}/src/rvio2"
    "${CMAKE_SOURCE_DIR}/src/rvio2/util/"
)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE 
    opencv_world 
    "${CMAKE_SOURCE_DIR}/thirds/realsense/lib/x64/realsense2.lib"
    "${CMAKE_SOURCE_DIR}/thirds/seconds/Project_commons.lib"
)
if(MSVC)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE /bigobj)
endif()


# NEW MAIN PROGRAMM
add_library(matplotlib_cpp INTERFACE)
target_compile_features(matplotlib_cpp INTERFACE cxx_std_11)
target_link_libraries(matplotlib_cpp INTERFACE Python3::Python)
if(Python3_NumPy_FOUND)
    target_link_libraries(matplotlib_cpp INTERFACE Python3::NumPy)
else()
    target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
endif()


file(GLOB NEWMAIN_SRC CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/new_vers/*.cpp"
)

add_executable(new_vers 
    ${NEWMAIN_SRC}
    ${RVIO2_CORE_SRC}
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
target_include_directories(new_vers PRIVATE 
    include
    "${CMAKE_SOURCE_DIR}/thirds/realsense/include" 
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/thirds/Eigen/eigen-3.4.0/"
    "${CMAKE_SOURCE_DIR}/src/rvio2"
    "${CMAKE_SOURCE_DIR}/src/rvio2/util"
    thirds/imgui
    thirds/imgui/backends
    thirds/implot
)
target_link_libraries(new_vers PRIVATE 
    opencv_world 
    matplotlib_cpp
    "${CMAKE_SOURCE_DIR}/thirds/realsense/lib/x64/realsense2.lib"
    "${CMAKE_SOURCE_DIR}/thirds/seconds/Project_commons.lib"
    d3d11
    dxgi
    d3dcompiler
)
if(MSVC)
    target_compile_options(new_vers PRIVATE /bigobj)
endif()
