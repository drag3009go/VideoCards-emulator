cmake_minimum_required(VERSION 3.20)
project(vgpu_stand CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.9
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw)

find_package(OpenGL REQUIRED)

add_executable(emulator
    emulator/main.cpp
    emulator/qemu_proxy.cpp
    emulator/cards/ati/128/ati128.cpp
)
target_include_directories(emulator PRIVATE common emulator)
if(WIN32)
    target_link_libraries(emulator PRIVATE ws2_32)
endif()

add_executable(tester
    tester/main.cpp
)
target_include_directories(tester PRIVATE common third_party/OpenCL)
if(WIN32)
    target_link_libraries(tester PRIVATE imgui_lib OpenGL::GL glfw ws2_32 "C:/Windows/System32/OpenCL.dll")
else()
    target_link_libraries(tester PRIVATE imgui_lib OpenGL::GL glfw OpenCL)
endif()