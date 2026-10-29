cmake_minimum_required(VERSION 3.20)
project(vgpu CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(emulator
    emulator/main.cpp
    emulator/qemu_proxy.cpp
    emulator/cards/ati/128/ati128.cpp
    emulator/cards/generic/generic_card.cpp
    common/vgpu_format.cpp
)
target_include_directories(emulator PRIVATE common emulator)
