#pragma once
#include "ivcard.h"
#include <memory>

namespace vgpu {
    int run_qemu_proxy(std::unique_ptr<IVideoCard> card, const char* host, int port);
}