#pragma once
#include "proto.h"
#include <cstdint>
#include <vector>

namespace rage128 {

struct EmulatorCore {
    std::vector<uint8_t> vram;
    std::vector<uint32_t> regs;

    EmulatorCore();

    void reset();
    uint32_t reg_read(uint32_t addr);
    void reg_write(uint32_t addr, uint32_t value);
    void dump_vram(uint32_t offset, uint32_t len, uint8_t* out) const;
};

}