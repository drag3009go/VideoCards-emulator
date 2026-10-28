#include "core.h"

namespace rage128 {

EmulatorCore::EmulatorCore()
    : vram(VRAM_SIZE, 0), regs(MMIO_REGS / 4, 0) {}

void EmulatorCore::reset() {
    std::fill(vram.begin(), vram.end(), 0);
    std::fill(regs.begin(), regs.end(), 0);
}

uint32_t EmulatorCore::reg_read(uint32_t addr) {
    if (addr + 4 > MMIO_REGS) return 0xFFFFFFFFu;
    return regs[addr / 4];
}

void EmulatorCore::reg_write(uint32_t addr, uint32_t value) {
    if (addr + 4 > MMIO_REGS) return;
    regs[addr / 4] = value;
}

void EmulatorCore::dump_vram(uint32_t offset, uint32_t len, uint8_t* out) const {
    if (offset >= VRAM_SIZE) return;
    if ((size_t)offset + len > VRAM_SIZE) len = (uint32_t)(VRAM_SIZE - offset);
    std::memcpy(out, vram.data() + offset, len);
}

}