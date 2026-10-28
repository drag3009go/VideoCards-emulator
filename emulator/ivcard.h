#pragma once
#include "proto.h"
#include <cstdint>

namespace vgpu {

class IVideoCard {
public:
    virtual ~IVideoCard() = default;
    virtual void reset() = 0;
    virtual uint32_t reg_read(uint32_t addr) = 0;
    virtual void reg_write(uint32_t addr, uint32_t value) = 0;
    virtual void dump_vram(uint32_t offset, uint32_t len, uint8_t* out) const = 0;
    virtual void upload_vram(uint32_t offset, uint32_t len, const uint8_t* data) = 0;
    virtual uint64_t get_exec_time_us() const = 0;
    virtual bool has_pending_irq() const = 0;
    virtual uint32_t pop_pending_irq() = 0;
    virtual uint32_t pci_config_read(uint32_t reg) = 0;
};

}