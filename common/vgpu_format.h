#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace vgpu {

struct BarDef {
    int index;
    std::string type;   // "vram" | "mmio"
    uint64_t size;
    bool prefetch;
};

struct PciDef {
    uint16_t vendor_id;
    uint16_t device_id;
    uint32_t class_code;
    uint8_t  revision;
    uint16_t subsystem_vendor_id;
    uint16_t subsystem_id;
    std::vector<BarDef> bars;
};

struct LatchRule {
    std::string on_write;
    std::string when;                  // выражение
    std::vector<std::string> capture;  // какие регистры защёлкивать
};

struct OperationDef {
    std::string type;                  // "fill" | "copy"
    std::string trigger;               // имя регистра
    std::string condition;             // выражение
    std::map<std::string, std::string> params;
    std::map<std::string, std::string> clip;
};

struct VgpuDefinition {
    int format_version = 1;
    std::string name;
    std::string author;
    std::string description;

    PciDef pci;
    uint64_t vram_size = 0;
    uint64_t mmio_size = 0;

    uint32_t default_h_total = 640;
    uint32_t default_v_total = 480;
    uint32_t default_offset  = 0;
    uint32_t default_pitch   = 2560;

    std::map<std::string, uint32_t> registers;
    std::vector<LatchRule> latches;
    std::vector<OperationDef> operations;

    uint32_t irq_idle_bit = 1;
    bool irq_on_operation_done = true;

    std::string cursor_offset_reg;
    std::string cursor_pos_reg;
    std::string cursor_enable_reg;
    int cursor_size = 32;

    static VgpuDefinition load(const std::string& path);
    static VgpuDefinition from_string(const std::string& json_str);

    uint32_t reg_addr(const std::string& name) const;
};

}