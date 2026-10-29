#include "vgpu_format.h"
#include "../third_party/nlohmann/json.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace vgpu {

uint32_t VgpuDefinition::reg_addr(const std::string& name) const {
    auto it = registers.find(name);
    if (it == registers.end())
        throw std::runtime_error("unknown register: " + name);
    return it->second;
}

VgpuDefinition VgpuDefinition::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open vgpu file: " + path);
    std::stringstream ss; ss << f.rdbuf();
    return from_string(ss.str());
}

VgpuDefinition VgpuDefinition::from_string(const std::string& s) {
    json j = json::parse(s);
    VgpuDefinition def;

    def.format_version = j.value("format_version", 1);
    def.name = j.value("name", "Unknown");
    def.author = j.value("author", "");
    def.description = j.value("description", "");

    auto& pci = j["pci"];
    def.pci.vendor_id = pci.value("vendor_id", 0x1002);
    def.pci.device_id = pci.value("device_id", 0x5246);
    def.pci.class_code = pci.value("class_code", 0x030000);
    def.pci.revision = pci.value("revision", 0);
    def.pci.subsystem_vendor_id = pci.value("subsystem_vendor_id", def.pci.vendor_id);
    def.pci.subsystem_id = pci.value("subsystem_id", 0);

    for (auto& b : pci["bars"]) {
        BarDef bar;
        bar.index = b.value("index", 0);
        bar.type = b.value("type", "mmio");
        bar.size = b.value("size", 0);
        bar.prefetch = b.value("prefetch", false);
        def.pci.bars.push_back(bar);
    }

    def.vram_size = j.value("vram_size", 0);
    def.mmio_size = j.value("mmio_size", 0);

    if (j.contains("default_crtc")) {
        auto& d = j["default_crtc"];
        def.default_h_total = d.value("h_total_disp", 640);
        def.default_v_total = d.value("v_total_disp", 480);
        def.default_offset  = d.value("offset", 0);
        def.default_pitch   = d.value("pitch", 2560);
    }

    for (auto& [k, v] : j["registers"].items())
        def.registers[k] = v.get<uint32_t>();

    if (j.contains("latch")) {
        for (auto& l : j["latch"]) {
            LatchRule r;
            r.on_write = l["on_write"];
            r.when     = l["when"];
            for (auto& c : l["capture"]) r.capture.push_back(c.get<std::string>());
            def.latches.push_back(r);
        }
    }

    if (j.contains("operations")) {
        for (auto& o : j["operations"]) {
            OperationDef op;
            op.type = o["type"];
            op.trigger = o["trigger"];
            op.condition = o.value("condition", "");
            for (auto& [k, v] : o["params"].items()) op.params[k] = v.get<std::string>();
            if (o.contains("clip"))
                for (auto& [k, v] : o["clip"].items()) op.clip[k] = v.get<std::string>();
            def.operations.push_back(op);
        }
    }

    if (j.contains("irq")) {
        auto& i = j["irq"];
        def.irq_idle_bit = i.value("idle_bit", 1);
        def.irq_on_operation_done = i.value("on_operation_done", true);
    }

    if (j.contains("cursor")) {
        auto& c = j["cursor"];
        def.cursor_offset_reg = c.value("offset_reg", "");
        def.cursor_pos_reg    = c.value("pos_reg", "");
        def.cursor_enable_reg = c.value("enable_reg", "");
        def.cursor_size       = c.value("size", 32);
    }

    return def;
}

}