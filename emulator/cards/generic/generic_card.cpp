#include "generic_card.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <chrono>

namespace vgpu::generic {

GenericCard::GenericCard(const VgpuDefinition& d)
    : def(d), regs(d.mmio_size / 4, 0), latched(d.mmio_size / 4, 0) {
    vram_owned = true;
    vram = (uint8_t*)std::malloc(d.vram_size);
    std::memset(vram, 0, d.vram_size);

    crtc_offset  = d.default_offset;
    crtc_pitch   = d.default_pitch;
    crtc_h_total = d.default_h_total;
    crtc_v_total = d.default_v_total;

    // PCI config
    pci_config[0] = (uint32_t)d.pci.device_id << 16 | d.pci.vendor_id;
    pci_config[2] = d.pci.class_code;
    pci_config[0x2C / 4] = (uint32_t)d.pci.subsystem_id << 16 | d.pci.subsystem_vendor_id;

    // BAR-ы (QEMU распределит реальные адреса, тут для гостя-«base»)
    for (auto& bar : d.pci.bars) {
        if (bar.index == 0)      pci_config[0x10 / 4] = 0xE0000000;
        else if (bar.index == 2) pci_config[0x18 / 4] = 0xE1000000;
    }

    pci_config[0x3C / 4] = 0x00000100; // interrupt pin = 1 (INTA)
}

GenericCard::~GenericCard() {
    if (vram_owned && vram) std::free(vram);
}

void GenericCard::attach_shared(uint8_t* p) {
    if (vram_owned && vram) std::free(vram);
    vram = p;
    vram_owned = false;
}

void GenericCard::reset() {
    std::memset(vram, 0, def.vram_size);
    std::fill(regs.begin(), regs.end(), 0);
    std::fill(latched.begin(), latched.end(), 0);
    latch_state.clear();
    crtc_offset  = def.default_offset;
    crtc_pitch   = def.default_pitch;
    crtc_h_total = def.default_h_total;
    crtc_v_total = def.default_v_total;
    gen_int_cntl = 0;
    gen_int_status = 0;
    pending_irqs.clear();
    total_exec_time_us = 0;
}

uint32_t GenericCard::pci_config_read(uint32_t reg) {
    if (reg >= 64) return 0xFFFFFFFF;
    return pci_config[reg];
}

uint32_t GenericCard::reg_read(uint32_t addr) {
    if (addr + 4 > def.mmio_size) return 0xFFFFFFFF;
    // Специальные регистры CRTC
    if (addr == def.reg_addr("CRTC_OFFSET"))       return crtc_offset;
    if (addr == def.reg_addr("CRTC_PITCH"))        return crtc_pitch;
    if (addr == def.reg_addr("CRTC_H_TOTAL_DISP")) return crtc_h_total - 1;
    if (addr == def.reg_addr("CRTC_V_TOTAL_DISP")) return crtc_v_total - 1;
    return regs[addr / 4];
}

void GenericCard::reg_write(uint32_t addr, uint32_t value) {
    if (addr + 4 > def.mmio_size) return;
    regs[addr / 4] = value;

    // CRTC-регистры — обновляют поля
    if (addr == def.reg_addr("CRTC_OFFSET"))       crtc_offset = value;
    if (addr == def.reg_addr("CRTC_PITCH"))        crtc_pitch = value;
    if (addr == def.reg_addr("CRTC_H_TOTAL_DISP")) crtc_h_total = (value & 0xFFFF) + 1;
    if (addr == def.reg_addr("CRTC_V_TOTAL_DISP")) crtc_v_total = (value & 0xFFFF) + 1;

    // Latch: защёлкивание
    capture_latches(addr, value);

    // Operations: триггер?
    for (auto& op : def.operations) {
        uint32_t trig_addr = def.reg_addr(op.trigger);
        if (addr == trig_addr) {
            execute_operation(op, value);
        }
    }
}

void GenericCard::capture_latches(uint32_t written_addr, uint32_t value) {
    for (auto& rule : def.latches) {
        uint32_t rule_addr = def.reg_addr(rule.on_write);
        if (written_addr != rule_addr) continue;

        // Простой парсер условия: "(value & 0x2) == 0"
        // TODO: реализовать полноценный парсер. Пока — хардкод для & 0x2 == 0.
        bool cond_ok = true;
        if (rule.when == "(value & 0x2) == 0")
            cond_ok = (value & 0x2) == 0;

        if (!cond_ok) continue;

        for (auto& name : rule.capture) {
            uint32_t a = def.reg_addr(name);
            latch_state[name] = regs[a / 4];
        }
    }
}

uint32_t GenericCard::read_param(const std::string& name) {
    // Если параметр — имя регистра, возвращаем текущее значение (или latch)
    if (latch_state.count(name)) return latch_state[name];
    if (def.registers.count(name)) {
        uint32_t a = def.reg_addr(name);
        return regs[a / 4];
    }
    // Иначе — литерал hex ("0xFF0000") или decimal ("320")
    return (uint32_t)std::strtoul(name.c_str(), nullptr, 0);
}

void GenericCard::execute_operation(const OperationDef& op, uint32_t trigger_value) {
    // Проверка condition (TODO: общий парсер, пока хардкод)
    if (!op.condition.empty()) {
        uint32_t master = regs[def.reg_addr("DP_GUI_MASTER_CNTL") / 4];
        if (op.condition.find("& 0x80000000") != std::string::npos) {
            if ((master & 0x80000000) == 0) return;
        }
        if (op.condition.find("& 0x00010000") != std::string::npos) {
            if ((master & 0x00010000) == 0) return;
        }
    }

    if (op.type == "fill") perform_fill(op);
    else if (op.type == "copy") perform_copy(op);

    // IRQ
    if (def.irq_on_operation_done) {
        gen_int_status |= def.irq_idle_bit;
        if (gen_int_cntl & def.irq_idle_bit)
            pending_irqs.push_back(gen_int_status);
    }
}

void GenericCard::perform_fill(const OperationDef& op) {
    auto t1 = std::chrono::high_resolution_clock::now();

    uint32_t dst_off = read_param(op.params.at("dst_offset"));
    uint32_t dst_pitch = read_param(op.params.at("dst_pitch"));
    uint32_t w = read_param(op.params.at("width"));
    uint32_t h = read_param(op.params.at("height"));
    uint32_t color = read_param(op.params.at("color"));
    uint32_t rop = read_param(op.params.at("rop"));

    for (uint32_t y = 0; y < h; ++y) {
        uint32_t row = dst_off + y * dst_pitch;
        for (uint32_t x = 0; x < w; ++x) {
            uint32_t px = row + x * 4;
            if (px + 4 > def.vram_size) continue;
            uint32_t old = *reinterpret_cast<uint32_t*>(&vram[px]);
            uint32_t res = color;
            if ((rop & 0xFF) == 0x66) res = color ^ old;
            *reinterpret_cast<uint32_t*>(&vram[px]) = res;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    total_exec_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

void GenericCard::perform_copy(const OperationDef& op) {
    auto t1 = std::chrono::high_resolution_clock::now();

    uint32_t so = read_param(op.params.at("src_offset"));
    uint32_t sp = read_param(op.params.at("src_pitch"));
    uint32_t d_off = read_param(op.params.at("dst_offset"));
    uint32_t dp = read_param(op.params.at("dst_pitch"));
    uint32_t w = read_param(op.params.at("width"));
    uint32_t h = read_param(op.params.at("height"));
    uint32_t rop = read_param(op.params.at("rop"));

    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            uint32_t s = so + y * sp + x * 4;
            uint32_t d = d_off + y * dp + x * 4;
            if (s + 4 > def.vram_size || d + 4 > def.vram_size) continue;
            uint32_t sv = *reinterpret_cast<uint32_t*>(&vram[s]);
            uint32_t dv = *reinterpret_cast<uint32_t*>(&vram[d]);
            uint32_t res = sv;
            if ((rop & 0xFF) == 0x66) res = sv ^ dv;
            *reinterpret_cast<uint32_t*>(&vram[d]) = res;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    total_exec_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

void GenericCard::dump_vram(uint32_t offset, uint32_t len, uint8_t* out) const {
    if (offset >= def.vram_size) return;
    if ((size_t)offset + len > def.vram_size) len = (uint32_t)(def.vram_size - offset);
    std::memcpy(out, vram + offset, len);
}

void GenericCard::upload_vram(uint32_t offset, uint32_t len, const uint8_t* data) {
    if (offset + len <= def.vram_size)
        std::memcpy(vram + offset, data, len);
}

uint32_t GenericCard::pop_pending_irq() {
    if (pending_irqs.empty()) return 0;
    uint32_t v = pending_irqs.front();
    pending_irqs.erase(pending_irqs.begin());
    return v;
}

}