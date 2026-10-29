#pragma once
#include "../../ivcard.h"
#include "../../../common/vgpu_format.h"
#include <vector>
#include <map>
#include <memory>

namespace vgpu::generic {

class GenericCard : public IVideoCard {
public:
    explicit GenericCard(const VgpuDefinition& def);
    ~GenericCard() override;

    void reset() override;
    uint32_t reg_read(uint32_t addr) override;
    void reg_write(uint32_t addr, uint32_t value) override;
    void dump_vram(uint32_t offset, uint32_t len, uint8_t* out) const override;
    void upload_vram(uint32_t offset, uint32_t len, const uint8_t* data) override;
    uint64_t get_exec_time_us() const override { return total_exec_time_us; }
    bool has_pending_irq() const override { return !pending_irqs.empty(); }
    uint32_t pop_pending_irq() override;
    uint32_t pci_config_read(uint32_t reg) override;

    void attach_shared(uint8_t* vram_ptr);

private:
    void execute_operation(const OperationDef& op, uint32_t trigger_value);
    void perform_fill(const OperationDef& op);
    void perform_copy(const OperationDef& op);
    uint32_t read_param(const std::string& name);
    void capture_latches(uint32_t written_addr, uint32_t value);

    VgpuDefinition def;
    uint8_t* vram = nullptr;
    bool vram_owned = true;
    std::vector<uint32_t> regs;
    std::vector<uint32_t> latched;

    std::map<std::string, uint32_t> latch_state;

    uint32_t crtc_offset, crtc_pitch, crtc_h_total, crtc_v_total;
    uint32_t gen_int_cntl, gen_int_status;

    std::vector<uint32_t> pending_irqs;
    uint64_t total_exec_time_us = 0;

    uint32_t pci_config[64] = {0};
};

}
