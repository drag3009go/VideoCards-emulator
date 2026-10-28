#pragma once
#include "../../../ivcard.h"
#include <vector>
#include <cstdint>

namespace vgpu::ati128 {

class ATI128Card : public IVideoCard {
public:
    ATI128Card();
    virtual ~ATI128Card() = default;

    void reset() override;
    uint32_t reg_read(uint32_t addr) override;
    void reg_write(uint32_t addr, uint32_t value) override;
    void dump_vram(uint32_t offset, uint32_t len, uint8_t* out) const override;
    void upload_vram(uint32_t offset, uint32_t len, const uint8_t* data) override;
    uint64_t get_exec_time_us() const override { return total_exec_time_us; }
    bool has_pending_irq() const override { return !pending_irqs.empty(); }
    uint32_t pop_pending_irq() override;
    uint32_t pci_config_read(uint32_t reg) override;

    void set_realistic_timing(bool enable) { realistic_timing = enable; }

private:
    void perform_fill(int start_x, int start_y);
    void perform_copy(int start_x, int start_y);
    void process_cce();
    void trigger_irq(uint32_t bit);
    void rasterize_triangle(uint32_t v0, uint32_t v1, uint32_t v2);
    void write_palette(uint32_t index, uint32_t rgb);

    std::vector<uint8_t> vram;
    std::vector<float> zbuffer;
    std::vector<uint32_t> regs;

    uint32_t latch_src_offset = 0;
    uint32_t latch_src_pitch = 0;
    uint32_t latch_dst_offset = 0;
    uint32_t latch_dst_pitch = 0;
    uint32_t latch_brush_color = 0;
    uint32_t latch_width = 0;
    uint32_t latch_height = 0;
    uint32_t rop3_code = 0xCC;

    uint32_t sc_left = 0;
    uint32_t sc_right = 0xFFFF;
    uint32_t sc_top = 0;
    uint32_t sc_bottom = 0xFFFF;

    uint32_t dp_gui_master_cntl = 0;
    bool engine_ready = false;

    uint32_t cur_offset = 0;
    uint32_t cur_pos_x = 0;
    uint32_t cur_pos_y = 0;
    bool cur_enabled = false;

    uint32_t crtc_h_total_disp = 640 - 1;
    uint32_t crtc_v_total_disp = 480 - 1;
    uint32_t crtc_offset = 0;
    uint32_t crtc_pitch = 640 * 4;

    uint32_t cce_buf_addr = 0;
    uint32_t cce_buf_size = 0;

    uint32_t gen_int_cntl = 0;
    uint32_t gen_int_status = 0;

    uint32_t pci_config[64] = {0};
    std::vector<uint8_t> vbios;

    uint8_t palette_index = 0;
    uint32_t palette_ram[256] = {0};

    uint32_t vc_prim_type = 0;
    uint32_t vc_cntl = 0;
    uint32_t vtx[3][3] = {{0}};
    int vtx_count = 0;

    uint32_t dma_major01 = 0;
    uint32_t dma_major12 = 0;
    uint32_t dma_major02 = 0;

    std::vector<uint32_t> pending_irqs;
    uint64_t total_exec_time_us = 0;
    bool realistic_timing = false;
};

}