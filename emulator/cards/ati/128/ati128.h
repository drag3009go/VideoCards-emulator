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

private:
    void perform_fill(int start_x, int start_y);
    void perform_copy(int start_x, int start_y);
    void process_cce();

    std::vector<uint8_t> vram;
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
};

}