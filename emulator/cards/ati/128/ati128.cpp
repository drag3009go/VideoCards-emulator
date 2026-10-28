#include "ati128.h"
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <chrono>

namespace vgpu::ati128 {

constexpr uint32_t REG_CRTC_OFFSET         = 0x0024;
constexpr uint32_t REG_CRTC_PITCH          = 0x0028;
constexpr uint32_t REG_CRTC_H_TOTAL_DISP   = 0x0050;
constexpr uint32_t REG_CRTC_V_TOTAL_DISP   = 0x0054;
constexpr uint32_t REG_PM4_BUFFER_CNTL     = 0x07B0;
constexpr uint32_t REG_PM4_BUFFER_ADDR     = 0x07B4;
constexpr uint32_t REG_GEN_INT_CNTL        = 0x0100;
constexpr uint32_t REG_GEN_INT_STATUS      = 0x0104;
constexpr uint32_t REG_GEN_INT_ACK         = 0x0108;

constexpr uint32_t REG_SRC_OFFSET        = 0x1400;
constexpr uint32_t REG_DST_OFFSET        = 0x1404;
constexpr uint32_t REG_DST_PITCH         = 0x1408;
constexpr uint32_t REG_SRC_PITCH         = 0x140C;
constexpr uint32_t REG_DP_MIX            = 0x1418;
constexpr uint32_t REG_DST_Y_X           = 0x1438;
constexpr uint32_t REG_SC_LEFT           = 0x1440;
constexpr uint32_t REG_SC_RIGHT          = 0x1444;
constexpr uint32_t REG_SC_TOP            = 0x1448;
constexpr uint32_t REG_SC_BOTTOM         = 0x144C;
constexpr uint32_t REG_DP_GUI_MASTER_CNTL= 0x146c;
constexpr uint32_t REG_CUR_OFFSET        = 0x1470;
constexpr uint32_t REG_CUR_POS           = 0x1474;
constexpr uint32_t REG_DP_BRUSH_FRGD_CLR = 0x147c;
constexpr uint32_t REG_CUR_CNTL          = 0x1480;
constexpr uint32_t REG_DST_WIDTH_X       = 0x1588;
constexpr uint32_t REG_DST_HEIGHT_Y      = 0x15a0;

constexpr uint32_t IRQ_GUI_IDLE = 0x01;

ATI128Card::ATI128Card()
    : vram(VRAM_SIZE, 0), regs(MMIO_REGS / 4, 0) {}

void ATI128Card::reset() {
    std::fill(vram.begin(), vram.end(), 0);
    std::fill(regs.begin(), regs.end(), 0);
    latch_src_offset = 0;
    latch_src_pitch = 0;
    latch_dst_offset = 0;
    latch_dst_pitch = 0;
    latch_brush_color = 0;
    latch_width = 0;
    latch_height = 0;
    rop3_code = 0xCC;
    sc_left = 0;
    sc_right = 0xFFFF;
    sc_top = 0;
    sc_bottom = 0xFFFF;
    dp_gui_master_cntl = 0;
    engine_ready = false;
    cur_offset = 0;
    cur_pos_x = 0;
    cur_pos_y = 0;
    cur_enabled = false;
    crtc_h_total_disp = 640 - 1;
    crtc_v_total_disp = 480 - 1;
    crtc_offset = 0;
    crtc_pitch = 640 * 4;
    cce_buf_addr = 0;
    cce_buf_size = 0;
    gen_int_cntl = 0;
    gen_int_status = 0;
    pending_irqs.clear();
    total_exec_time_us = 0;
}

uint32_t ATI128Card::pop_pending_irq() {
    if (pending_irqs.empty()) return 0;
    uint32_t v = pending_irqs.front();
    pending_irqs.erase(pending_irqs.begin());
    return v;
}

void ATI128Card::trigger_irq(uint32_t bit) {
    gen_int_status |= bit;
    if (gen_int_cntl & bit) {
        pending_irqs.push_back(gen_int_status);
    }
}

uint32_t ATI128Card::reg_read(uint32_t addr) {
    if (addr + 4 > MMIO_REGS) return 0xFFFFFFFFu;
    if (addr == REG_GEN_INT_STATUS) return gen_int_status;
    return regs[addr / 4];
}

void ATI128Card::reg_write(uint32_t addr, uint32_t value) {
    if (addr + 4 > MMIO_REGS) return;
    regs[addr / 4] = value;

    if (addr == REG_GEN_INT_CNTL) gen_int_cntl = value;
    else if (addr == REG_GEN_INT_ACK) gen_int_status &= ~value;
    else if (addr == REG_CRTC_OFFSET) crtc_offset = value;
    else if (addr == REG_CRTC_PITCH) crtc_pitch = value;
    else if (addr == REG_CRTC_H_TOTAL_DISP) crtc_h_total_disp = value & 0xFFFF;
    else if (addr == REG_CRTC_V_TOTAL_DISP) crtc_v_total_disp = value & 0xFFFF;
    else if (addr == REG_PM4_BUFFER_ADDR) cce_buf_addr = value;
    else if (addr == REG_PM4_BUFFER_CNTL) {
        cce_buf_size = value & 0xFFFF;
        if (value & 0x80000000) process_cce();
    }
    else if (addr == REG_DP_BRUSH_FRGD_CLR) latch_brush_color = value;
    else if (addr == REG_DP_MIX) rop3_code = value & 0xFF;
    else if (addr == REG_SC_LEFT) sc_left = value;
    else if (addr == REG_SC_RIGHT) sc_right = value;
    else if (addr == REG_SC_TOP) sc_top = value;
    else if (addr == REG_SC_BOTTOM) sc_bottom = value;
    else if (addr == REG_CUR_OFFSET) cur_offset = value;
    else if (addr == REG_CUR_POS) {
        cur_pos_x = value & 0xFFFF;
        cur_pos_y = (value >> 16) & 0xFFFF;
    }
    else if (addr == REG_CUR_CNTL) cur_enabled = (value & 0x01) != 0;
    else if (addr == REG_DP_GUI_MASTER_CNTL) {
        dp_gui_master_cntl = value;
        if (value & 0x80000000) engine_ready = true;
        if ((value & 0x00000002) == 0) {
            latch_dst_offset = regs[REG_DST_OFFSET / 4];
            latch_dst_pitch  = regs[REG_DST_PITCH / 4];
            latch_src_offset = regs[REG_SRC_OFFSET / 4];
            latch_src_pitch  = regs[REG_SRC_PITCH / 4];
        }
    }
    else if (addr == REG_DST_WIDTH_X) latch_width = value;
    else if (addr == REG_DST_HEIGHT_Y) latch_height = value;
    else if (addr == REG_DST_Y_X) {
        if (engine_ready && latch_dst_pitch > 0) {
            int start_y = (value >> 16) & 0xFFFF;
            int start_x = value & 0xFFFF;
            if (dp_gui_master_cntl & 0x00010000) perform_copy(start_x, start_y);
            else perform_fill(start_x, start_y);
            trigger_irq(IRQ_GUI_IDLE);
        }
    }
}

void ATI128Card::process_cce() {
    printf("[EMU] CCE: Processing command buffer at 0x%08X\n", cce_buf_addr);
    uint32_t ptr = cce_buf_addr;
    for (int i = 0; i < 10; ++i) {
        if (ptr + 4 > VRAM_SIZE) break;
        uint32_t header = *reinterpret_cast<uint32_t*>(&vram[ptr]);
        if (header == 0x00000000) break;
        if ((header >> 30) == 3) {
            uint32_t count = (header >> 16) & 0x3FFF;
            uint32_t reg_offset = header & 0xFFFF;
            ptr += 4;
            for (uint32_t j = 0; j < count; ++j) {
                if (ptr + 4 > VRAM_SIZE) break;
                uint32_t val = *reinterpret_cast<uint32_t*>(&vram[ptr]);
                reg_write(reg_offset + (j * 4), val);
                ptr += 4;
            }
        } else break;
    }
    printf("[EMU] CCE: Finished processing buffer.\n");
}

void ATI128Card::perform_fill(int start_x, int start_y) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (uint32_t y = 0; y < latch_height; ++y) {
        int abs_y = start_y + y;
        if (abs_y < (int)sc_top || abs_y >= (int)sc_bottom) continue;
        uint32_t row_start = latch_dst_offset + (y * latch_dst_pitch);
        for (uint32_t x = 0; x < latch_width; ++x) {
            int abs_x = start_x + x;
            if (abs_x < (int)sc_left || abs_x >= (int)sc_right) continue;
            uint32_t pixel_addr = row_start + (x * 4);
            if (pixel_addr + 4 <= VRAM_SIZE) {
                uint32_t dst_px = *reinterpret_cast<uint32_t*>(&vram[pixel_addr]);
                uint32_t res = latch_brush_color;
                if (rop3_code == 0x66) res = latch_brush_color ^ dst_px;
                *reinterpret_cast<uint32_t*>(&vram[pixel_addr]) = res;
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    total_exec_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

void ATI128Card::perform_copy(int start_x, int start_y) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (uint32_t y = 0; y < latch_height; ++y) {
        int abs_y = start_y + y;
        if (abs_y < (int)sc_top || abs_y >= (int)sc_bottom) continue;
        uint32_t src_row = latch_src_offset + (y * latch_src_pitch);
        uint32_t dst_row = latch_dst_offset + (y * latch_dst_pitch);
        for (uint32_t x = 0; x < latch_width; ++x) {
            int abs_x = start_x + x;
            if (abs_x < (int)sc_left || abs_x >= (int)sc_right) continue;
            uint32_t src_px = src_row + (x * 4);
            uint32_t dst_px = dst_row + (x * 4);
            if (src_px + 4 <= VRAM_SIZE && dst_px + 4 <= VRAM_SIZE) {
                uint32_t src_val = *reinterpret_cast<uint32_t*>(&vram[src_px]);
                uint32_t dst_val = *reinterpret_cast<uint32_t*>(&vram[dst_px]);
                uint32_t res = src_val;
                if (rop3_code == 0x66) res = src_val ^ dst_val;
                *reinterpret_cast<uint32_t*>(&vram[dst_px]) = res;
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    total_exec_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

void ATI128Card::dump_vram(uint32_t offset, uint32_t len, uint8_t* out) const {
    if (offset >= VRAM_SIZE) return;
    if ((size_t)offset + len > VRAM_SIZE) len = (uint32_t)(VRAM_SIZE - offset);
    std::memcpy(out, vram.data() + offset, len);
    if (cur_enabled) {
        for (int cy = 0; cy < 32; ++cy) {
            for (int cx = 0; cx < 32; ++cx) {
                uint32_t screen_x = cur_pos_x + cx;
                uint32_t screen_y = cur_pos_y + cy;
                uint32_t pixel_addr = (crtc_offset + (screen_y * crtc_pitch)) + (screen_x * 4);
                if (pixel_addr >= offset && pixel_addr + 4 <= offset + len) {
                    uint32_t cur_pixel_addr = cur_offset + (cy * 32 * 4) + (cx * 4);
                    if (cur_pixel_addr + 4 <= VRAM_SIZE) {
                        uint32_t cur_color = *reinterpret_cast<const uint32_t*>(&vram[cur_pixel_addr]);
                        if (cur_color != 0x00000000) {
                            uint32_t out_idx = pixel_addr - offset;
                            std::memcpy(out + out_idx, &cur_color, 4);
                        }
                    }
                }
            }
        }
    }
}

void ATI128Card::upload_vram(uint32_t offset, uint32_t len, const uint8_t* data) {
    if (offset + len <= VRAM_SIZE) {
        std::memcpy(vram.data() + offset, data, len);
    }
}

}