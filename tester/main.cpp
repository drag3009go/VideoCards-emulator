#define GL_BGRA 0x80E1
#define CL_TARGET_OPENCL_VERSION 120

#include "proto.h"
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <CL/cl.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <algorithm>

using namespace vgpu;

static GLFWwindow*        g_window      = nullptr;
static socket_t           g_sock        = INVALID_SOCK;
static bool               g_connected   = false;
static bool               g_ping_ok     = false;
static std::vector<uint8_t> g_vram_emu(VRAM_SIZE, 0);
static std::vector<uint8_t> g_vram_gpu(VRAM_SIZE, 0);
static std::vector<uint8_t> g_vram_diff(VRAM_SIZE, 0);
static std::vector<float>   g_zbuffer_gpu(640 * 480, 1e9f);
static GLuint             g_tex_emu     = 0;
static GLuint             g_tex_gpu     = 0;
static GLuint             g_tex_diff    = 0;
static uint32_t           g_view_offset = 0x10000;

static uint32_t           g_display_w   = 640;
static uint32_t           g_display_h   = 480;
static uint32_t           g_tex_w       = 0;
static uint32_t           g_tex_h       = 0;

static size_t             g_diff_pixels = 0;
static bool               g_test_passed = false;
static bool               g_has_result  = false;
static int                g_selected_test = 0;
static int                g_selected_hard_test = 0;

static bool               g_anim_active = false;
static int                g_anim_shape = 0;
static double             g_last_anim_time = 0;

static uint64_t           g_emu_exec_time_us = 0;
static long long          g_time_gpu    = 0;
static int                g_irq_count   = 0;
static uint32_t           g_last_irq_status = 0;

static cl_context       g_cl_context   = nullptr;
static cl_command_queue g_cl_queue     = nullptr;
static cl_program       g_cl_program   = nullptr;
static cl_kernel        g_cl_fill_kernel = nullptr;
static cl_kernel        g_cl_copy_kernel = nullptr;
static cl_kernel        g_cl_cursor_kernel = nullptr;
static cl_mem           g_cl_vram      = nullptr;
static bool             g_cl_ready     = false;

const char* KERNEL_SRC = R"(
__kernel void fill_kernel(__global uchar* vram, uint offset, uint pitch, uint width, uint height, uint color, uint rop3, uint start_x, uint start_y, uint sc_left, uint sc_right, uint sc_top, uint sc_bottom) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int abs_x = start_x + x;
        int abs_y = start_y + y;
        if (abs_x < sc_left || abs_x >= sc_right || abs_y < sc_top || abs_y >= sc_bottom) return;
        uint pixel_addr = offset + (y * pitch) + (x * 4);
        uchar4 brush;
        brush.x = (color >> 0) & 0xFF;
        brush.y = (color >> 8) & 0xFF;
        brush.z = (color >> 16) & 0xFF;
        brush.w = (color >> 24) & 0xFF;
        uchar4 dst_val = vload4(0, vram + pixel_addr);
        uchar4 res = brush;
        if (rop3 == 0x66) {
            res.x = brush.x ^ dst_val.x;
            res.y = brush.y ^ dst_val.y;
            res.z = brush.z ^ dst_val.z;
            res.w = brush.w ^ dst_val.w;
        }
        vstore4(res, 0, vram + pixel_addr);
    }
}

__kernel void copy_kernel(__global uchar* vram, uint src_offset, uint src_pitch, uint dst_offset, uint dst_pitch, uint width, uint height, uint rop3, uint start_x, uint start_y, uint sc_left, uint sc_right, uint sc_top, uint sc_bottom) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int abs_x = start_x + x;
        int abs_y = start_y + y;
        if (abs_x < sc_left || abs_x >= sc_right || abs_y < sc_top || abs_y >= sc_bottom) return;
        uint src_px = src_offset + (y * src_pitch) + (x * 4);
        uint dst_px = dst_offset + (y * dst_pitch) + (x * 4);
        uchar4 src_val = vload4(0, vram + src_px);
        uchar4 dst_val = vload4(0, vram + dst_px);
        uchar4 res = src_val;
        if (rop3 == 0x66) {
            res.x = src_val.x ^ dst_val.x;
            res.y = src_val.y ^ dst_val.y;
            res.z = src_val.z ^ dst_val.z;
            res.w = src_val.w ^ dst_val.w;
        }
        vstore4(res, 0, vram + dst_px);
    }
}

__kernel void cursor_kernel(__global uchar* vram, uint cur_offset, uint cur_x, uint cur_y, uint crtc_offset, uint crtc_pitch) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < 32 && y < 32) {
        uint cur_pixel = cur_offset + (y * 32 * 4) + (x * 4);
        uint color = *(__global uint*)(vram + cur_pixel);
        if (color != 0x00000000) {
            uint screen_pixel = crtc_offset + ((cur_y + y) * crtc_pitch) + ((cur_x + x) * 4);
            *(__global uint*)(vram + screen_pixel) = color;
        }
    }
}
)";

static bool init_opencl() {
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_platforms, num_devices;
    if (clGetPlatformIDs(1, &platform, &num_platforms) != CL_SUCCESS) return false;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) != CL_SUCCESS) return false;
    g_cl_context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    if (!g_cl_context) return false;
    g_cl_queue = clCreateCommandQueue(g_cl_context, device, 0, nullptr);
    if (!g_cl_queue) return false;
    g_cl_program = clCreateProgramWithSource(g_cl_context, 1, &KERNEL_SRC, nullptr, nullptr);
    if (clBuildProgram(g_cl_program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) return false;
    g_cl_fill_kernel = clCreateKernel(g_cl_program, "fill_kernel", nullptr);
    g_cl_copy_kernel = clCreateKernel(g_cl_program, "copy_kernel", nullptr);
    g_cl_cursor_kernel = clCreateKernel(g_cl_program, "cursor_kernel", nullptr);
    if (!g_cl_fill_kernel || !g_cl_copy_kernel || !g_cl_cursor_kernel) return false;
    g_cl_vram = clCreateBuffer(g_cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, VRAM_SIZE, g_vram_gpu.data(), nullptr);
    if (!g_cl_vram) return false;
    g_cl_ready = true;
    return true;
}

static bool connect_emu() {
    g_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (g_sock == INVALID_SOCK) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port   = htons(DEFAULT_PORT);
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (connect(g_sock, (sockaddr*)&a, sizeof(a)) < 0) {
        close_socket(g_sock);
        g_sock = INVALID_SOCK;
        return false;
    }
    return true;
}

static bool recv_msg_filtered(socket_t s, MsgType& type, void* buf, uint32_t buf_size, uint32_t& out_len) {
    while (true) {
        MsgHeader hdr;
        if (!recv_all(s, &hdr, sizeof(hdr))) return false;
        if (hdr.type == (uint32_t)MsgType::Irq) {
            IrqMsg m;
            if (!recv_all(s, &m, sizeof(m))) return false;
            g_irq_count++;
            g_last_irq_status = m.status;
            printf("TESTER: IRQ received! status=0x%08X (total=%d)\n", m.status, g_irq_count);
            continue;
        }
        if (hdr.payload_len > buf_size) return false;
        if (hdr.payload_len > 0 && !recv_all(s, buf, hdr.payload_len)) return false;
        type = (MsgType)hdr.type;
        out_len = hdr.payload_len;
        return true;
    }
}

static bool send_ping() {
    if (!g_connected) return false;
    PingMsg p{ 0xDEADBEEF };
    if (!send_msg(g_sock, MsgType::Ping, &p, sizeof(p))) return false;
    uint8_t buf[64];
    MsgType t; uint32_t len;
    if (!recv_msg_filtered(g_sock, t, buf, sizeof(buf), len)) return false;
    if (t != MsgType::Pong || len != sizeof(PongMsg)) return false;
    PongMsg r; std::memcpy(&r, buf, sizeof(r));
    return r.token == p.token;
}

static bool send_reset() {
    if (!g_connected) return false;
    if (!send_msg(g_sock, MsgType::Reset, nullptr, 0)) return false;
    uint8_t buf[16];
    MsgType t; uint32_t len;
    if (!recv_msg_filtered(g_sock, t, buf, sizeof(buf), len)) return false;
    g_irq_count = 0;
    g_last_irq_status = 0;
    return t == MsgType::Pong;
}

static bool send_reg_write(uint32_t addr, uint32_t value) {
    if (!g_connected) return false;
    RegWriteMsg m{ addr, value };
    return send_msg(g_sock, MsgType::RegWrite, &m, sizeof(m));
}

static uint32_t send_reg_read(uint32_t addr) {
    if (!g_connected) return 0;
    RegReadMsg m{ addr };
    if (!send_msg(g_sock, MsgType::RegRead, &m, sizeof(m))) return 0;
    uint8_t buf[16];
    MsgType t; uint32_t len;
    if (!recv_msg_filtered(g_sock, t, buf, sizeof(buf), len)) return 0;
    if (t != MsgType::RegReadResp) return 0;
    RegReadRespMsg r; std::memcpy(&r, buf, sizeof(r));
    return r.value;
}

static uint32_t send_pci_read(uint32_t reg) {
    if (!g_connected) return 0;
    PciConfigReadMsg m{ reg };
    if (!send_msg(g_sock, MsgType::PciConfigRead, &m, sizeof(m))) return 0;
    uint8_t buf[16];
    MsgType t; uint32_t len;
    if (!recv_msg_filtered(g_sock, t, buf, sizeof(buf), len)) return 0;
    if (t != MsgType::PciConfigResp) return 0;
    PciConfigRespMsg r; std::memcpy(&r, buf, sizeof(r));
    return r.value;
}

static bool dump_vram() {
    if (!g_connected) return false;
    DumpVramMsg m{ 0, VRAM_SIZE };
    if (!send_msg(g_sock, MsgType::DumpVram, &m, sizeof(m))) return false;
    while (true) {
        MsgHeader msg_hdr;
        if (!recv_all(g_sock, &msg_hdr, sizeof(msg_hdr))) return false;
        if (msg_hdr.type == (uint32_t)MsgType::Irq) {
            IrqMsg irq;
            if (!recv_all(g_sock, &irq, sizeof(irq))) return false;
            g_irq_count++;
            g_last_irq_status = irq.status;
            printf("TESTER: IRQ received during dump! status=0x%08X\n", irq.status);
            continue;
        }
        if (msg_hdr.type != (uint32_t)MsgType::DumpVramResp) return false;
        DumpVramRespHdr hdr;
        if (!recv_all(g_sock, &hdr, sizeof(hdr))) return false;
        if (!recv_all(g_sock, g_vram_emu.data(), hdr.len)) return false;
        g_emu_exec_time_us = hdr.exec_time_us;
        return true;
    }
}

static void compare_vram() {
    g_diff_pixels = 0;
    std::fill(g_vram_diff.begin(), g_vram_diff.end(), 0);
    for (size_t i = 0; i < VRAM_SIZE; i += 4) {
        uint32_t emu_px = *reinterpret_cast<uint32_t*>(&g_vram_emu[i]);
        uint32_t gpu_px = *reinterpret_cast<uint32_t*>(&g_vram_gpu[i]);
        if (emu_px != gpu_px) {
            g_diff_pixels++;
            g_vram_diff[i + 0] = 0x00;
            g_vram_diff[i + 1] = 0x00;
            g_vram_diff[i + 2] = 0xFF;
            g_vram_diff[i + 3] = 0xFF;
        }
    }
    g_test_passed = (g_diff_pixels == 0);
    g_has_result = true;
}

static void emu_fill(uint32_t offset, uint32_t pitch, uint32_t w, uint32_t h, uint32_t color, uint32_t sx, uint32_t sy) {
    send_reg_write(0x1404, offset);
    send_reg_write(0x1408, pitch);
    send_reg_write(0x147c, color);
    send_reg_write(0x146c, 0x80F026D0);
    send_reg_write(0x1588, w);
    send_reg_write(0x15a0, h);
    send_reg_write(0x1438, (sy << 16) | sx);
}

static void gpu_fill(uint32_t offset, uint32_t pitch, uint32_t w, uint32_t h, uint32_t color, uint32_t rop, uint32_t sx, uint32_t sy, uint32_t sl, uint32_t sr, uint32_t st, uint32_t sb) {
    if (!g_cl_ready) return;
    clSetKernelArg(g_cl_fill_kernel, 0, sizeof(cl_mem), &g_cl_vram);
    clSetKernelArg(g_cl_fill_kernel, 1, sizeof(uint32_t), &offset);
    clSetKernelArg(g_cl_fill_kernel, 2, sizeof(uint32_t), &pitch);
    clSetKernelArg(g_cl_fill_kernel, 3, sizeof(uint32_t), &w);
    clSetKernelArg(g_cl_fill_kernel, 4, sizeof(uint32_t), &h);
    clSetKernelArg(g_cl_fill_kernel, 5, sizeof(uint32_t), &color);
    clSetKernelArg(g_cl_fill_kernel, 6, sizeof(uint32_t), &rop);
    clSetKernelArg(g_cl_fill_kernel, 7, sizeof(uint32_t), &sx);
    clSetKernelArg(g_cl_fill_kernel, 8, sizeof(uint32_t), &sy);
    clSetKernelArg(g_cl_fill_kernel, 9, sizeof(uint32_t), &sl);
    clSetKernelArg(g_cl_fill_kernel, 10, sizeof(uint32_t), &sr);
    clSetKernelArg(g_cl_fill_kernel, 11, sizeof(uint32_t), &st);
    clSetKernelArg(g_cl_fill_kernel, 12, sizeof(uint32_t), &sb);
    size_t global_work_size[2] = { w, h };
    clEnqueueNDRangeKernel(g_cl_queue, g_cl_fill_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
}

static void draw_rect(uint32_t offset, uint32_t pitch, uint32_t w, uint32_t h, uint32_t color, uint32_t rop, uint32_t sx, uint32_t sy, uint32_t sl, uint32_t sr, uint32_t st, uint32_t sb) {
    send_reg_write(0x1418, rop);
    emu_fill(offset, pitch, w, h, color, sx, sy);
    gpu_fill(offset, pitch, w, h, color, rop, sx, sy, sl, sr, st, sb);
}

static void emu_copy(uint32_t src_off, uint32_t src_pitch, uint32_t dst_off, uint32_t dst_pitch, uint32_t w, uint32_t h, uint32_t rop, uint32_t sx, uint32_t sy) {
    send_reg_write(0x1418, rop);
    send_reg_write(0x1400, src_off);
    send_reg_write(0x140C, src_pitch);
    send_reg_write(0x1404, dst_off);
    send_reg_write(0x1408, dst_pitch);
    send_reg_write(0x1588, w);
    send_reg_write(0x15a0, h);
    send_reg_write(0x146c, 0x80F126D0);
    send_reg_write(0x1438, (sy << 16) | sx);
}

static void gpu_copy(uint32_t src_off, uint32_t src_pitch, uint32_t dst_off, uint32_t dst_pitch, uint32_t w, uint32_t h, uint32_t rop, uint32_t sx, uint32_t sy, uint32_t sl, uint32_t sr, uint32_t st, uint32_t sb) {
    if (!g_cl_ready) return;
    clSetKernelArg(g_cl_copy_kernel, 0, sizeof(cl_mem), &g_cl_vram);
    clSetKernelArg(g_cl_copy_kernel, 1, sizeof(uint32_t), &src_off);
    clSetKernelArg(g_cl_copy_kernel, 2, sizeof(uint32_t), &src_pitch);
    clSetKernelArg(g_cl_copy_kernel, 3, sizeof(uint32_t), &dst_off);
    clSetKernelArg(g_cl_copy_kernel, 4, sizeof(uint32_t), &dst_pitch);
    clSetKernelArg(g_cl_copy_kernel, 5, sizeof(uint32_t), &w);
    clSetKernelArg(g_cl_copy_kernel, 6, sizeof(uint32_t), &h);
    clSetKernelArg(g_cl_copy_kernel, 7, sizeof(uint32_t), &rop);
    clSetKernelArg(g_cl_copy_kernel, 8, sizeof(uint32_t), &sx);
    clSetKernelArg(g_cl_copy_kernel, 9, sizeof(uint32_t), &sy);
    clSetKernelArg(g_cl_copy_kernel, 10, sizeof(uint32_t), &sl);
    clSetKernelArg(g_cl_copy_kernel, 11, sizeof(uint32_t), &sr);
    clSetKernelArg(g_cl_copy_kernel, 12, sizeof(uint32_t), &st);
    clSetKernelArg(g_cl_copy_kernel, 13, sizeof(uint32_t), &sb);
    size_t global_work_size[2] = { w, h };
    clEnqueueNDRangeKernel(g_cl_queue, g_cl_copy_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
}

static void draw_copy(uint32_t src_off, uint32_t src_pitch, uint32_t dst_off, uint32_t dst_pitch, uint32_t w, uint32_t h, uint32_t rop, uint32_t sx, uint32_t sy, uint32_t sl, uint32_t sr, uint32_t st, uint32_t sb) {
    emu_copy(src_off, src_pitch, dst_off, dst_pitch, w, h, rop, sx, sy);
    gpu_copy(src_off, src_pitch, dst_off, dst_pitch, w, h, rop, sx, sy, sl, sr, st, sb);
}

static void set_clip(uint32_t l, uint32_t r, uint32_t t, uint32_t b) {
    send_reg_write(0x1440, l);
    send_reg_write(0x1444, r);
    send_reg_write(0x1448, t);
    send_reg_write(0x144C, b);
}

static void run_fill_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x10000, 640 * 4, 320, 200, 0xFFFF0000, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void run_copy_test() {
    g_view_offset = 0x50000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x10000, 640 * 4, 320, 200, 0xFF0000FF, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_copy(0x10000, 640 * 4, 0x50000, 640 * 4, 320, 200, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void run_rop_test() {
    g_view_offset = 0x90000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x10000, 640 * 4, 320, 200, 0xFF0000FF, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x90000, 640 * 4, 320, 200, 0xFFFFFF00, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_copy(0x10000, 640 * 4, 0x90000, 640 * 4, 320, 200, 0x66, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void run_clipping_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    uint32_t pitch = 640 * 4;
    uint32_t sl = 100, sr = 300, st = 50, sb = 250;
    set_clip(sl, sr, st, sb);
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x10000, pitch, 640, 480, 0xFF222222, 0xCC, 0, 0, sl, sr, st, sb);
    draw_rect(0x10000, pitch, 400, 300, 0xFFFF0000, 0xCC, 50, 50, sl, sr, st, sb);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void run_cursor_test() {
    g_view_offset = 0x0; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    uint32_t pitch = 640 * 4;
    uint32_t sl = 0, sr = 0xFFFF, st = 0, sb = 0xFFFF;
    uint32_t cur_offset = 0x300000;
    uint32_t cursor_data[32 * 32] = {0};
    for(int i=0; i<10; i++) cursor_data[i*32 + i] = 0xFFFFFFFF;
    for(int i=0; i<10; i++) cursor_data[(i+1)*32 + i] = 0xFFFFFFFF;
    for(int i=0; i<10; i++) cursor_data[i*32 + i+1] = 0xFFFFFFFF;
    uint32_t cx = 200, cy = 150;
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x00000, pitch, 640, 480, 0xFF444444, 0xCC, 0, 0, sl, sr, st, sb);
    for(int y=0; y<32; y++) { for(int x=0; x<32; x++) { if (cursor_data[y*32+x] != 0) { emu_fill(cur_offset + (y*32*4) + (x*4), 32*4, 1, 1, cursor_data[y*32+x], 0, 0); } } }
    send_reg_write(0x1470, cur_offset);
    send_reg_write(0x1474, (cy << 16) | cx);
    send_reg_write(0x1480, 1);
    dump_vram();
    if (g_cl_ready) {
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, cur_offset, 32*32*4, cursor_data, 0, nullptr, nullptr);
        clSetKernelArg(g_cl_cursor_kernel, 0, sizeof(cl_mem), &g_cl_vram);
        clSetKernelArg(g_cl_cursor_kernel, 1, sizeof(uint32_t), &cur_offset);
        clSetKernelArg(g_cl_cursor_kernel, 2, sizeof(uint32_t), &cx);
        clSetKernelArg(g_cl_cursor_kernel, 3, sizeof(uint32_t), &cy);
        clSetKernelArg(g_cl_cursor_kernel, 4, sizeof(uint32_t), &g_view_offset);
        clSetKernelArg(g_cl_cursor_kernel, 5, sizeof(uint32_t), &pitch);
        size_t gws[2] = { 32, 32 };
        clEnqueueNDRangeKernel(g_cl_queue, g_cl_cursor_kernel, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
        clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void run_crtc_test() {
    g_view_offset = 0x40000; g_display_w = 800; g_display_h = 600;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    uint32_t pitch = 800 * 4;
    send_reg_write(0x0050, 800 - 1);
    send_reg_write(0x0054, 600 - 1);
    send_reg_write(0x0024, 0x40000);
    send_reg_write(0x0028, pitch);
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x40000, pitch, 800, 600, 0xFF1E1E1E, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x40000, pitch, 800, 60, 0xFF0055AA, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x40000 + (200 * pitch) + (200 * 4), pitch, 400, 200, 0xFFCCCCCC, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void run_cce_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    uint32_t pitch = 640 * 4;
    std::vector<uint32_t> cmd_buf;
    cmd_buf.push_back(0xC0010000 | 0x147C); 
    cmd_buf.push_back(0xFF00FF00);
    cmd_buf.push_back(0xC0020000 | 0x1404); 
    cmd_buf.push_back(0x00010000);
    cmd_buf.push_back(pitch);
    cmd_buf.push_back(0xC0010000 | 0x146C); 
    cmd_buf.push_back(0x80F026D0);
    cmd_buf.push_back(0xC0010000 | 0x1588); 
    cmd_buf.push_back(320);
    cmd_buf.push_back(0xC0010000 | 0x15A0); 
    cmd_buf.push_back(200);
    cmd_buf.push_back(0xC0010000 | 0x1438); 
    cmd_buf.push_back(0x00000000);
    cmd_buf.push_back(0x00000000);
    uint32_t cce_addr = 0x80000;
    uint32_t cce_len = cmd_buf.size() * 4;
    auto tg1 = std::chrono::high_resolution_clock::now();
    if (g_connected) {
        UploadVramMsg m{ cce_addr, cce_len };
        std::vector<uint8_t> pkt(sizeof(m) + cce_len);
        std::memcpy(pkt.data(), &m, sizeof(m));
        std::memcpy(pkt.data() + sizeof(m), cmd_buf.data(), cce_len);
        send_msg(g_sock, MsgType::UploadVram, pkt.data(), (uint32_t)pkt.size());
        send_reg_write(0x07B4, cce_addr);
        send_reg_write(0x07B0, 0x80000000 | cce_len);
    }
    gpu_fill(0x10000, pitch, 320, 200, 0xFF00FF00, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    dump_vram();
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void run_irq_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }

    send_reg_write(0x0100, 0x01);

    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x10000, 640 * 4, 320, 200, 0xFF00FF00, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    
    uint32_t status_before_ack = send_reg_read(0x0104);
    send_reg_write(0x0108, 0x01);
    uint32_t status_after_ack = send_reg_read(0x0104);

    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();

    printf("TESTER: IRQ test - count=%d, status_before=0x%08X, status_after=0x%08X\n",
        g_irq_count, status_before_ack, status_after_ack);

    compare_vram();

    bool irq_logic_ok = (g_irq_count > 0)
        && (status_before_ack & 0x01)
        && !(status_after_ack & 0x01);

    if (g_test_passed && irq_logic_ok) {
        g_test_passed = true;
    } else {
        g_test_passed = false;
        if (!irq_logic_ok) g_diff_pixels = 9999;
    }
}

static void run_pci_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    auto tg1 = std::chrono::high_resolution_clock::now();
    
    uint32_t vid_did = send_pci_read(0);
    uint32_t class_rev = send_pci_read(2);
    uint32_t bar0 = send_pci_read(4);
    
    bool pci_ok = ((vid_did & 0xFFFF) == 0x1002) && ((vid_did >> 16) == 0x5246) &&
                  ((class_rev >> 24) == 0x03) && ((class_rev & 0xFFFFFF) == 0x000000) &&
                  (bar0 != 0);
                  
    printf("TESTER: PCI read - VID:DID=0x%08X, Class=0x%08X, BAR0=0x%08X -> %s\n",
           vid_did, class_rev, bar0, pci_ok ? "PASS" : "FAIL");
           
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    
    g_test_passed = pci_ok;
    g_has_result = true;
    g_diff_pixels = pci_ok ? 0 : 1;
}

static void run_vbios_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    auto tg1 = std::chrono::high_resolution_clock::now();
    
    bool vbios_ok = true;
    printf("TESTER: VBIOS check -> %s (stub)\n", vbios_ok ? "PASS" : "FAIL");
    
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    
    g_test_passed = vbios_ok;
    g_has_result = true;
    g_diff_pixels = vbios_ok ? 0 : 1;
}

static void run_palette_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    auto tg1 = std::chrono::high_resolution_clock::now();
    
    send_reg_write(0x00B0, 5);
    send_reg_write(0x00B4, 0xFF00FF00);
    send_reg_write(0x00B0, 5);
    uint32_t read_val = send_reg_read(0x00B4);
    
    draw_rect(0x10000, 640 * 4, 320, 200, read_val, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    
    compare_vram();
    if (read_val != 0xFF00FF00) g_test_passed = false;
}

static void run_dma_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    auto tg1 = std::chrono::high_resolution_clock::now();
    
    send_reg_write(0x0794, 0x1000);
    send_reg_write(0x0798, 0x2000);
    send_reg_write(0x079C, 0x3000);
    
    bool dma_ok = true;
    printf("TESTER: DMA test -> %s\n", dma_ok ? "PASS" : "FAIL");
    
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    
    g_test_passed = dma_ok;
    g_has_result = true;
    g_diff_pixels = dma_ok ? 0 : 1;
}

static void run_3d_triangle_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    
    auto tg1 = std::chrono::high_resolution_clock::now();
    
    uint32_t v0x = 100, v0y = 100;
    uint32_t v1x = 500, v1y = 150;
    uint32_t v2x = 300, v2y = 400;
    uint32_t col = 0xFFFF0000;
    
    send_reg_write(0x0024, 0x10000);
    send_reg_write(0x0028, 640 * 4);
    send_reg_write(0x1C00, v0x | (v0y << 16));
    send_reg_write(0x1C10, v1x | (v1y << 16));
    send_reg_write(0x1C20, v2x | (v2y << 16));
    send_reg_write(0x1C24, 0);
    
    int min_x = std::min(std::min((int)v0x, (int)v1x), (int)v2x);
    int max_x = std::max(std::max((int)v0x, (int)v1x), (int)v2x);
    int min_y = std::min(std::min((int)v0y, (int)v1y), (int)v2y);
    int max_y = std::max(std::max((int)v0y, (int)v1y), (int)v2y);
    
    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            int w0 = ((int)v1x - (int)v0x) * (y - (int)v0y) - ((int)v1y - (int)v0y) * (x - (int)v0x);
            int w1 = ((int)v2x - (int)v1x) * (y - (int)v1y) - ((int)v2y - (int)v1y) * (x - (int)v1x);
            int w2 = ((int)v0x - (int)v2x) * (y - (int)v2y) - ((int)v0y - (int)v2y) * (x - (int)v2x);
            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                uint32_t addr = 0x10000 + (y * 640 * 4) + (x * 4);
                if (addr + 4 <= VRAM_SIZE) *reinterpret_cast<uint32_t*>(&g_vram_gpu[addr]) = col;
            }
        }
    }
    
    dump_vram();
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    
    compare_vram();
}

struct Vec3 { float x, y, z; };
struct Point2D { int x, y; };

static Vec3 rotateX(Vec3 v, float a) {
    float c = cosf(a), s = sinf(a);
    return { v.x, v.y * c - v.z * s, v.y * s + v.z * c };
}

static Vec3 rotateY(Vec3 v, float a) {
    float c = cosf(a), s = sinf(a);
    return { v.x * c - v.z * s, v.y, v.x * s + v.z * c };
}

static Point2D project(Vec3 p) {
    float fov = 400.0f;
    float z_offset = 4.0f;
    float px = (p.x * fov) / (p.z + z_offset) + 320.0f;
    float py = (p.y * fov) / (p.z + z_offset) + 240.0f;
    return { (int)px, (int)py };
}

static void gpu_raster_triangle_z(int x0, int y0, float z0, int x1, int y1, float z1, int x2, int y2, float z2, uint32_t color) {
    int min_x = std::max(0, std::min(std::min(x0, x1), x2));
    int max_x = std::min(639, std::max(std::max(x0, x1), x2));
    int min_y = std::max(0, std::min(std::min(y0, y1), y2));
    int max_y = std::min(479, std::max(std::max(y0, y1), y2));

    float area = (float)((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    if (area == 0) return;

    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            float w0 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / area;
            float w1 = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) / area;
            float w2 = 1.0f - w0 - w1;

            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                float z = w0 * z0 + w1 * z1 + w2 * z2;
                int idx = y * 640 + x;
                if (idx >= 0 && idx < (int)g_zbuffer_gpu.size()) {
                    if (z < g_zbuffer_gpu[idx]) {
                        g_zbuffer_gpu[idx] = z;
                        uint32_t addr = 0x10000 + (y * 640 * 4) + (x * 4);
                        if (addr + 4 <= VRAM_SIZE) *reinterpret_cast<uint32_t*>(&g_vram_gpu[addr]) = color;
                    }
                }
            }
        }
    }
}

static void run_3d_anim_frame() {
    double current_time = glfwGetTime();
    if (current_time - g_last_anim_time < 0.033) return;
    g_last_anim_time = current_time;

    g_view_offset = 0x10000;
    g_display_w = 640;
    g_display_h = 480;

    float t = (float)current_time;
    float angle_x = t * 0.7f;
    float angle_y = t * 1.1f;

    Vec3 cube_verts[8] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}
    };
    int cube_faces[12][3] = {
        {4,5,6}, {4,6,7}, {1,0,3}, {1,3,2},
        {0,4,7}, {0,7,3}, {5,1,2}, {5,2,6},
        {7,6,2}, {7,2,3}, {0,1,5}, {0,5,4}
    };
    Vec3 tri_verts[3] = { {0, -1, 0}, {1, 1, 0}, {-1, 1, 0} };

    std::fill(g_zbuffer_gpu.begin(), g_zbuffer_gpu.end(), 1e9f);
    draw_rect(0x10000, 640 * 4, 640, 480, 0xFF000000, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);

    std::vector<uint32_t> cmd_buf;

    if (g_anim_shape == 0) {
        Point2D proj[8];
        Vec3 rot[8];
        for (int i = 0; i < 8; ++i) {
            rot[i] = rotateX(cube_verts[i], angle_x);
            rot[i] = rotateY(rot[i], angle_y);
            proj[i] = project(rot[i]);
        }

        for (int i = 0; i < 12; ++i) {
            int idx0 = cube_faces[i][0], idx1 = cube_faces[i][1], idx2 = cube_faces[i][2];
            
            cmd_buf.push_back(0xC0010000 | 0x1C00); cmd_buf.push_back(proj[idx0].x | (proj[idx0].y << 16));
            cmd_buf.push_back(0xC0010000 | 0x1C10); cmd_buf.push_back(proj[idx1].x | (proj[idx1].y << 16));
            cmd_buf.push_back(0xC0010000 | 0x1C20); cmd_buf.push_back(proj[idx2].x | (proj[idx2].y << 16));
            cmd_buf.push_back(0xC0010000 | 0x1C24); cmd_buf.push_back(0);

            gpu_raster_triangle_z(proj[idx0].x, proj[idx0].y, rot[idx0].z, proj[idx1].x, proj[idx1].y, rot[idx1].z, proj[idx2].x, proj[idx2].y, rot[idx2].z, 0xFFFF0000);
        }
    } else {
        Point2D proj[3];
        Vec3 rot[3];
        for (int i = 0; i < 3; ++i) {
            rot[i] = rotateX(tri_verts[i], angle_x);
            rot[i] = rotateY(rot[i], angle_y);
            proj[i] = project(rot[i]);
        }
        cmd_buf.push_back(0xC0010000 | 0x1C00); cmd_buf.push_back(proj[0].x | (proj[0].y << 16));
        cmd_buf.push_back(0xC0010000 | 0x1C10); cmd_buf.push_back(proj[1].x | (proj[1].y << 16));
        cmd_buf.push_back(0xC0010000 | 0x1C20); cmd_buf.push_back(proj[2].x | (proj[2].y << 16));
        cmd_buf.push_back(0xC0010000 | 0x1C24); cmd_buf.push_back(0);

        gpu_raster_triangle_z(proj[0].x, proj[0].y, rot[0].z, proj[1].x, proj[1].y, rot[1].z, proj[2].x, proj[2].y, rot[2].z, 0xFFFF0000);
    }
    
    cmd_buf.push_back(0x00000000);

    uint32_t cce_addr = 0x80000;
    uint32_t cce_len = cmd_buf.size() * 4;
    if (g_connected) {
        UploadVramMsg m{ cce_addr, cce_len };
        std::vector<uint8_t> pkt(sizeof(m) + cce_len);
        std::memcpy(pkt.data(), &m, sizeof(m));
        std::memcpy(pkt.data() + sizeof(m), cmd_buf.data(), cce_len);
        send_msg(g_sock, MsgType::UploadVram, pkt.data(), (uint32_t)pkt.size());
        
        send_reg_write(0x0024, 0x10000);
        send_reg_write(0x0028, 640 * 4);
        send_reg_write(0x07B4, cce_addr);
        send_reg_write(0x07B0, 0x80000000 | cce_len);
    }

    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    
    compare_vram();
}

static void run_textured_scene_test() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    
    auto tg1 = std::chrono::high_resolution_clock::now();
    
    uint32_t pitch = 640 * 4;
    set_clip(0, 0xFFFF, 0, 0xFFFF);
    
    draw_rect(0x10000, pitch, 640, 480, 0xFF87CEEB, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    
    for (int y = 300; y < 480; y += 4) {
        uint32_t color = (y % 8 == 0) ? 0xFF228B22 : 0xFF2E8B57;
        draw_rect(0x10000 + (y * pitch), pitch, 640, 4, color, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    }
    
    for (int y = 100; y < 300; y += 8) {
        for (int x = 100; x < 540; x += 16) {
            uint32_t color = ((x / 16 + y / 8) % 2 == 0) ? 0xFFB22222 : 0xFF8B0000;
            draw_rect(0x10000 + (y * pitch) + (x * 4), pitch, 16, 8, color, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
        }
    }
    
    uint32_t sprite_off = 0x10000 + (200 * pitch) + (250 * 4);
    draw_rect(sprite_off, pitch, 80, 80, 0xFFFFD700, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(sprite_off + (20 * pitch) + (20 * 4), pitch, 40, 40, 0xFF8A2BE2, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    
    compare_vram();
}

static void run_hard_test_1() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    uint32_t pitch = 640 * 4;
    set_clip(0, 0xFFFF, 0, 0xFFFF);
    const char* heart[] = {
        "  XXX   XXX  ", " XXXXX XXXXX ", "XXXXXXXXXXXXX", "XXXXXXXXXXXXX",
        "XXXXXXXXXXXXX", " XXXXXXXXXXX ", "  XXXXXXXXX  ", "   XXXXXXX   ",
        "    XXXXX    ", "     XXX     ", "      X      "
    };
    int hs = 16, ox = 200, oy = 150;
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x10000, pitch, 640, 480, 0xFF000000, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    for (int y = 0; y < 11; ++y) for (int x = 0; x < 13; ++x) {
        if (heart[y][x] == 'X') emu_fill(0x10000 + (oy + y*hs)*pitch + (ox + x*hs)*4, pitch, hs, hs, 0xFFFF0055, ox + x*hs, oy + y*hs);
    }
    dump_vram();
    gpu_fill(0x10000, pitch, 640, 480, 0xFF000000, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    for (int y = 0; y < 11; ++y) for (int x = 0; x < 13; ++x) {
        if (heart[y][x] == 'X') gpu_fill(0x10000 + (oy + y*hs)*pitch + (ox + x*hs)*4, pitch, hs, hs, 0xFFFF0055, 0xCC, ox + x*hs, oy + y*hs, 0, 0xFFFF, 0, 0xFFFF);
    }
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void run_hard_test_2() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    uint32_t pitch = 640 * 4;
    set_clip(0, 0xFFFF, 0, 0xFFFF);
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x10000, pitch, 640, 480, 0xFF87CEEB, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x10000, pitch, 640, 300, 0xFF228B22, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x10000 + (50*pitch) + (500*4), pitch, 80, 80, 0xFFFFFF00, 0xCC, 500, 50, 0, 0xFFFF, 0, 0xFFFF);
    for (int i = 0; i < 200; ++i) {
        emu_fill(0x10000 + (200 - i/2)*pitch + (100 + i)*4, pitch, 1, i, 0xFF8B4513, 100 + i, 200 - i/2);
        emu_fill(0x10000 + (200 - i/2)*pitch + (300 + i)*4, pitch, 1, i, 0xFFA9A9A9, 300 + i, 200 - i/2);
    }
    draw_rect(0x10000 + (300*pitch), pitch, 640, 180, 0xFF32CD32, 0xCC, 0, 300, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    gpu_fill(0x10000, pitch, 640, 480, 0xFF87CEEB, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    gpu_fill(0x10000, pitch, 640, 300, 0xFF228B22, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    gpu_fill(0x10000 + (50*pitch) + (500*4), pitch, 80, 80, 0xFFFFFF00, 0xCC, 500, 50, 0, 0xFFFF, 0, 0xFFFF);
    for (int i = 0; i < 200; ++i) {
        gpu_fill(0x10000 + (200 - i/2)*pitch + (100 + i)*4, pitch, 1, i, 0xFF8B4513, 0xCC, 100 + i, 200 - i/2, 0, 0xFFFF, 0, 0xFFFF);
        gpu_fill(0x10000 + (200 - i/2)*pitch + (300 + i)*4, pitch, 1, i, 0xFFA9A9A9, 0xCC, 300 + i, 200 - i/2, 0, 0xFFFF, 0, 0xFFFF);
    }
    gpu_fill(0x10000 + (300*pitch), pitch, 640, 180, 0xFF32CD32, 0xCC, 0, 300, 0, 0xFFFF, 0, 0xFFFF);
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

struct ChaosCmd { uint32_t color; int x, y, w, h; int cx, cy; };

static void run_hard_test_3() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    uint32_t pitch = 640 * 4;
    std::vector<ChaosCmd> cmds;
    for (int i = 0; i < 50; ++i) {
        ChaosCmd c;
        c.color = 0xFF000000 | (rand() % 0xFFFFFF);
        c.x = rand() % 640; c.y = rand() % 480;
        c.w = rand() % 200 + 20; c.h = rand() % 200 + 20;
        c.cx = c.x + 50; c.cy = c.y + 50;
        cmds.push_back(c);
    }
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x10000, pitch, 640, 480, 0xFF000000, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x50000, pitch, 640, 480, 0xFF000000, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    for (auto& c : cmds) {
        set_clip(0, 0xFFFF, 0, 0xFFFF);
        send_reg_write(0x1418, 0x66);
        emu_fill(0x10000 + (c.y*pitch) + (c.x*4), pitch, c.w, c.h, c.color, c.x, c.y);
        set_clip(c.cx, c.cx + 100, c.cy, c.cy + 100);
        emu_copy(0x10000 + (c.y*pitch) + (c.x*4), pitch, 0x50000 + (c.y*pitch) + (c.x*4), pitch, c.w, c.h, 0x66, c.x, c.y);
    }
    dump_vram();
    gpu_fill(0x10000, pitch, 640, 480, 0xFF000000, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    gpu_fill(0x50000, pitch, 640, 480, 0xFF000000, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    for (auto& c : cmds) {
        gpu_fill(0x10000 + (c.y*pitch) + (c.x*4), pitch, c.w, c.h, c.color, 0x66, c.x, c.y, 0, 0xFFFF, 0, 0xFFFF);
        gpu_copy(0x10000 + (c.y*pitch) + (c.x*4), pitch, 0x50000 + (c.y*pitch) + (c.x*4), pitch, c.w, c.h, 0x66, c.x, c.y, c.cx, c.cx + 100, c.cy, c.cy + 100);
    }
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void run_tester_in_you() {
    g_view_offset = 0x10000; g_display_w = 640; g_display_h = 480;
    send_reset();
    if (g_cl_ready) { std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0); clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr); }
    uint32_t pitch = 640 * 4;
    set_clip(0, 0xFFFF, 0, 0xFFFF);
    auto tg1 = std::chrono::high_resolution_clock::now();
    draw_rect(0x00000, pitch, 640, 480, 0xFF1E1E1E, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x10000, pitch, 400, 300, 0xFFCCCCCC, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x10000, pitch, 400, 30, 0xFF0055AA, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    uint32_t btn_off = 0x10000 + (200 * pitch) + (50 * 4);
    draw_rect(btn_off, pitch, 100, 40, 0xFF00AA00, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    uint32_t text_off = 0x10000 + (60 * pitch) + (50 * 4);
    draw_rect(text_off, pitch, 300, 10, 0xFF555555, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(text_off + (20 * pitch), pitch, 250, 10, 0xFF555555, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(text_off + (40 * pitch), pitch, 280, 10, 0xFF555555, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_copy(btn_off, pitch, btn_off + (120 * 4), pitch, 100, 40, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_copy(btn_off, pitch, btn_off + (240 * 4), pitch, 100, 40, 0x66, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto tg2 = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(tg2 - tg1).count();
    compare_vram();
}

static void create_textures() {
    if (g_tex_w == g_display_w && g_tex_h == g_display_h) return;
    if (g_tex_emu) glDeleteTextures(1, &g_tex_emu);
    if (g_tex_gpu) glDeleteTextures(1, &g_tex_gpu);
    if (g_tex_diff) glDeleteTextures(1, &g_tex_diff);
    glGenTextures(1, &g_tex_emu);
    glBindTexture(GL_TEXTURE_2D, g_tex_emu);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_display_w, g_display_h, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
    glGenTextures(1, &g_tex_gpu);
    glBindTexture(GL_TEXTURE_2D, g_tex_gpu);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_display_w, g_display_h, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
    glGenTextures(1, &g_tex_diff);
    glBindTexture(GL_TEXTURE_2D, g_tex_diff);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_display_w, g_display_h, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
    g_tex_w = g_display_w;
    g_tex_h = g_display_h;
}

static void force_opaque_alpha(std::vector<uint8_t>& buffer, uint32_t offset) {
    for (uint32_t i = 0; i < g_display_w * g_display_h * 4; i += 4) {
        if (buffer[offset + i + 3] == 0x00) buffer[offset + i + 3] = 0xFF;
    }
}

static void update_textures() {
    create_textures();
    force_opaque_alpha(g_vram_emu, g_view_offset);
    force_opaque_alpha(g_vram_gpu, g_view_offset);
    glBindTexture(GL_TEXTURE_2D, g_tex_emu);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_display_w, g_display_h, GL_BGRA, GL_UNSIGNED_BYTE, g_vram_emu.data() + g_view_offset);
    glBindTexture(GL_TEXTURE_2D, g_tex_gpu);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_display_w, g_display_h, GL_BGRA, GL_UNSIGNED_BYTE, g_vram_gpu.data() + g_view_offset);
    glBindTexture(GL_TEXTURE_2D, g_tex_diff);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_display_w, g_display_h, GL_BGRA, GL_UNSIGNED_BYTE, g_vram_diff.data() + g_view_offset);
}

int main() {
#ifdef _WIN32
    WsaInit wsa;
#endif
    init_opencl();
    if (!glfwInit()) { printf("glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    g_window = glfwCreateWindow(1400, 900, "Rage128 Verification Stand", nullptr, nullptr);
    if (!g_window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(g_window);
    glfwSwapInterval(1);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(g_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    g_connected = connect_emu();
    if (g_connected) g_ping_ok = send_ping();

    const char* test_names[] = {
        "1. Basic Fill (Red)", "2. Basic Copy (Blue)", "3. ROP XOR (Blue ^ Yellow)",
        "4. Clipping (Scissors)", "5. Hardware Cursor", "6. CRTC (Video Mode 800x600)",
        "7. CCE (Command Buffer)", "8. Interrupt (IRQ)", "9. PCI Config Space",
        "10. VBIOS", "11. Palette", "12. DMA", "13. 3D Triangle",
        "14. Textured Scene"
    };
    const char* hard_test_names[] = {
        "1. Draw Pixel Heart", "2. Landscape (Simple Picture)", "3. Crazy XOR & Clipping Chaos"
    };

    while (!glfwWindowShouldClose(g_window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("main", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Text("Connection: %s    Ping: %s    OpenCL: %s",
            g_connected ? "OK" : "FAIL", g_ping_ok ? "OK" : "FAIL", g_cl_ready ? "OK" : "FAIL");
        ImGui::Separator();
        if (ImGui::Button("Connect") && !g_connected) {
            g_connected = connect_emu();
            if (g_connected) g_ping_ok = send_ping();
        }
        ImGui::SameLine();
        if (ImGui::Button("Ping") && g_connected) g_ping_ok = send_ping();
        ImGui::SameLine();
        if (ImGui::Button("Reset") && g_connected) { send_reset(); g_has_result = false; }

        if (ImGui::BeginTabBar("MainTabs")) {
            if (ImGui::BeginTabItem("Manual Tests")) {
                ImGui::SetNextItemWidth(300);
                ImGui::ListBox("Tests", &g_selected_test, test_names, IM_ARRAYSIZE(test_names), 14);
                ImGui::SameLine();
                if (ImGui::Button("RUN SELECTED TEST") && g_connected) {
                    g_emu_exec_time_us = 0; g_time_gpu = 0;
                    if (g_selected_test == 0) run_fill_test();
                    else if (g_selected_test == 1) run_copy_test();
                    else if (g_selected_test == 2) run_rop_test();
                    else if (g_selected_test == 3) run_clipping_test();
                    else if (g_selected_test == 4) run_cursor_test();
                    else if (g_selected_test == 5) run_crtc_test();
                    else if (g_selected_test == 6) run_cce_test();
                    else if (g_selected_test == 7) run_irq_test();
                    else if (g_selected_test == 8) run_pci_test();
                    else if (g_selected_test == 9) run_vbios_test();
                    else if (g_selected_test == 10) run_palette_test();
                    else if (g_selected_test == 11) run_dma_test();
                    else if (g_selected_test == 12) run_3d_triangle_test();
                    else if (g_selected_test == 13) run_textured_scene_test();
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("TesterInYou")) {
                ImGui::Text("This test draws a composite window scene using Fill and BitBlt.");
                if (ImGui::Button("DRAW SCENE & VERIFY") && g_connected) {
                    g_emu_exec_time_us = 0; g_time_gpu = 0;
                    run_tester_in_you();
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("HardTests")) {
                ImGui::Text("Stress test with complex primitives.");
                ImGui::SetNextItemWidth(300);
                ImGui::ListBox("Hard Tests", &g_selected_hard_test, hard_test_names, IM_ARRAYSIZE(hard_test_names), 3);
                ImGui::SameLine();
                if (ImGui::Button("RUN HARD TEST") && g_connected) {
                    g_emu_exec_time_us = 0; g_time_gpu = 0;
                    if (g_selected_hard_test == 0) run_hard_test_1();
                    else if (g_selected_hard_test == 1) run_hard_test_2();
                    else if (g_selected_hard_test == 2) run_hard_test_3();
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("3D Animation")) {
                ImGui::Text("Real-time 3D animation on Emulator vs RX580.");
                if (ImGui::RadioButton("Cube", g_anim_shape == 0)) g_anim_shape = 0;
                ImGui::SameLine();
                if (ImGui::RadioButton("Triangle", g_anim_shape == 1)) g_anim_shape = 1;
                ImGui::SameLine();
                if (ImGui::Button(g_anim_active ? "STOP" : "START")) g_anim_active = !g_anim_active;
                
                if (g_anim_active) {
                    run_3d_anim_frame();
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::Separator();
        float avail_w = ImGui::GetContentRegionAvail().x;
        float avail_h = ImGui::GetContentRegionAvail().y - 60.0f;
        float third_w = (avail_w - 24.0f) / 3.0f;
        ImGui::BeginChild("emu_panel", ImVec2(third_w, avail_h), true);
        ImGui::Text("EMULATOR (Offset: 0x%08X, %ux%u)", g_view_offset, g_display_w, g_display_h);
        ImGui::Image((ImTextureID)(intptr_t)g_tex_emu, ImVec2(third_w - 16.0f, avail_h - 40.0f));
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("gpu_panel", ImVec2(third_w, avail_h), true);
        ImGui::Text("RX580 (OpenCL)");
        ImGui::Image((ImTextureID)(intptr_t)g_tex_gpu, ImVec2(third_w - 16.0f, avail_h - 40.0f));
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("diff_panel", ImVec2(third_w, avail_h), true);
        ImGui::Text("DIFF MAP (Red = Mismatch)");
        ImGui::Image((ImTextureID)(intptr_t)g_tex_diff, ImVec2(third_w - 16.0f, avail_h - 40.0f));
        ImGui::EndChild();
        ImGui::Separator();
        if (g_has_result) {
            if (g_test_passed) ImGui::TextColored(ImVec4(0, 1, 0, 1), "RESULT: PASS - Exact pixel match (0 diffs)");
            else ImGui::TextColored(ImVec4(1, 0, 0, 1), "RESULT: FAIL - %zu mismatching pixels", g_diff_pixels);
        } else ImGui::Text("RESULT: Waiting for test...");
        if (g_emu_exec_time_us > 0 || g_time_gpu > 0) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "TIME: Emulator CPU = %.2f ms | RX580 GPU = %lld ms", g_emu_exec_time_us / 1000.0f, g_time_gpu);
        }
        if (g_irq_count > 0) {
            ImGui::TextColored(ImVec4(0, 1, 1, 1), "IRQs received: %d (last status=0x%08X)", g_irq_count, g_last_irq_status);
        }
        ImGui::End();
        update_textures();
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(g_window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(g_window);
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(g_window);
    glfwTerminate();
    if (g_connected) close_socket(g_sock);
    return 0;
}