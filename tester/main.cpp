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

using namespace vgpu;

static GLFWwindow*        g_window      = nullptr;
static socket_t           g_sock        = INVALID_SOCK;
static bool               g_connected   = false;
static bool               g_ping_ok     = false;
static std::vector<uint8_t> g_vram_emu(VRAM_SIZE, 0);
static std::vector<uint8_t> g_vram_gpu(VRAM_SIZE, 0);
static std::vector<uint8_t> g_vram_diff(VRAM_SIZE, 0);
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

static long long          g_time_emu    = 0;
static long long          g_time_gpu    = 0;

static cl_context       g_cl_context   = nullptr;
static cl_command_queue g_cl_queue     = nullptr;
static cl_program       g_cl_program   = nullptr;
static cl_kernel        g_cl_fill_kernel = nullptr;
static cl_kernel        g_cl_copy_kernel = nullptr;
static cl_kernel        g_cl_cursor_kernel = nullptr;
static cl_mem           g_cl_vram      = nullptr;
static bool             g_cl_ready     = false;

const char* KERNEL_SRC = R"(
__kernel void fill_kernel(__global uchar* vram, uint offset, uint pitch, uint width, uint height, uint color, uint start_x, uint start_y, uint sc_left, uint sc_right, uint sc_top, uint sc_bottom) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int abs_x = start_x + x;
        int abs_y = start_y + y;
        if (abs_x < sc_left || abs_x >= sc_right || abs_y < sc_top || abs_y >= sc_bottom) return;
        
        uint pixel_addr = offset + (y * pitch) + (x * 4);
        vram[pixel_addr + 0] = (color >> 0) & 0xFF;
        vram[pixel_addr + 1] = (color >> 8) & 0xFF;
        vram[pixel_addr + 2] = (color >> 16) & 0xFF;
        vram[pixel_addr + 3] = (color >> 24) & 0xFF;
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

static bool send_ping() {
    if (!g_connected) return false;
    PingMsg p{ 0xDEADBEEF };
    if (!send_msg(g_sock, MsgType::Ping, &p, sizeof(p))) return false;
    uint8_t buf[64];
    MsgType t; uint32_t len;
    if (!recv_msg(g_sock, t, buf, sizeof(buf), len)) return false;
    if (t != MsgType::Pong || len != sizeof(PongMsg)) return false;
    PongMsg r; std::memcpy(&r, buf, sizeof(r));
    return r.token == p.token;
}

static bool send_reset() {
    if (!g_connected) return false;
    if (!send_msg(g_sock, MsgType::Reset, nullptr, 0)) return false;
    uint8_t buf[16];
    MsgType t; uint32_t len;
    if (!recv_msg(g_sock, t, buf, sizeof(buf), len)) return false;
    return t == MsgType::Pong;
}

static bool send_reg_write(uint32_t addr, uint32_t value) {
    if (!g_connected) return false;
    RegWriteMsg m{ addr, value };
    return send_msg(g_sock, MsgType::RegWrite, &m, sizeof(m));
}

static bool dump_vram() {
    if (!g_connected) return false;
    DumpVramMsg m{ 0, VRAM_SIZE };
    if (!send_msg(g_sock, MsgType::DumpVram, &m, sizeof(m))) return false;
    
    uint8_t* buf = g_vram_emu.data();
    MsgType t; uint32_t len;
    if (!recv_msg(g_sock, t, buf, VRAM_SIZE, len)) return false;
    return t == MsgType::DumpVramResp;
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

static void gpu_fill(uint32_t offset, uint32_t pitch, uint32_t w, uint32_t h, uint32_t color, uint32_t sx, uint32_t sy, uint32_t sl, uint32_t sr, uint32_t st, uint32_t sb) {
    if (!g_cl_ready) return;
    clSetKernelArg(g_cl_fill_kernel, 0, sizeof(cl_mem), &g_cl_vram);
    clSetKernelArg(g_cl_fill_kernel, 1, sizeof(uint32_t), &offset);
    clSetKernelArg(g_cl_fill_kernel, 2, sizeof(uint32_t), &pitch);
    clSetKernelArg(g_cl_fill_kernel, 3, sizeof(uint32_t), &w);
    clSetKernelArg(g_cl_fill_kernel, 4, sizeof(uint32_t), &h);
    clSetKernelArg(g_cl_fill_kernel, 5, sizeof(uint32_t), &color);
    clSetKernelArg(g_cl_fill_kernel, 6, sizeof(uint32_t), &sx);
    clSetKernelArg(g_cl_fill_kernel, 7, sizeof(uint32_t), &sy);
    clSetKernelArg(g_cl_fill_kernel, 8, sizeof(uint32_t), &sl);
    clSetKernelArg(g_cl_fill_kernel, 9, sizeof(uint32_t), &sr);
    clSetKernelArg(g_cl_fill_kernel, 10, sizeof(uint32_t), &st);
    clSetKernelArg(g_cl_fill_kernel, 11, sizeof(uint32_t), &sb);
    size_t global_work_size[2] = { w, h };
    clEnqueueNDRangeKernel(g_cl_queue, g_cl_fill_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
}

static void draw_rect(uint32_t offset, uint32_t pitch, uint32_t w, uint32_t h, uint32_t color, uint32_t sx, uint32_t sy, uint32_t sl, uint32_t sr, uint32_t st, uint32_t sb) {
    emu_fill(offset, pitch, w, h, color, sx, sy);
    gpu_fill(offset, pitch, w, h, color, sx, sy, sl, sr, st, sb);
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
    g_view_offset = 0x10000;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    draw_rect(0x10000, 640 * 4, 320, 200, 0xFFFF0000, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    compare_vram();
}

static void run_copy_test() {
    g_view_offset = 0x50000;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    draw_rect(0x10000, 640 * 4, 320, 200, 0xFF0000FF, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_copy(0x10000, 640 * 4, 0x50000, 640 * 4, 320, 200, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    compare_vram();
}

static void run_rop_test() {
    g_view_offset = 0x90000;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    draw_rect(0x10000, 640 * 4, 320, 200, 0xFF0000FF, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x90000, 640 * 4, 320, 200, 0xFFFFFF00, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_copy(0x10000, 640 * 4, 0x90000, 640 * 4, 320, 200, 0x66, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    compare_vram();
}

static void run_clipping_test() {
    g_view_offset = 0x10000;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    uint32_t pitch = 640 * 4;
    uint32_t sl = 100, sr = 300, st = 50, sb = 250;
    set_clip(sl, sr, st, sb);
    draw_rect(0x10000, pitch, 640, 480, 0xFF222222, 0, 0, sl, sr, st, sb);
    draw_rect(0x10000, pitch, 400, 300, 0xFFFF0000, 50, 50, sl, sr, st, sb);
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    compare_vram();
}

static void run_cursor_test() {
    g_view_offset = 0x0;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    uint32_t pitch = 640 * 4;
    uint32_t sl = 0, sr = 0xFFFF, st = 0, sb = 0xFFFF;
    draw_rect(0x00000, pitch, 640, 480, 0xFF444444, 0, 0, sl, sr, st, sb);
    uint32_t cur_offset = 0x300000;
    uint32_t cursor_data[32 * 32] = {0};
    for(int i=0; i<10; i++) cursor_data[i*32 + i] = 0xFFFFFFFF;
    for(int i=0; i<10; i++) cursor_data[(i+1)*32 + i] = 0xFFFFFFFF;
    for(int i=0; i<10; i++) cursor_data[i*32 + i+1] = 0xFFFFFFFF;
    for(int y=0; y<32; y++) {
        for(int x=0; x<32; x++) {
            if (cursor_data[y*32+x] != 0) {
                emu_fill(cur_offset + (y*32*4) + (x*4), 32*4, 1, 1, cursor_data[y*32+x], 0, 0);
            }
        }
    }
    if (g_cl_ready) clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, cur_offset, 32*32*4, cursor_data, 0, nullptr, nullptr);
    uint32_t cx = 200, cy = 150;
    send_reg_write(0x1470, cur_offset);
    send_reg_write(0x1474, (cy << 16) | cx);
    send_reg_write(0x1480, 1);
    if (g_cl_ready) {
        clSetKernelArg(g_cl_cursor_kernel, 0, sizeof(cl_mem), &g_cl_vram);
        clSetKernelArg(g_cl_cursor_kernel, 1, sizeof(uint32_t), &cur_offset);
        clSetKernelArg(g_cl_cursor_kernel, 2, sizeof(uint32_t), &cx);
        clSetKernelArg(g_cl_cursor_kernel, 3, sizeof(uint32_t), &cy);
        clSetKernelArg(g_cl_cursor_kernel, 4, sizeof(uint32_t), &g_view_offset);
        clSetKernelArg(g_cl_cursor_kernel, 5, sizeof(uint32_t), &pitch);
        size_t global_work_size[2] = { 32, 32 };
        clEnqueueNDRangeKernel(g_cl_queue, g_cl_cursor_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    }
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    compare_vram();
}

static void run_crtc_test() {
    g_view_offset = 0x40000;
    g_display_w = 800;
    g_display_h = 600;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    
    uint32_t pitch = 800 * 4;
    send_reg_write(0x0050, 800 - 1);
    send_reg_write(0x0054, 600 - 1);
    send_reg_write(0x0024, 0x40000);
    send_reg_write(0x0028, pitch);

    draw_rect(0x40000, pitch, 800, 600, 0xFF1E1E1E, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x40000, pitch, 800, 60, 0xFF0055AA, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x40000 + (200 * pitch) + (200 * 4), pitch, 400, 200, 0xFFCCCCCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    compare_vram();
}

static void run_cce_test() {
    g_view_offset = 0x10000;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }

    uint32_t pitch = 640 * 4;
    
    std::vector<uint32_t> cmd_buf;
    
    cmd_buf.push_back(0xC0000100 | 0x147C); 
    cmd_buf.push_back(0xFF00FF00);
    
    cmd_buf.push_back(0xC0000200 | 0x1404); 
    cmd_buf.push_back(0x00010000);
    cmd_buf.push_back(pitch);
    
    cmd_buf.push_back(0xC0000300 | 0x146C);
    cmd_buf.push_back(0x80F026D0);
    cmd_buf.push_back(320);
    cmd_buf.push_back(200);
    
    cmd_buf.push_back(0xC0000100 | 0x1438);
    cmd_buf.push_back(0x00000000);
    
    cmd_buf.push_back(0x00000000);

    uint32_t cce_addr = 0x80000;
    uint32_t cce_len = cmd_buf.size() * 4;
    
    if (g_connected) {
        UploadVramMsg m{ cce_addr, cce_len };
        std::vector<uint8_t> pkt(sizeof(m) + cce_len);
        std::memcpy(pkt.data(), &m, sizeof(m));
        std::memcpy(pkt.data() + sizeof(m), cmd_buf.data(), cce_len);
        send_msg(g_sock, MsgType::UploadVram, pkt.data(), (uint32_t)pkt.size());
        
        send_reg_write(0x07B4, cce_addr);
        send_reg_write(0x07B0, 0x80000000 | cce_len);
    }

    draw_rect(0x10000, pitch, 320, 200, 0xFF00FF00, 0, 0, 0, 0xFFFF, 0, 0xFFFF);

    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    compare_vram();
}

static void run_hard_test_1() {
    g_view_offset = 0x10000;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    uint32_t pitch = 640 * 4;
    set_clip(0, 0xFFFF, 0, 0xFFFF);

    draw_rect(0x10000, pitch, 640, 480, 0xFF000000, 0, 0, 0, 0xFFFF, 0, 0xFFFF);

    const char* heart[] = {
        "  XXX   XXX  ",
        " XXXXX XXXXX ",
        "XXXXXXXXXXXXX",
        "XXXXXXXXXXXXX",
        "XXXXXXXXXXXXX",
        " XXXXXXXXXXX ",
        "  XXXXXXXXX  ",
        "   XXXXXXX   ",
        "    XXXXX    ",
        "     XXX     ",
        "      X      "
    };
    int hs = 16;
    int ox = 200, oy = 150;
    for (int y = 0; y < 11; ++y) {
        for (int x = 0; x < 13; ++x) {
            if (heart[y][x] == 'X') {
                draw_rect(0x10000 + (oy + y*hs)*pitch + (ox + x*hs)*4, pitch, hs, hs, 0xFFFF0055, ox + x*hs, oy + y*hs, 0, 0xFFFF, 0, 0xFFFF);
            }
        }
    }
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    compare_vram();
}

static void run_hard_test_2() {
    g_view_offset = 0x10000;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    uint32_t pitch = 640 * 4;
    set_clip(0, 0xFFFF, 0, 0xFFFF);

    draw_rect(0x10000, pitch, 640, 480, 0xFF87CEEB, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x10000, pitch, 640, 300, 0xFF228B22, 0, 0, 0, 0xFFFF, 0, 0xFFFF);

    draw_rect(0x10000 + (50*pitch) + (500*4), pitch, 80, 80, 0xFFFFFF00, 500, 50, 0, 0xFFFF, 0, 0xFFFF);

    for (int i = 0; i < 200; ++i) {
        draw_rect(0x10000 + (200 - i/2)*pitch + (100 + i)*4, pitch, 1, i, 0xFF8B4513, 100 + i, 200 - i/2, 0, 0xFFFF, 0, 0xFFFF);
        draw_rect(0x10000 + (200 - i/2)*pitch + (300 + i)*4, pitch, 1, i, 0xFFA9A9A9, 300 + i, 200 - i/2, 0, 0xFFFF, 0, 0xFFFF);
    }

    draw_rect(0x10000 + (300*pitch), pitch, 640, 180, 0xFF32CD32, 0, 300, 0, 0xFFFF, 0, 0xFFFF);
    
    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    compare_vram();
}

static void run_hard_test_3() {
    g_view_offset = 0x10000;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    uint32_t pitch = 640 * 4;
    
    draw_rect(0x10000, pitch, 640, 480, 0xFF000000, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x50000, pitch, 640, 480, 0xFF000000, 0, 0, 0, 0xFFFF, 0, 0xFFFF);

    auto start_emu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 50; ++i) {
        uint32_t color = 0xFF000000 | (rand() % 0xFFFFFF);
        int x = rand() % 640;
        int y = rand() % 480;
        int w = rand() % 200 + 20;
        int h = rand() % 200 + 20;
        
        set_clip(0, 0xFFFF, 0, 0xFFFF);
        emu_fill(0x10000 + (y*pitch) + (x*4), pitch, w, h, color, x, y);

        int cx = x + 50, cy = y + 50;
        set_clip(cx, cx + 100, cy, cy + 100);
        emu_copy(0x10000 + (y*pitch) + (x*4), pitch, 0x50000 + (y*pitch) + (x*4), pitch, w, h, 0x66, x, y);
    }
    dump_vram();
    auto end_emu = std::chrono::high_resolution_clock::now();
    g_time_emu = std::chrono::duration_cast<std::chrono::milliseconds>(end_emu - start_emu).count();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        uint32_t color = 0xFF000000 | (rand() % 0xFFFFFF);
        int x = rand() % 640;
        int y = rand() % 480;
        int w = rand() % 200 + 20;
        int h = rand() % 200 + 20;
        
        gpu_fill(0x10000 + (y*pitch) + (x*4), pitch, w, h, color, x, y, 0, 0xFFFF, 0, 0xFFFF);

        int cx = x + 50, cy = y + 50;
        gpu_copy(0x10000 + (y*pitch) + (x*4), pitch, 0x50000 + (y*pitch) + (x*4), pitch, w, h, 0x66, x, y, cx, cx + 100, cy, cy + 100);
    }
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    g_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count();

    compare_vram();
}

static void run_tester_in_you() {
    g_view_offset = 0x10000;
    g_display_w = 640;
    g_display_h = 480;
    send_reset();
    if (g_cl_ready) {
        std::fill(g_vram_gpu.begin(), g_vram_gpu.end(), 0);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
    }
    uint32_t pitch = 640 * 4;
    set_clip(0, 0xFFFF, 0, 0xFFFF);

    draw_rect(0x00000, pitch, 640, 480, 0xFF1E1E1E, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x10000, pitch, 400, 300, 0xFFCCCCCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(0x10000, pitch, 400, 30, 0xFF0055AA, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    
    uint32_t btn_off = 0x10000 + (200 * pitch) + (50 * 4);
    draw_rect(btn_off, pitch, 100, 40, 0xFF00AA00, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    
    uint32_t text_off = 0x10000 + (60 * pitch) + (50 * 4);
    draw_rect(text_off, pitch, 300, 10, 0xFF555555, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(text_off + (20 * pitch), pitch, 250, 10, 0xFF555555, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_rect(text_off + (40 * pitch), pitch, 280, 10, 0xFF555555, 0, 0, 0, 0xFFFF, 0, 0xFFFF);

    draw_copy(btn_off, pitch, btn_off + (120 * 4), pitch, 100, 40, 0xCC, 0, 0, 0, 0xFFFF, 0, 0xFFFF);
    draw_copy(btn_off, pitch, btn_off + (240 * 4), pitch, 100, 40, 0x66, 0, 0, 0, 0xFFFF, 0, 0xFFFF);

    dump_vram();
    if (g_cl_ready) clEnqueueReadBuffer(g_cl_queue, g_cl_vram, CL_TRUE, 0, VRAM_SIZE, g_vram_gpu.data(), 0, nullptr, nullptr);
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
        if (buffer[offset + i + 3] == 0x00) {
            buffer[offset + i + 3] = 0xFF;
        }
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
        "1. Basic Fill (Red)", 
        "2. Basic Copy (Blue)", 
        "3. ROP XOR (Blue ^ Yellow)",
        "4. Clipping (Scissors)",
        "5. Hardware Cursor",
        "6. CRTC (Video Mode 800x600)",
        "7. CCE (Command Buffer)"
    };

    const char* hard_test_names[] = { 
        "1. Draw Pixel Heart",
        "2. Landscape (Simple Picture)",
        "3. Crazy XOR & Clipping Chaos"
    };

    while (!glfwWindowShouldClose(g_window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("main", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        ImGui::Text("Connection: %s    Ping: %s    OpenCL: %s",
            g_connected ? "OK" : "FAIL",
            g_ping_ok   ? "OK" : "FAIL",
            g_cl_ready  ? "OK" : "FAIL");
        ImGui::Separator();

        if (ImGui::Button("Connect") && !g_connected) {
            g_connected = connect_emu();
            if (g_connected) g_ping_ok = send_ping();
        }
        ImGui::SameLine();
        if (ImGui::Button("Ping") && g_connected) {
            g_ping_ok = send_ping();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset") && g_connected) {
            send_reset();
            g_has_result = false;
        }

        if (ImGui::BeginTabBar("MainTabs")) {
            if (ImGui::BeginTabItem("Manual Tests")) {
                ImGui::SetNextItemWidth(300);
                ImGui::ListBox("Tests", &g_selected_test, test_names, IM_ARRAYSIZE(test_names), 7);
                ImGui::SameLine();
                if (ImGui::Button("RUN SELECTED TEST") && g_connected) {
                    g_time_emu = 0;
                    g_time_gpu = 0;
                    if (g_selected_test == 0) run_fill_test();
                    else if (g_selected_test == 1) run_copy_test();
                    else if (g_selected_test == 2) run_rop_test();
                    else if (g_selected_test == 3) run_clipping_test();
                    else if (g_selected_test == 4) run_cursor_test();
                    else if (g_selected_test == 5) run_crtc_test();
                    else if (g_selected_test == 6) run_cce_test();
                }
                ImGui::EndTabItem();
            }
            
            if (ImGui::BeginTabItem("TesterInYou")) {
                ImGui::Text("This test draws a composite window scene using Fill and BitBlt.");
                if (ImGui::Button("DRAW SCENE & VERIFY") && g_connected) {
                    g_time_emu = 0;
                    g_time_gpu = 0;
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
                    g_time_emu = 0;
                    g_time_gpu = 0;
                    if (g_selected_hard_test == 0) run_hard_test_1();
                    else if (g_selected_hard_test == 1) run_hard_test_2();
                    else if (g_selected_hard_test == 2) run_hard_test_3();
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
            if (g_test_passed) {
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "RESULT: PASS - Exact pixel match (0 diffs)");
            } else {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "RESULT: FAIL - %zu mismatching pixels", g_diff_pixels);
            }
        } else {
            ImGui::Text("RESULT: Waiting for test...");
        }

        if (g_time_emu > 0 || g_time_gpu > 0) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "TIME: Emulator = %lld ms | RX580 = %lld ms", g_time_emu, g_time_gpu);
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