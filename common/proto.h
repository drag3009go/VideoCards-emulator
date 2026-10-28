#pragma once

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/select.h>
#endif

#include <cstdint>
#include <cstring>
#include <cstddef>

namespace vgpu {

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t INVALID_SOCK = INVALID_SOCKET;
#else
using socket_t = int;
constexpr socket_t INVALID_SOCK = -1;
#endif

constexpr uint16_t DEFAULT_PORT = 14880;
constexpr size_t VRAM_SIZE = 64ULL * 1024ULL * 1024ULL;
constexpr size_t MMIO_REGS = 0x2000;

enum class MsgType : uint32_t {
    Ping         = 1,
    Pong         = 2,
    Reset        = 3,
    RegWrite     = 4,
    RegRead      = 5,
    RegReadResp  = 6,
    DumpVram     = 7,
    DumpVramResp = 8,
    UploadVram   = 9,
    Irq          = 10,
    PciConfigRead  = 11,
    PciConfigResp  = 12,
};

#pragma pack(push, 1)
struct MsgHeader       { uint32_t type; uint32_t payload_len; };
struct PingMsg         { uint32_t token; };
struct PongMsg         { uint32_t token; };
struct RegWriteMsg     { uint32_t addr; uint32_t value; };
struct RegReadMsg      { uint32_t addr; };
struct RegReadRespMsg  { uint32_t addr; uint32_t value; };
struct DumpVramMsg     { uint32_t offset; uint32_t len; };
struct DumpVramRespHdr { uint32_t offset; uint32_t len; uint64_t exec_time_us; };
struct UploadVramMsg   { uint32_t offset; uint32_t len; };
struct IrqMsg          { uint32_t status; };
struct PciConfigReadMsg  { uint32_t reg; };
struct PciConfigRespMsg  { uint32_t reg; uint32_t value; };
#pragma pack(pop)

inline bool send_all(socket_t s, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len > 0) {
        int n = (int)::send(s, p, (int)len, 0);
        if (n <= 0) return false;
        p += n;
        len -= (size_t)n;
    }
    return true;
}

inline bool recv_all(socket_t s, void* data, size_t len) {
    char* p = static_cast<char*>(data);
    while (len > 0) {
        int n = (int)::recv(s, p, (int)len, 0);
        if (n <= 0) return false;
        p += n;
        len -= (size_t)n;
    }
    return true;
}

inline bool send_msg(socket_t s, MsgType type, const void* payload, uint32_t len) {
    MsgHeader hdr{ (uint32_t)type, len };
    if (!send_all(s, &hdr, sizeof(hdr))) return false;
    if (len > 0 && !send_all(s, payload, len)) return false;
    return true;
}

inline bool recv_msg(socket_t s, MsgType& type, void* buf, uint32_t buf_size, uint32_t& out_len) {
    MsgHeader hdr{};
    if (!recv_all(s, &hdr, sizeof(hdr))) return false;
    if (hdr.payload_len > buf_size) return false;
    if (hdr.payload_len > 0 && !recv_all(s, buf, hdr.payload_len)) return false;
    type = (MsgType)hdr.type;
    out_len = hdr.payload_len;
    return true;
}

inline void close_socket(socket_t s) {
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

#ifdef _WIN32
struct WsaInit {
    WsaInit()  { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); }
    ~WsaInit() { WSACleanup(); }
};
#endif

}