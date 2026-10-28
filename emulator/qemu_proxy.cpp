#include "qemu_proxy.h"
#include "proto.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace vgpu {

enum mpqemu_msg_type : uint64_t {
    MPQEMU_MSG_HANDSHAKE = 1,
    MPQEMU_MSG_CMD = 2,
    MPQEMU_MSG_IRQ = 3,
    MPQEMU_MSG_PCI_CONFIG_READ = 4,
    MPQEMU_MSG_PCI_CONFIG_WRITE = 5,
    MPQEMU_MSG_MMIO_READ = 6,
    MPQEMU_MSG_MMIO_WRITE = 7,
    MPQEMU_MSG_DEVICE_RESET = 8,
    MPQEMU_MSG_GET_CONFIG = 9,
    MPQEMU_MSG_READ_CONFIG = 10,
};

#pragma pack(push, 1)
struct mpqemu_msg_header {
    uint64_t type;
    uint64_t size;
};

struct mpqemu_msg_pci_access {
    uint64_t addr;
    uint64_t size;
    uint64_t value;
};

struct mpqemu_msg_mmio_access {
    uint64_t addr;
    uint64_t size;
    uint64_t value;
};

struct mpqemu_msg_irq {
    uint64_t irq;
};
#pragma pack(pop)

static bool send_all_q(socket_t s, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len > 0) {
        int n = (int)::send(s, p, (int)len, 0);
        if (n <= 0) return false;
        p += n;
        len -= (size_t)n;
    }
    return true;
}

static bool recv_all_q(socket_t s, void* data, size_t len) {
    char* p = static_cast<char*>(data);
    while (len > 0) {
        int n = (int)::recv(s, p, (int)len, 0);
        if (n <= 0) return false;
        p += n;
        len -= (size_t)n;
    }
    return true;
}

static void send_reply(socket_t s, uint64_t type, const void* payload, uint64_t size) {
    mpqemu_msg_header hdr{ type, size };
    send_all_q(s, &hdr, sizeof(hdr));
    if (size > 0) send_all_q(s, payload, (size_t)size);
}

static void send_irq(socket_t s, uint64_t irq) {
    mpqemu_msg_irq m{ irq };
    mpqemu_msg_header hdr{ MPQEMU_MSG_IRQ, sizeof(m) };
    send_all_q(s, &hdr, sizeof(hdr));
    send_all_q(s, &m, sizeof(m));
}

int run_qemu_proxy(std::unique_ptr<IVideoCard> card, const char* host, int port) {
#ifdef _WIN32
    WsaInit wsa;
#endif
    socket_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCK) { printf("qemu: socket fail\n"); return 1; }

    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, host, &a.sin_addr);

    printf("qemu: connecting to %s:%d...\n", host, port);
    if (connect(s, (sockaddr*)&a, sizeof(a)) < 0) {
        printf("qemu: connect fail\n");
        close_socket(s);
        return 1;
    }
    printf("qemu: connected\n");

    std::thread irq_thread([&]() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            while (card->has_pending_irq()) {
                uint32_t irq = card->pop_pending_irq();
                send_irq(s, irq);
            }
        }
    });
    irq_thread.detach();

    while (true) {
        mpqemu_msg_header hdr;
        if (!recv_all_q(s, &hdr, sizeof(hdr))) break;

        switch (hdr.type) {
            case MPQEMU_MSG_HANDSHAKE: {
                printf("qemu: HANDSHAKE received\n");
                uint64_t reply = 0;
                send_reply(s, MPQEMU_MSG_HANDSHAKE, &reply, sizeof(reply));
                break;
            }
            case MPQEMU_MSG_PCI_CONFIG_READ: {
                mpqemu_msg_pci_access m;
                if (hdr.size < sizeof(m)) { std::vector<uint8_t> d(hdr.size); recv_all_q(s, d.data(), d.size()); break; }
                recv_all_q(s, &m, sizeof(m));
                uint32_t val = card->pci_config_read((uint32_t)m.addr);
                printf("qemu: PCI READ @%llx -> %08x\n", (unsigned long long)m.addr, val);
                uint64_t reply = val;
                send_reply(s, MPQEMU_MSG_PCI_CONFIG_READ, &reply, sizeof(reply));
                break;
            }
            case MPQEMU_MSG_PCI_CONFIG_WRITE: {
                mpqemu_msg_pci_access m;
                if (hdr.size < sizeof(m)) { std::vector<uint8_t> d(hdr.size); recv_all_q(s, d.data(), d.size()); break; }
                recv_all_q(s, &m, sizeof(m));
                printf("qemu: PCI WRITE @%llx = %08x\n", (unsigned long long)m.addr, (uint32_t)m.value);
                break;
            }
            case MPQEMU_MSG_MMIO_READ: {
                mpqemu_msg_mmio_access m;
                if (hdr.size < sizeof(m)) { std::vector<uint8_t> d(hdr.size); recv_all_q(s, d.data(), d.size()); break; }
                recv_all_q(s, &m, sizeof(m));
                uint32_t val = card->reg_read((uint32_t)m.addr);
                printf("qemu: MMIO READ @%llx -> %08x\n", (unsigned long long)m.addr, val);
                uint64_t reply = val;
                send_reply(s, MPQEMU_MSG_MMIO_READ, &reply, sizeof(reply));
                break;
            }
            case MPQEMU_MSG_MMIO_WRITE: {
                mpqemu_msg_mmio_access m;
                if (hdr.size < sizeof(m)) { std::vector<uint8_t> d(hdr.size); recv_all_q(s, d.data(), d.size()); break; }
                recv_all_q(s, &m, sizeof(m));
                printf("qemu: MMIO WRITE @%llx = %08x\n", (unsigned long long)m.addr, (uint32_t)m.value);
                card->reg_write((uint32_t)m.addr, (uint32_t)m.value);
                break;
            }
            case MPQEMU_MSG_DEVICE_RESET: {
                printf("qemu: DEVICE RESET\n");
                card->reset();
                break;
            }
            default: {
                printf("qemu: unknown msg %llu size %llu\n", (unsigned long long)hdr.type, (unsigned long long)hdr.size);
                if (hdr.size > 0) {
                    std::vector<uint8_t> d(hdr.size);
                    recv_all_q(s, d.data(), d.size());
                }
                break;
            }
        }
    }

    printf("qemu: disconnected\n");
    close_socket(s);
    return 0;
}

}