#include "proto.h"
#include "ivcard.h"
#include "qemu_proxy.h"
#include "cards/ati/128/ati128.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <memory>

using namespace vgpu;

int main(int argc, char** argv) {
    if (argc >= 4 && std::strcmp(argv[1], "--qemu") == 0) {
        const char* host = argv[2];
        int port = std::atoi(argv[3]);
        auto ati = std::make_unique<ati128::ATI128Card>();
        ati->set_realistic_timing(false);
        return run_qemu_proxy(std::move(ati), host, port);
    }

#ifdef _WIN32
    WsaInit wsa;
#endif

    int choice = 0;
    while (choice < 1 || choice > 2) {
        printf("\n=== Video Card Emulator ===\n");
        printf("1. ATI Rage 128 (Realistic 100 MHz)\n");
        printf("2. ATI Rage 128 (Turbo, Max Speed)\n");
        printf("Select video card: ");
        if (scanf("%d", &choice) != 1) {
            while (getchar() != '\n');
            choice = 0;
        }
    }

    std::unique_ptr<IVideoCard> card;
    if (choice == 1) {
        auto ati = std::make_unique<ati128::ATI128Card>();
        ati->set_realistic_timing(true);
        card = std::move(ati);
        printf("[EMU] Loaded: ATI Rage 128 (Realistic Timing)\n");
    } else {
        auto ati = std::make_unique<ati128::ATI128Card>();
        ati->set_realistic_timing(false);
        card = std::move(ati);
        printf("[EMU] Loaded: ATI Rage 128 (Turbo Mode)\n");
    }

    socket_t lsock = socket(AF_INET, SOCK_STREAM, 0);
    if (lsock == INVALID_SOCK) { printf("socket fail\n"); return 1; }

    int opt = 1;
    setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(DEFAULT_PORT);

    if (bind(lsock, (sockaddr*)&addr, sizeof(addr)) < 0) { printf("bind fail\n"); return 1; }
    if (listen(lsock, 1) < 0) { printf("listen fail\n"); return 1; }

    printf("emulator: listening on port %u\n", (unsigned)DEFAULT_PORT);
    fflush(stdout);

    socket_t client = accept(lsock, nullptr, nullptr);
    if (client == INVALID_SOCK) { printf("accept fail\n"); return 1; }
    printf("emulator: tester connected\n");
    fflush(stdout);

    std::vector<uint8_t> buf(VRAM_SIZE);
    MsgType type;
    uint32_t len;

    while (true) {
        while (card->has_pending_irq()) {
            uint32_t irq = card->pop_pending_irq();
            IrqMsg m{ irq };
            send_msg(client, MsgType::Irq, &m, sizeof(m));
            printf("emulator: IRQ sent (status=0x%08X)\n", irq);
            fflush(stdout);
        }

        if (!recv_msg(client, type, buf.data(), (uint32_t)buf.size(), len)) break;

        switch (type) {
            case MsgType::Ping: {
                PingMsg p; std::memcpy(&p, buf.data(), sizeof(p));
                send_msg(client, MsgType::Pong, &p, sizeof(p));
                break;
            }
            case MsgType::Reset: {
                card->reset();
                send_msg(client, MsgType::Pong, nullptr, 0);
                printf("emulator: RESET\n");
                fflush(stdout);
                break;
            }
            case MsgType::RegWrite: {
                RegWriteMsg r; std::memcpy(&r, buf.data(), sizeof(r));
                card->reg_write(r.addr, r.value);
                break;
            }
            case MsgType::RegRead: {
                RegReadMsg r; std::memcpy(&r, buf.data(), sizeof(r));
                RegReadRespMsg resp{ r.addr, card->reg_read(r.addr) };
                send_msg(client, MsgType::RegReadResp, &resp, sizeof(resp));
                break;
            }
            case MsgType::PciConfigRead: {
                PciConfigReadMsg r; std::memcpy(&r, buf.data(), sizeof(r));
                PciConfigRespMsg resp{ r.reg, card->pci_config_read(r.reg) };
                send_msg(client, MsgType::PciConfigResp, &resp, sizeof(resp));
                break;
            }
            case MsgType::DumpVram: {
                DumpVramMsg r; std::memcpy(&r, buf.data(), sizeof(r));
                std::vector<uint8_t> out(r.len);
                card->dump_vram(r.offset, r.len, out.data());
                DumpVramRespHdr resp{ r.offset, r.len, card->get_exec_time_us() };
                MsgHeader msg_hdr{ (uint32_t)MsgType::DumpVramResp, (uint32_t)(sizeof(resp) + r.len) };
                send_all(client, &msg_hdr, sizeof(msg_hdr));
                send_all(client, &resp, sizeof(resp));
                send_all(client, out.data(), r.len);
                break;
            }
            case MsgType::UploadVram: {
                UploadVramMsg r; std::memcpy(&r, buf.data(), sizeof(r));
                card->upload_vram(r.offset, r.len, buf.data() + sizeof(r));
                break;
            }
            default: break;
        }
    }

    close_socket(client);
    close_socket(lsock);
    return 0;
}