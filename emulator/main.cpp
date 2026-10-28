#include "proto.h"
#include "ivcard.h"
#include "cards/ati/128/ati128.h"
#include <cstdio>
#include <vector>
#include <memory>

using namespace vgpu;

int main() {
#ifdef _WIN32
    WsaInit wsa;
#endif
    std::unique_ptr<IVideoCard> card = std::make_unique<ati128::ATI128Card>();

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

    printf("emulator: listening on port %u (ATI Rage 128)\n", (unsigned)DEFAULT_PORT);
    fflush(stdout);

    socket_t client = accept(lsock, nullptr, nullptr);
    if (client == INVALID_SOCK) { printf("accept fail\n"); return 1; }
    printf("emulator: tester connected\n");
    fflush(stdout);

    std::vector<uint8_t> buf(VRAM_SIZE);
    MsgType type;
    uint32_t len;

    while (recv_msg(client, type, buf.data(), (uint32_t)buf.size(), len)) {
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
                printf("emulator: REG[%04x] <= %08x\n", r.addr, r.value);
                fflush(stdout);
                break;
            }
            case MsgType::RegRead: {
                RegReadMsg r; std::memcpy(&r, buf.data(), sizeof(r));
                RegReadRespMsg resp{ r.addr, card->reg_read(r.addr) };
                send_msg(client, MsgType::RegReadResp, &resp, sizeof(resp));
                break;
            }
            case MsgType::DumpVram: {
                DumpVramMsg r; std::memcpy(&r, buf.data(), sizeof(r));
                std::vector<uint8_t> out(r.len);
                card->dump_vram(r.offset, r.len, out.data());
                send_msg(client, MsgType::DumpVramResp, out.data(), r.len);
                printf("emulator: DUMP %u bytes @ %08x\n", r.len, r.offset);
                fflush(stdout);
                break;
            }
            case MsgType::UploadVram: {
                UploadVramMsg r; std::memcpy(&r, buf.data(), sizeof(r));
                card->upload_vram(r.offset, r.len, buf.data() + sizeof(r));
                printf("emulator: UPLOAD %u bytes to VRAM @ %08x\n", r.len, r.offset);
                fflush(stdout);
                break;
            }
            default:
                break;
        }
    }

    close_socket(client);
    close_socket(lsock);
    return 0;
}