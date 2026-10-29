#include "proto.h"
#include "ivcard.h"
#include "qemu_proxy.h"
#include "cards/ati/128/ati128.h"
#include "cards/generic/generic_card.h"
#include "../common/vgpu_format.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <string>

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#endif

using namespace vgpu;

static std::unique_ptr<IVideoCard> g_card;
static std::string g_current_name = "(built-in ATI Rage 128)";

static std::unique_ptr<IVideoCard> create_builtin() {
    auto ati = std::make_unique<ati128::ATI128Card>();
    ati->set_realistic_timing(false);
    return ati;
}

static std::unique_ptr<IVideoCard> load_vgpu(const std::string& path) {
    auto def = VgpuDefinition::load(path);
    printf("[menu] loaded: %s (VID=0x%04X DID=0x%04X vram=%llu mmio=%llu)\n",
           def.name.c_str(), def.pci.vendor_id, def.pci.device_id,
           (unsigned long long)def.vram_size, (unsigned long long)def.mmio_size);
    return std::make_unique<generic::GenericCard>(def);
}

#ifndef _WIN32
static bool run_shm_mode(const std::string& vram_path, const std::string& mmio_path) {
    printf("[shm] waiting for QEMU to create:\n");
    printf("[shm]   vram: %s\n", vram_path.c_str());
    printf("[shm]   mmio: %s\n", mmio_path.c_str());
    printf("[shm] press Ctrl+C to cancel\n");

    int fdv = -1, fdm = -1;
    int waited_ms = 0;
    while (true) {
        fdv = open(vram_path.c_str(), O_RDWR);
        fdm = open(mmio_path.c_str(), O_RDWR);

        if (fdv >= 0 && fdm >= 0) {
            // Проверим, что файлы уже нужного размера (QEMU truncate их)
            struct stat st_v, st_m;
            if (fstat(fdv, &st_v) == 0 && fstat(fdm, &st_m) == 0
                && st_v.st_size >= 64LL * 1024 * 1024
                && st_m.st_size >= 8192) {
                printf("[shm] files ready (vram=%lld, mmio=%lld)\n",
                       (long long)st_v.st_size, (long long)st_m.st_size);
                break;
            }
            // файлы есть, но ещё не готовы (QEMU в процессе)
            if (fdv >= 0) close(fdv);
            if (fdm >= 0) close(fdm);
            fdv = fdm = -1;
        } else {
            if (fdv >= 0) { close(fdv); fdv = -1; }
            if (fdm >= 0) { close(fdm); fdm = -1; }
        }

        if (waited_ms % 2000 == 0) {
            printf("[shm] still waiting (%d s)...\n", waited_ms / 1000);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        waited_ms += 500;
    }



    uint64_t vs = 64ULL * 1024 * 1024;
    uint64_t ms = 8192;
    uint8_t* vram = (uint8_t*)mmap(nullptr, vs, PROT_READ | PROT_WRITE, MAP_SHARED, fdv, 0);
    uint8_t* mmio = (uint8_t*)mmap(nullptr, ms, PROT_READ | PROT_WRITE, MAP_SHARED, fdm, 0);
    close(fdv);
    close(fdm);

    if (vram == MAP_FAILED || mmio == MAP_FAILED) {
        printf("ERROR: mmap failed\n");
        return false;
    }

    if (auto* ati = dynamic_cast<ati128::ATI128Card*>(g_card.get())) {
        ati->attach_shared(vram);
    } else if (auto* gen = dynamic_cast<generic::GenericCard*>(g_card.get())) {
        gen->attach_shared(vram);
    } else {
        printf("ERROR: card type does not support SHM\n");
        return false;
    }

    printf("[shm] attached vram=%p mmio=%p\n", vram, mmio);
    printf("[shm] waiting for guest writes to MMIO...\n");

    std::vector<uint32_t> last_mmio(ms / 4, 0);
    uint64_t last_vram_hash = 0;
    auto last_log = std::chrono::steady_clock::now();
    int mmio_writes = 0, triggers = 0;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        for (uint32_t off = 0; off < ms; off += 4) {
            uint32_t val = *(uint32_t*)(mmio + off);
            if (val != last_mmio[off / 4]) {
                last_mmio[off / 4] = val;
                g_card->reg_write(off, val);
                mmio_writes++;
                if (off == 0x1438) triggers++;
            }
        }
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log).count() >= 1000) {
            last_log = now;
            uint64_t hash = 0;
            for (size_t i = 0; i < vs; i += 4096) {
                hash ^= *reinterpret_cast<uint32_t*>(vram + i);
            }
            const char* status = (hash != last_vram_hash) ? "VRAM CHANGED" : "idle";
            printf("[shm] 1s: %d MMIO, %d cmd, %s\n", mmio_writes, triggers, status);
            last_vram_hash = hash;
            mmio_writes = 0;
            triggers = 0;
        }
    }
    return true;
}
#endif

static std::vector<std::string> list_vgpu_files() {
    std::vector<std::string> out;
#ifndef _WIN32
    const char* home = std::getenv("HOME");
    std::string dir = home ? std::string(home) + "/vgpu/cards" : "cards";
    DIR* d = opendir(dir.c_str());
    if (!d) {
        // fallback: ищем в текущей папке cards/
        d = opendir("cards");
        dir = "cards";
        if (!d) return out;
    }
    struct dirent* ent;
    while ((ent = readdir(d))) {
        std::string name = ent->d_name;
        if (name.size() > 5 && name.substr(name.size() - 5) == ".vgpu") {
            out.push_back(dir + "/" + name);
        }
    }
    closedir(d);
#endif
    return out;
}

static void select_vgpu() {
    auto files = list_vgpu_files();
    if (files.empty()) {
        printf("\n[menu] no .vgpu files found (looking in ~/vgpu/cards/ and ./cards/)\n");
        return;
    }
    printf("\n[menu] Available .vgpu cards:\n");
    for (size_t i = 0; i < files.size(); ++i)
        printf("  [%zu] %s\n", i + 1, files[i].c_str());
    printf("  [0] Cancel\n");
    printf("Choice: ");

    int c = 0;
    if (scanf("%d", &c) != 1) { while (getchar() != '\n'); return; }
    if (c < 1 || c > (int)files.size()) return;

    try {
        g_card = load_vgpu(files[c - 1]);
        g_current_name = files[c - 1];
        printf("[menu] loaded: %s\n", g_current_name.c_str());
        printf("[menu] NOTE: запусти QEMU с --device vgpu-proxy,\n");
        printf("       VID/DID взяты из .vgpu файла\n");
    } catch (std::exception& e) {
        printf("[menu] ERROR loading .vgpu: %s\n", e.what());
    }
}

static void interactive_menu() {
    if (!g_card) g_card = create_builtin();

    while (true) {
        printf("\n=== VGPU Emulator ===\n");
        printf("Current card: %s\n", g_current_name.c_str());
        printf("  1. Start (QEMU shared memory)\n");
        printf("  2. Select VGPU card (.vgpu file)\n");
        printf("  3. Exit\n");
        printf("Choice: ");

        int c = 0;
        if (scanf("%d", &c) != 1) { while (getchar() != '\n'); c = 0; }
        if (c == 1) {
#ifndef _WIN32
            run_shm_mode("/dev/shm/vgpu_vram", "/dev/shm/vgpu_mmio");
#else
            printf("[menu] SHM mode is Linux-only\n");
#endif
        } else if (c == 2) {
            select_vgpu();
        } else if (c == 3) {
            return;
        }
    }
}

int main(int argc, char** argv) {
#ifndef _WIN32
    if (argc >= 2 && std::strcmp(argv[1], "--shm") == 0) {
        std::string vgpu_path;
        std::string vp = "/dev/shm/vgpu_vram";
        std::string mp = "/dev/shm/vgpu_mmio";

        for (int i = 2; i < argc - 1; ++i) {
            if (std::strcmp(argv[i], "--vgpu") == 0) vgpu_path = argv[i + 1];
            if (std::strcmp(argv[i], "--vram") == 0) vp = argv[i + 1];
            if (std::strcmp(argv[i], "--mmio") == 0) mp = argv[i + 1];
        }
        try {
            if (!vgpu_path.empty()) {
                g_card = load_vgpu(vgpu_path);
                g_current_name = vgpu_path;
            } else {
                g_card = create_builtin();
            }
        } catch (std::exception& e) {
            printf("ERROR: %s\n", e.what());
            return 1;
        }
        run_shm_mode(vp, mp);
        return 0;
    }
#endif

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

    interactive_menu();
    return 0;
}