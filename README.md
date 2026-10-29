# 🎮 Video Cards Emulator

A **paravirtualized video card emulator** that lets your own OS talk to emulated legacy GPUs — with bit-exact output verification on real hardware via OpenCL.

The emulator implements classic GPU architectures (starting with **ATI Rage 128**) in software. Guest OS drivers write to MMIO registers and VRAM as if talking to real hardware. The emulator processes those writes, renders to virtual framebuffer, and the host GPU (via OpenCL) verifies bit-exact equality.

Integration with QEMU is done through a custom PCI device (`vgpu-proxy`) that bridges guest MMIO to the emulator via shared memory (`/dev/shm`). No TCP. Sub-2ms latency.

---

## 📊 Supported Video Cards

| Name | VID:DID | 2D | 3D | QEMU | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ATI Rage 128** | `0x1002:0x5246` | ✅ | ✅ | ✅ | 0 diffs vs RX580 |
| **DragonGPU** (example) | `0xD4A6:0x0001` | ✅ | — | ✅ | Custom |

---

## ✨ Features

### Emulation core
- **2D engine** — Fill, Copy, BitBlt, ROP3 (XOR/AND/OR), clipping (scissors), latching of `DP_GUI_MASTER_CNTL`.
- **3D rasterizer** — Z-buffered triangles, barycentric coordinates, back-face culling.
- **Hardware cursor** — `CUR_OFFSET` / `CUR_POS` / `CUR_CNTL` register interface.
- **CRTC** — Video mode control (`CRTC_OFFSET`, `CRTC_PITCH`, `H_TOTAL`, `V_TOTAL`).
- **Palette (RAMDAC)** — Index/data registers with auto-increment.
- **CCE** — Command buffer parser for batched register writes (Packet0 format).
- **IRQ** — `GEN_INT_CNTL` / `GEN_INT_STATUS` / `GEN_INT_ACK` with proper state transitions.
- **PCI Configuration** — Full VID/DID/BARs/IRQ, visible in `lspci`.

### QEMU integration
- **`vgpu-proxy` PCI device** — Custom QEMU device that bridges guest to emulator via shared memory.
- **DisplaySurface** — Renders VRAM to QEMU window (SDL/GTK/VNC).
- **Dirty rect optimization** — Guest flushes only changed regions; QEMU updates only those. Idle CPU usage = 0.
- **Sub-2ms latency** on WSLg, ~60 FPS on native.

### `.VGPU` format
- **JSON-based video card definitions** — Describe PCI IDs, register map, operations.
- **Runtime loading** — `GenericCard` reads `.vgpu` at startup. No recompilation.
- **Web constructor** — Single-file HTML editor to build custom cards visually.
- **Extensible** — Add new operation types (`line`, `gradient`, `triangle`) with ~50 lines of C++.

### Verification
- **OpenCL tester** — Bit-exact comparison against real physical GPU.
- **Automated tests** — 14 scenarios: fill, copy, ROP XOR, clipping, cursor, CRTC, CCE, IRQ, PCI, VBIOS, palette, DMA, 3D triangle, textured scene.

---

## 🗺️ Roadmap

- [x] Core 2D engine (Fill, Copy, ROP3)
- [x] Clipping, hardware cursor
- [x] CRTC + video mode switching
- [x] CCE + IRQ
- [x] PCI Configuration Space + VBIOS stub
- [x] 3D triangle rasterizer with Z-buffer
- [x] QEMU `vgpu-proxy` device with shared memory
- [x] DisplaySurface + dirty rect rendering
- [x] `.VGPU` JSON format + `GenericCard`
- [x] Web constructor (HTML/JS)
- [ ] Vulkan Compute backend (replace OpenCL)
- [ ] IRQ via `eventfd` (replace polling)
- [ ] More AMD cards: Rage 128 Pro, Radeon 7500, etc.
- [ ] Palette-driven 8-bit mode
- [ ] Texturing (UV coordinates in 3D)
- [ ] 2D line drawing primitive

---

## 🛠️ Project Structure

```
vgpu/
├── common/
│   ├── proto.h                 # IPC protocol (legacy, for tester)
│   ├── vgpu_format.h           # .VGPU descriptor structs
│   └── vgpu_format.cpp         # JSON parser
├── emulator/
│   ├── main.cpp                # Menu: Start / Select VGPU / Exit
│   ├── qemu_proxy.cpp          # mpqemu TCP client (legacy)
│   ├── qemu_proxy.h
│   ├── ivcard.h                # IVideoCard interface
│   └── cards/
│       ├── ati/128/            # Hardcoded ATI Rage 128
│       │   ├── ati128.cpp
│       │   ├── ati128.h
│       │   └── ati128.vgpu
│       └── generic/            # JSON-driven card
│           ├── generic_card.cpp
│           └── generic_card.h
├── tester/                     # OpenCL verification tool
│   └── main.cpp
├── cards/                      # Installed .vgpu files
├── third_party/nlohmann/       # JSON library
├── qemu/                       # QEMU integration files
│   ├── README.md               # How to build QEMU
│   ├── vgpu.patch              # Apply to QEMU 8.2.0
│   └── vgpu_proxy.c            # Full source (reference)
├── vgpu-constructor.html       # Web editor
├── CMakeLists.txt
├── CMakeLists-linux.txt        # Linux-only build (no tester/OpenCL)
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Linux** (native or WSL2) — for QEMU integration
- **CMake** ≥ 3.20
- **C++17** compiler: GCC 13+, Clang 15+, or MinGW-w64
- Optional: **OpenCL** (AMD/NVIDIA drivers) — for tester only

### 1. Build the Emulator

```bash
git clone https://github.com/<you>/VideoCards-emulator.git
cd VideoCards-emulator
mkdir build-linux && cd build-linux
cp ../CMakeLists-linux.txt ../CMakeLists.txt
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Binary: `build-linux/emulator`

### 2. Build QEMU with `vgpu-proxy`

See [`qemu/README.md`](qemu/README.md) for full details. Summary:

```bash
git clone --depth 1 --branch v8.2.0 https://gitlab.com/qemu-project/qemu.git
cd qemu
git apply /path/to/vgpu/qemu/vgpu.patch

sudo apt install -y build-essential ninja-build pkg-config \
                    libglib2.0-dev libpixman-1-dev libsdl2-dev libslirp-dev

mkdir build && cd build
../configure --target-list=x86_64-softmmu --enable-sdl --disable-werror
make -j$(nproc)
```

Verify:

```bash
~/qemu/build/qemu-system-x86_64 -device help | grep vgpu
# → name "vgpu-proxy"
```

### 3. Run Guest OS

**Terminal 1** — QEMU:

```bash
truncate -s 64M /dev/shm/vgpu_vram
truncate -s 8K  /dev/shm/vgpu_mmio

~/qemu/build/qemu-system-x86_64 \
    -m 2048 \
    -vga none \
    -device vgpu-proxy,bus=pci.0,addr=0x6,\
vendor-id=0x1002,device-id=0x5246,\
vram-size=67108864,mmio-size=8192,width=1280,height=800,\
vram-shm=/dev/shm/vgpu_vram,mmio-shm=/dev/shm/vgpu_mmio \
    -cdrom your_os.iso
```

**Terminal 2** — Emulator:

```bash
cd ~/vgpu/build-linux
./emulator
# → 1. Start (QEMU shared memory)
# → 2. Select VGPU (to load .vgpu instead of hardcoded card)
```

Guest OS sees `ATI Rage 128 [1002:5246]` in `lspci`. Driver writes to MMIO. Emulator renders to VRAM. QEMU displays it.

---

## 📝 `.VGPU` Format

A `.vgpu` file is a JSON descriptor of a video card. Loaded at runtime, no recompilation.

```json
{
  "format_version": 1,
  "name": "My Custom GPU",
  "pci": {
    "vendor_id": 54438,
    "device_id": 1,
    "class_code": 196608,
    "bars": [
      { "index": 0, "type": "vram", "size": 33554432, "prefetch": true },
      { "index": 2, "type": "mmio", "size": 8192, "prefetch": false }
    ]
  },
  "vram_size": 33554432,
  "mmio_size": 8192,
  "registers": {
    "FILL_TRIGGER": 280,
    "FILL_COLOR":   272,
    "FILL_WIDTH":   264,
    "FILL_HEIGHT":  268
  },
  "operations": [
    { "type": "fill", "trigger": "FILL_TRIGGER",
      "params": {
        "dst_offset": "FILL_OFFSET",
        "width": "FILL_WIDTH",
        "height": "FILL_HEIGHT",
        "color": "FILL_COLOR"
      }
    }
  ]
}
```

Full example: `emulator/cards/ati/128/ati128.vgpu`.

Build visually: open `vgpu-constructor.html` in browser → fill fields → **Download .VGPU** → drop into `~/vgpu/cards/`.

---

## 🧪 Verification

The `tester/` module runs identical operations on:

1. The emulator (software, via TCP IPC).
2. The real physical GPU (via OpenCL compute kernels).

Results compared pixel-by-pixel. `0 diffs` = bit-exact match.

Run:

```bash
cd build
./tester      # requires OpenCL
./emulator    # in another terminal
```

Test scenarios: basic fill, copy, ROP XOR, clipping, cursor, CRTC, CCE, IRQ, PCI, VBIOS, palette, DMA, 3D triangle, textured scene.

---

## 🎯 Verified Behavior

- ✅ **Bit-exact** 2D operations vs AMD RX580 via OpenCL
- ✅ **PCI enumeration** — `lspci` sees `1002:5246`
- ✅ **Shared memory** — no TCP bottleneck for VRAM
- ✅ **Sub-2ms input latency** on WSLg after dirty-rect optimization
- ✅ **Zero CPU on idle** — no updates if VRAM unchanged

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **BALATON Zoltan** — `ati-vga` in QEMU, primary reference for Rage 128 register behavior.
- **Linux kernel** — `r128_reg.h` and `r128_driver.c` for register definitions.
- **Bitsavers.org** — original ATI Rage 128 Register Reference Guide.
- **QEMU project** — educational device examples (`edu.c`, `bochs-display.c`).