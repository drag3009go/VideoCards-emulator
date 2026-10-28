# 🎮 Vidio Cards Emulator

A high-performance hardware emulator focused on recreating legacy graphics processing units. The project aims to bring classic GPU architectures back to life through software emulation.

---

## 📊 Supported Video Cards

| Name | Number | Working | State |
| :--- | :--- | :--- | :--- |
| **ATI Rage 128** | 1 | 🚀 90% | 🛠️ Devolopend |

---

## ✨ Features

* **OpenCL Tester** — Real-time performance and behavior verification by comparing the emulated GPU output against a real physical graphics card using OpenCL.
* **Modular Codebase** — Clean architecture with a dedicated vendor core and separate cards implementation.

---

## 🗺️ Roadmap

* [ ] **QEMU Support** — Integration with QEMU for complete system emulation.
* [ ] **Add Other AMD Video Cards** — Expanding the catalog with more classic architectures from AMD/ATI.

---

## 🛠️ Project Structure

* **`common/`** — Shared core utilities, math libraries, and cross-platform definitions.
* **`emulator/`** — Core logic containing `core.cpp`, `main.cpp`, and the `cards/ati/128` implementation.
* **`tester/`** — Automated test suites, OpenCL validation tools, and verification framework.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have **CMake** and a modern **C++ compiler** installed on your system.

### Building the Project
1. Create a build directory and navigate into it:
   ```bash
   mkdir build && cd build
   ```
2. Generate build files using CMake:
   ```bash
   cmake ..
   ```
3. Compile the emulator:
   ```bash
   cmake --build .
   ```

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
