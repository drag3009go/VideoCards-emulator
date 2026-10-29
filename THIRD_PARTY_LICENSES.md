\# Third-Party Licenses



This project uses the following open-source libraries and references:



\## nlohmann/json

\- \*\*License:\*\* MIT

\- \*\*Website:\*\* https://github.com/nlohmann/json

\- \*\*Usage:\*\* JSON parsing for `.vgpu` configuration files.



\## GLFW

\- \*\*License:\*\* zlib/libpng

\- \*\*Website:\*\* https://www.glfw.org/

\- \*\*Usage:\*\* Window and input handling in the `tester` application.



\## Dear ImGui

\- \*\*License:\*\* MIT

\- \*\*Website:\*\* https://github.com/ocornut/imgui

\- \*\*Usage:\*\* GUI for the `tester` application (dual-monitor verification).



\## QEMU (vgpu-proxy patch)

\- \*\*License:\*\* GPL-2.0 (for the QEMU patch itself)

\- \*\*Website:\*\* https://www.qemu.org/

\- \*\*Usage:\*\* The `vgpu-proxy` device is provided as a patch for QEMU. 

&#x20; The emulator core itself is independent and does not link against QEMU.



\## References (Not included in source)

\- \*\*Linux Kernel (`r128\_reg.h`, `r128\_driver.c`):\*\* Used as a reference for ATI Rage 128 register behavior (GPL-2.0).

\- \*\*QEMU (`ati-vga`):\*\* Used as a reference for Rage 128 emulation behavior (GPL-2.0).

\- \*\*Bitsavers.org:\*\* Original ATI Rage 128 Register Reference Guide.

