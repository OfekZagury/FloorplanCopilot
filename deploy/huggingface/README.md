---
title: FloorplanCopilot
emoji: 🏠
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Describe an apartment, furnish it, export an AutoCAD DXF
models:
  - stable-diffusion-v1-5/stable-diffusion-v1-5
  - lllyasviel/control_v11p_sd15_seg
  - lllyasviel/control_v11p_sd15_canny
---

# FloorplanCopilot: "Cursor for Architects"

Describe the apartment you want in plain English, pick the best match from ~17,000 real floor plans, resize it to your target area, furnish it in the editor and download a layered **DXF file that opens in AutoCAD**. After export, an optional Stable Diffusion + ControlNet render visualises the furnished plan.

**How to use**
1. Describe a layout (e.g. *"3 bedrooms, 2 bathrooms, open kitchen next to the living room"*) and optionally set room filters, then press **Search**.
2. Click a candidate to preview it, set a target area if you like, and press **Select this plan**.
3. Move, add or delete furniture on the canvas, then press **Done** to export the DXF (and the render).

Source code, architecture and tests: [github.com/OfekZagury/FloorplanCopilot](https://github.com/OfekZagury/FloorplanCopilot)

Floor plan data: [ResPlan](https://github.com/m-agour/ResPlan) by Abouagour & Garyfallidis, licensed CC BY 4.0. Exported files are deleted from the server after an hour.
