# ShrinkWrap

ShrinkWrap is a lightweight tool for converting detailed PCB assembly models into simplified, 3D-printable envelope geometry.

Instead of preserving every resistor lead, connector detail, solder joint, and component feature, ShrinkWrap generates a smooth outer surface around the PCB assembly — similar to stretching a sheet of fabric over the board.

The goal is to preserve the dimensions and overall volume of a PCB assembly while dramatically reducing geometric complexity.

## Features

- Import STL, OBJ, PLY, GLB and GLTF models
- Interactive OpenGL 3D viewport
- Orbit, pan and zoom controls
- Automatic XY height-map generation
- Top and bottom PCB envelope generation
- Preserves the overall thickness of the assembly
- Adjustable sampling resolution
- Adjustable gap bridging
- Adjustable cloth slope
- Surface smoothing
- Configurable clearance
- Wireframe preview
- Switch between original and generated geometry
- Export generated geometry to STL, OBJ, PLY or GLB

## How it works

ShrinkWrap converts the input model into two height fields:

```text
                    TOP ENVELOPE
                 ____--------____
              __/                \__
             /                      \
============ PCB ASSEMBLY ============
             \__                  __/
                \____--------____/
                   BOTTOM ENVELOPE
