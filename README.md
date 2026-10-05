<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-light.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/logo-dark.svg">
  <img src="./assets/logo-light.svg" alt="Shrinkwrap" width="420">
</picture>

<br>

**PCB-to-3D workflow for rapid physical prototyping and mechanical verification.**

</div>

<br>

## About

Shrinkwrap converts PCB designs into simplified, dimensionally representative 3D models optimized for rapid 3D printing.

It provides a fast way to produce a physical representation of a PCB before fabrication, allowing mechanical aspects of a design to be evaluated without waiting for manufactured boards.

> [!NOTE]
> **Shrinkwrap is designed to make PCB designs physically testable as early as possible.**
> Generate the model, print it, and verify the mechanical design before committing to PCB fabrication.

The generated geometry preserves the physical features relevant to mechanical validation, including board dimensions, mounting holes, component placement, component height, and connector positions.

## Features

- **PCB-to-3D conversion** — Generate printable geometry directly from PCB design data.
- **Print-optimized geometry** — Simplified models designed for fast and reliable 3D printing.
- **Board geometry** — Preserve board outlines, dimensions, cutouts, and mounting holes.
- **Component representation** — Reproduce component position and physical height.
- **Interactive preview** — Inspect the generated model before export.
- **3D model export** — Export geometry ready for slicing and physical prototyping.
- **Mechanical verification** — Validate fit, alignment, clearances, and mechanical constraints.

## Use Cases

Shrinkwrap is primarily intended for **rapid physical PCB prototyping before fabrication**.

A generated model can be sent directly to a 3D printer and used as a physical stand-in for the final PCB during mechanical development.

This allows early verification of:

- enclosure fit and internal clearances
- PCB dimensions and board outline
- mounting-hole position and alignment
- connector position and accessibility
- component placement and height
- interaction with brackets, panels, and other mechanical parts

> [!IMPORTANT]
> Shrinkwrap models represent the **physical geometry** of a PCB. They are intended for mechanical prototyping and verification, not electrical simulation or functional PCB replacement.

By moving mechanical validation ahead of fabrication, design errors can be identified before committing to a manufactured board.