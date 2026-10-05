<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-light.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/logo-dark.svg">
  <img src="./assets/logo-light.svg" alt="Shrinkwrap" width="420">
</picture>

<br>

**Physical prototyping for PCB designs.**

</div>

<br>

## About

Shrinkwrap is a tool for converting PCB designs into simplified, 3D-printable physical models.

It is built around a straightforward idea: a PCB should be mechanically testable before it is manufactured. Shrinkwrap creates a printable representation of the board that can be used during enclosure design, assembly planning, and mechanical validation.

Rather than reproducing the complete PCB model, Shrinkwrap extracts and preserves the geometry that matters physically — board shape, mounting features, component placement, connector positions, and component height.

The resulting model is intentionally simplified for practical 3D printing, making it possible to produce a physical representation of a board early in the design process.

## Geometry

Shrinkwrap focuses on the mechanical envelope of the PCB.

The generated model preserves:

- board outline and dimensions
- cutouts and mounting holes
- component position and height
- connector geometry and placement
- overall PCB mechanical envelope

Complex component geometry is reduced where possible, keeping the model lightweight and suitable for physical prototyping without unnecessary detail.

## Project Status

Shrinkwrap is under active development. The current focus is PCB geometry processing, reliable model generation, and producing clean geometry suitable for 3D printing.