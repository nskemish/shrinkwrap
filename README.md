<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/logo-light.svg">
</picture>

<br>

**Turn PCB designs into quick, 3D-printable physical prototypes.**

</div>

## About

Shrinkwrap is a tool for quickly turning PCB designs into simplified, 3D-printable models.

The goal is simple: **print the PCB before you manufacture it.**

Instead of waiting for fabricated boards to check whether a design physically fits, Shrinkwrap generates a model that can be printed in minutes and used as a physical reference during development.

The generated model preserves the geometry that matters for mechanical verification, including board dimensions, mounting holes, component placement and component height.

## Features

- Fast PCB-to-3D workflow
- Simplified geometry optimized for 3D printing
- Board outline and mounting-hole reproduction
- Component position and height representation
- Interactive 3D preview
- 3D-printable model export
- Physical fit and clearance verification

## Use Cases

Shrinkwrap was designed around **rapid physical prototyping of PCBs**.

A PCB model can be generated and sent to a 3D printer early in the design process, providing a physical representation of the board before committing to fabrication.

This makes it useful for quickly checking:

- enclosure fit
- board dimensions
- mounting-hole alignment
- connector placement
- component clearances
- mechanical interaction with other parts

The printed model is not intended to reproduce the electrical functionality of the PCB. It provides a fast and inexpensive way to verify its **physical form** before manufacturing the real board.

## Development

Shrinkwrap is currently under active development.

Built with **Python, PySide6, OpenGL and native C++ components**.