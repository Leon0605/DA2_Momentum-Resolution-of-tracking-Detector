# DA2 Momentum Resolution of a Tracking Detector

Data Analysis 2 Spring Semester 2026 – Group Project I  
University of Zurich

**Authors:** Mike Buder, Siro Petrini, Leon Schwager

## Overview

Simulation and evaluation of a simplified charged particle tracking detector. Charged particles are propagated through equally-spaced detection planes and a magnetic dipole. Using Monte Carlo methods and least-squares fits, we estimate the point of creation as well as the tracking and momentum resolutions of the particle's trajectory.

The experiment is split into two parts:
- **Tracking resolution:** five detection planes before the magnet, straight-line fit to reconstruct $x_0$ and $s_0$
- **Momentum resolution:** full setup with magnet and three additional planes, $p_T$ reconstructed from the bending angle

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python Project.py
```

At the bottom of `Project.py`, comment/uncomment the sections you want to run (tracking resolution, momentum resolution, or both).

## Detector Parameters

| Parameter | Value |
|-----------|-------|
| Detection planes (before magnet) | n = 5 |
| Plane spacing | Δz = 2 cm |
| Magnet length | L = 10 cm |
| Default magnetic field | B = 0.5 T |
| Cell width | 500 µm |
| Hit uncertainty | σ = cellWidth / √12 |

## Physics

Transverse momentum is reconstructed using an exact geometric expression derived from the circular arc geometry inside the magnet:

$$p_T = \frac{L \cdot q \cdot B}{\sin(\theta_\text{out}) - \sin(\theta_\text{in})}$$

where $\theta_i = \arctan(s_i)$. This avoids the systematic bias introduced by the small-angle approximation $\theta \approx L/\rho$.

## Key Findings

- Pull distributions for $x_0$ and $p_T$ are consistent with $\mathcal{N}(0, 1)$, confirming unbiased reconstruction and correct uncertainty estimation
- The pull width for $s_0$ occasionally fails the $\chi^2$ test, attributed to the non-Gaussian (uniform) nature of cell hit errors
- Higher $p_T$ degrades resolution due to smaller deflection angles and the resulting sensitivity to measurement noise
- Stronger magnetic fields improve momentum resolution by increasing the bending angle

## Files

- `Project.py` — main simulation script
- `DA2_Momentum_Resolution_of_tracking_Detector_Lab_Report.pdf` ( full lab report with derivations and results )
