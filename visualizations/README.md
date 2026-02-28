# Visualizations

Interactive geometric explorations of the MBD framework's state spaces.

---

## Stella Octangula: The Influence Cube

**File:** `visualize_stella.py`  
**Date:** 2026-02-16  
**Author:** Brandon Everett

The Stella Octangula is the dual compound of two tetrahedra inscribed in a unit cube. Here it maps the eight eigenstates of the Influence Cube: four constructive (Haven) vertices and four destructive (Sink) vertices, connected by their diagonal duals.

### The Space

Three binary axes:

| Axis | 0 | 1 |
|------|---|---|
| **X — Locus** | Internal | External |
| **Y — Coupling** | Low-κ | High-κ |
| **Z — Temporality** | Static | Dynamic |

Eight vertices at the corners:

**Haven tetrahedron** (gold):
- **Nature** — Internal, Low-κ, Static: innate character
- **Nurture** — External, High-κ, Static: stable bonds
- **Heaven** — Internal, High-κ, Dynamic: peak experience
- **Home** — External, Low-κ, Dynamic: safe harbor

**Sink tetrahedron** (crimson):
- **Displacement** — External, Low-κ, Static: uprootedness
- **Fixation** — Internal, High-κ, Static: rumination
- **Degeneration** — Internal, Low-κ, Dynamic: isolation spiral
- **Capture** — External, High-κ, Dynamic: enmeshment

Each Haven vertex is diagonally opposite its Sink dual — bitwise complement across the cube.

### What It Shows

An agent's centroid in this space is the weighted mean of all eight vertex pressures. The demo scenario traces a life arc: stable routine → friendship → creative breakthrough → sudden loss → isolation → unexpected kindness → safe harbor. The trajectory is rendered in real time as the novelty signal (the "poppit") drives baseline updates.

### Running It

This script requires the `dynamics/` module from [MBD-Framework](https://github.com/YellowHapax/MBD-Framework). Clone that repository alongside this one, or install it as a package.

```bash
# From MBD-Framework root (with venv active):
python Noetic-Lab/visualizations/visualize_stella.py

# Or from this directory, with MBD-Framework on sys.path:
python visualize_stella.py
```

**Controls:** Drag to rotate · Scroll to zoom · Hover vertices for descriptions

---

*The cube is necessary. The cube is not sufficient.*
