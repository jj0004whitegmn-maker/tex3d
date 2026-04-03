<div align="center">

<img src="https://img.shields.io/badge/arXiv-2604.01618-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv">
<img src="https://img.shields.io/badge/ACM_MM-2026-blue?style=for-the-badge" alt="ACM MM 2026">
<img src="https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python" alt="Python">
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">

# 🎯 Tex3D

## Objects as Attack Surfaces via Adversarial 3D Textures for Vision-Language-Action Models

<p align="center">
  <strong>Jiawei Chen*</strong><sup>1,2</sup> &nbsp;·&nbsp;
  <strong>Simin Huang*</strong><sup>1</sup> &nbsp;·&nbsp;
  <strong>Jiawei Du</strong><sup>3</sup> &nbsp;·&nbsp;
  <strong>Shuaihang Chen</strong><sup>2,5</sup> &nbsp;·&nbsp;
  <strong>Yu Tian</strong><sup>4</sup> &nbsp;·&nbsp;
  <strong>Mingjie Wei</strong><sup>2,5</sup> &nbsp;·&nbsp;
  <strong>Chao Yu†</strong><sup>4</sup> &nbsp;·&nbsp;
  <strong>Zhaoxia Yin†</strong><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>East China Normal University &nbsp;·&nbsp;
  <sup>2</sup>Zhongguancun Academy &nbsp;·&nbsp;
  <sup>3</sup>CFAR, A*STAR, Singapore<br>
  <sup>4</sup>Tsinghua University &nbsp;·&nbsp;
  <sup>5</sup>Harbin Institute of Technology
</p>

<p align="center">
  <em>* Equal contribution &nbsp; † Corresponding authors</em>
</p>

<p align="center">
  <a href="https://vla-attack.github.io/tex3d">🌐 Project Page</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2604.01618">📄 Paper</a> &nbsp;|&nbsp;
  <a href="#citation">📖 Citation</a>
</p>

---

</div>

## 📌 Abstract

Vision-Language-Action (VLA) models have shown strong performance in robotic manipulation, yet their robustness to physically realizable adversarial attacks remains underexplored. Existing studies reveal vulnerabilities through language perturbations and 2D visual attacks, but these attack surfaces are either less representative of real deployment or limited in physical realism.

**Tex3D** is the **first framework** for end-to-end optimization of adversarial 3D textures directly within the VLA simulation environment. By introducing two core techniques:

- 🔧 **Foreground-Background Decoupling (FBD)** — establishes a differentiable optimization path from VLA objectives back to object textures via dual-renderer alignment
- 🎯 **Trajectory-Aware Adversarial Optimization (TAAO)** — prioritizes behaviorally critical frames via latent dynamics and stabilizes optimization with vertex-based parameterization

Experiments across simulation and real-robot settings achieve task **failure rates of up to 96.7%**, exposing critical vulnerabilities of VLA systems to physically grounded 3D adversarial attacks.

---

## 🆚 Comparison with Prior Attack Paradigms

| Attack Type | Interface | Physical Grounding | View Robustness | Perceptibility |
|:-----------:|:---------:|:-----------------:|:---------------:|:--------------:|
| Language-Based | Language-dependent | ❌ Limited | ❌ | ✅ Low |
| 2D Patch-Based | Visual front-end | ⚠️ Moderate | ❌ View-specific | ❌ High |
| **Tex3D (Ours)** | Object-centric | ✅ Strong | ✅ Multi-view | ✅ Naturalistic |

---

## 🏗️ Framework Overview

```
Tex3D Framework
├── Foreground-Background Decoupling (FBD)
│   ├── Environmental background rendering (MuJoCo)
│   ├── Target foreground rendering (Nvdiffrast)
│   ├── Cross-renderer geometric alignment (MVP transform)
│   ├── Lighting alignment (Ia, Id, ρ)
│   └── Scene compositing
│
└── Trajectory-Aware Adversarial Optimization (TAAO)
    ├── Latent dynamics-guided frame weighting
    │   ├── Visual encoder feature extraction
    │   ├── Latent velocity & acceleration (central differences)
    │   └── Criticality scoring + temperature-scaled softmax
    ├── Vertex-based texture parameterization
    ├── Untargeted attack objective
    ├── Targeted attack objective
    └── EoT for physical-world transfer
```

---

## 📁 Repository Structure

```
tex3d/
├── experiments/
│   └── robot/                      # Robot experiment scripts
│       ├── __pycache__/
│       ├── bridge/                 # Bridge robot experiments
│       └── libero/                 # LIBERO benchmark experiments
│           ├── __pycache__/
│           ├── attack_oft.py       # Attack script for OpenVLA-OFT
│           ├── attack_openvla.py   # Attack script for OpenVLA
│           ├── attack_pi.py        # Attack script for π0
│           ├── attack_pi05.py      # Attack script for π0.5
│           ├── openvla_utils.py    # OpenVLA utility functions
│           └── robot_utils.py      # Common robot utility functions
│
├── image/                          # Images for project page / paper
├── scripts/                        # Helper scripts
├── video/                          # Demo videos
│
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile
├── index.html                      # Project page
├── pyproject.toml                  # Python project configuration
└── requirements-min.txt            # Minimal dependencies
```

---

## ⚙️ Environment Setup

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (tested on NVIDIA A100 80GB)
- [MuJoCo](https://github.com/google-deepmind/mujoco) 2.3+

### Step 1 — Clone the Repository

```bash
git clone https://github.com/vla-attack/tex3d.git
cd tex3d
```

### Step 2 — Create a Virtual Environment

```bash
conda create -n tex3d python=3.10 -y
conda activate tex3d
```

### Step 3 — Install Core Dependencies

```bash
pip install -r requirements-min.txt
```

### Step 4 — Install Nvdiffrast (Differentiable Renderer)

```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

### Step 5 — Install LIBERO Benchmark

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
```

### Step 6 — Install VLA Models

<details>
<summary><strong>OpenVLA</strong></summary>

```bash
pip install git+https://github.com/openvla/openvla.git
```
</details>

<details>
<summary><strong>OpenVLA-OFT</strong></summary>

```bash
# Follow installation at: https://github.com/moojink/openvla-oft
pip install transformers accelerate
```
</details>

<details>
<summary><strong>π0 / π0.5</strong></summary>

```bash
# Install Physical Intelligence's openpi
pip install git+https://github.com/Physical-Intelligence/openpi.git
```
</details>

---

## 🚀 Quick Start

### Untargeted Attack on OpenVLA (LIBERO)

```bash
python experiments/robot/libero/attack_openvla.py \
    --task_suite spatial \
    --attack_type untargeted \
    --n_views 4 \
    --n_epochs 50 \
    --tau 0.1 \
    --output_dir outputs/openvla_spatial_untargeted
```

### Targeted Attack on OpenVLA (LIBERO)

```bash
python experiments/robot/libero/attack_openvla.py \
    --task_suite spatial \
    --attack_type targeted \
    --target_task "place_in_basket" \
    --n_views 4 \
    --n_epochs 50 \
    --tau 0.1 \
    --output_dir outputs/openvla_spatial_targeted
```

### Attack on OpenVLA-OFT

```bash
python experiments/robot/libero/attack_oft.py \
    --task_suite object \
    --attack_type untargeted \
    --perturbation_level L3 \
    --output_dir outputs/oft_object_untargeted
```

### Attack on π0

```bash
python experiments/robot/libero/attack_pi.py \
    --task_suite goal \
    --attack_type untargeted \
    --output_dir outputs/pi0_goal_untargeted
```

### Attack on π0.5

```bash
python experiments/robot/libero/attack_pi05.py \
    --task_suite long \
    --attack_type targeted \
    --output_dir outputs/pi05_long_targeted
```

---

## 📊 Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--task_suite` | LIBERO task suite: `spatial`, `object`, `goal`, `long` | `spatial` |
| `--attack_type` | Attack mode: `untargeted` or `targeted` | `untargeted` |
| `--perturbation_level` | Budget level: `L0`, `L1`, `L2`, `L3` | `L3` |
| `--n_views` | Number of sampled viewpoints per step | `4` |
| `--n_epochs` | Optimization epochs | `50` |
| `--tau` | Temperature for TAAO frame weighting softmax | `0.1` |
| `--use_eot` | Enable Expectation over Transformations for physical attack | `False` |
| `--output_dir` | Directory to save adversarial textures and logs | `outputs/` |

---

## 📈 Main Results

Task failure rates (%) on LIBERO benchmark (higher = stronger attack):

| Model | Clean | Gaussian | Tex3D (Untargeted) | Tex3D (Targeted) |
|-------|:-----:|:--------:|:------------------:|:----------------:|
| OpenVLA | 24.1 | 31.1 | **88.1** | **90.5** |
| OpenVLA-OFT | 4.7 | 6.5 | **76.0** | **79.3** |
| π0 | 4.6 | 10.7 | **71.8** | **73.3** |
| π0.5 | 2.8 | 7.4 | **69.3** | **71.2** |

Peak performance: **96.7%** failure rate on OpenVLA Spatial (targeted attack).

---

## 🧪 Evaluation Protocol

Each task is executed for **50 independent trials**. Performance is measured by **Task Failure Rate (FR)** — proportion of failed task completions over all trials.

```bash
# Run clean evaluation baseline
python experiments/robot/libero/attack_openvla.py \
    --attack_type none \
    --task_suite spatial \
    --n_trials 50
```

---

## 🌍 Physical World Experiments

For real-robot deployment on a **Franka Emika Panda** manipulator:

1. Print adversarial objects using a 3D printer (STL files provided in `experiments/robot/`)
2. Set up a monocular RGB camera
3. Run with EoT enabled for sim-to-real transfer:

```bash
python experiments/robot/libero/attack_openvla.py \
    --use_eot True \
    --eot_brightness 0.3 \
    --eot_contrast 0.3 \
    --eot_blur_radius 2 \
    --task_suite spatial \
    --output_dir outputs/physical_world
```

---

## 📐 Perturbation Levels

| Level | ε | Additional Constraint | Notes |
|-------|---|----------------------|-------|
| L0 | 64/255 | + MSE loss (naturalness) | Near-imperceptible, stealthy |
| L1 | 16/255 | — | Low budget |
| L2 | 32/255 | — | Medium budget |
| L3 | 64/255 | — | Full budget (default) |

---

## 🔬 Ablation Study Summary

| FBD Components | TAAO Weighting | Failure Rate | Step Time |
|----------------|---------------|:------------:|:---------:|
| w/o MVP alignment | Dynamics | 65.8% | ~7.2s |
| w/o Lighting | Dynamics | 76.8% | ~7.2s |
| w/o Decoupling | Dynamics | 84.6% | ~24.8s |
| Full FBD | Random | 73.7% | ~7.2s |
| Full FBD | Uniform | 82.1% | ~7.2s |
| **Full FBD** | **Dynamics** | **88.1%** | **~7.2s** |

---

## 📖 Citation

If you find Tex3D useful for your research, please cite:

```bibtex
@article{chen2026tex3d,
  title     = {Tex3D: Objects as Attack Surfaces via Adversarial 3D Textures for Vision-Language-Action Models},
  author    = {Chen, Jiawei and Huang, Simin and Du, Jiawei and Chen, Shuaihang and Tian, Yu and Wei, Mingjie and Yu, Chao and Yin, Zhaoxia},
  journal   = {arXiv preprint arXiv:2604.01618},
  year      = {2026},
  note      = {ACM MM 2026}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

We thank the authors of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [OpenVLA](https://github.com/openvla/openvla), [Nvdiffrast](https://github.com/NVlabs/nvdiffrast), and [MuJoCo](https://github.com/google-deepmind/mujoco) for their open-source contributions that made this work possible.

---

<div align="center">
  <sub>⭐ If you find this work useful, please consider starring the repository!</sub>
</div>
