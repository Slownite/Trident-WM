==========
trident-wm
==========

*Slownite World Model: Perception, Memory, and Control*

.. image:: https://img.shields.io/badge/GitHub-Slownite-black?logo=github
   :target: https://github.com/Slownite/trident-wm
.. image:: https://img.shields.io/badge/Architecture-V--M--C-blue
.. image:: https://img.shields.io/badge/Framework-PyTorch_Lightning-792ee5
.. image:: https://img.shields.io/badge/Environment-Nix-7ebd26

Overview
========

**trident-wm** is a high-performance World Model implementation designed for robotics tasks like Push-T. It modularizes intelligence into three distinct pillars:

* **Vision (V)**: A frozen perception backbone (DINOv2) for robust latent state extraction.
* **Memory (M)**: A Transformer-based dynamics model optimized with custom **Triton** kernels for latent transitions.
* **Controller (C)**: A lightweight policy layer for action generation based on imagined trajectories.

Project Structure
=================

::

    .
    ├── Justfile           # Task runner (setup, train, clean)
    ├── flake.nix          # Pinned NixOS 25.11 environment
    ├── pyproject.toml     # Build metadata (Hatchling + uv)
    ├── configs/           # YAML hyperparameters
    └── src/
        └── trident_wm/    # Core package
            ├── pillars/   # V, M, and C neural modules
            ├── system.py  # LightningModule orchestration
            └── cli.py     # CLI Entrypoint

Installation
============

Local Development (CPU)
-----------------------

.. code-block:: bash

    # Enter Nix shell
    nix develop --impure

    # Initialize venv and install editable package
    just setup

RunPod / Cloud GPU (Ubuntu)
---------------------------

.. code-block:: bash

    # 1. Install Determinate Nix
    just install-nix

    # 2. Map drivers and install GPU dependencies
    just setup-gpu

Usage
=====

Training
--------

To start training the World Model using the parameters defined in the config:

.. code-block:: bash

    just train

Running Tests
-------------

To verify Triton kernels and neural architecture shapes:

.. code-block:: bash

    just test

Development
-----------

To enter the environment manually for debugging:

.. code-block:: bash

    just dev

Architecture
============



The system leverages **PyTorch Lightning** for device-agnostic training, allowing seamless transitions between a local development PC and high-performance cloud GPUs. Training progress and latent "imagination" visualizations are logged via **Weights & Biases**.

License
=======

This project is licensed under the MIT License - see the LICENSE file for details.
