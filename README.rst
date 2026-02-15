==========
trident-wm
==========

*Slownite World Model: Perception, Memory, and Imagination*

.. image:: https://img.shields.io/badge/GitHub-Slownite-black?logo=github
   :target: https://github.com/Slownite/trident-wm
.. image:: https://img.shields.io/badge/Architecture-V--M--D-blue
.. image:: https://img.shields.io/badge/Framework-PyTorch_Lightning-792ee5
.. image:: https://img.shields.io/badge/Data-LeRobot-green
.. image:: https://img.shields.io/badge/Environment-Nix-7ebd26

Overview
========

**trident-wm** is a high-performance World Model implementation designed for robotics tasks like Push-T. It modularizes intelligence into three distinct pillars to enable efficient latent-space imagination:

* **Vision (V)**: A perception backbone using frozen DINOv2 features with a trainable linear neck.
* **Memory (M)**: A causal Transformer-based dynamics model for predicting future latent transitions.
* **Decoder (D)**: A deconvolutional visual decoder that reconstructs imagined latents back into video pixels.

Project Structure
=================

::

    .
    ├── Justfile           # Task runner for setup, train, and evaluation
    ├── flake.nix          # Pinned NixOS environment for reproducibility
    ├── pyproject.toml     # Build metadata and CLI entrypoints
    ├── configs/           # YAML hyperparameters (CPU, GPU-Sprint, GPU-Heavy)
    └── src/
        └── trident_wm/
            ├── pillars/   # V, M, and D neural modules
            ├── system.py  # LightningModule (MSE Latent + Visual Loss)
            ├── datamodule.py # LeRobot dataset integration with automated resizing
            └── cli.py     # Click-based CLI for training and evaluation

Installation
============

Local Development (CPU)
-----------------------

.. code-block:: bash

    # Enter Nix shell
    nix develop --impure

    # Initialize venv and install package in editable mode
    just dev-install

RunPod / Cloud GPU (Ubuntu)
---------------------------

.. code-block:: bash

    # 1. Install Determinate Nix
    just install-nix

    # 2. Map drivers and install GPU dependencies
    just setup-gpu

Usage
=====

The system uses a YAML-driven CLI. Note that images are automatically resized to 224x224 via the DataModule to satisfy DINOv2 patch requirements.



Training
--------

To start training using the parameters defined in the config:

.. code-block:: bash

    # Default CPU testing
    just train

    # Cloud GPU training with specific config
    just train configs/config_gpu_sprint.yaml gpu

Evaluation
----------

Verify the model's imagination on the unseen test split:

.. code-block:: bash

    just evaluate checkpoints/model.ckpt configs/config_gpu_sprint.yaml

Visualizing Imagination
=======================



Training progress and latent "imagination" visualizations are logged via **Weights & Biases**. The system generates comparison videos showing the actual future frames from the LeRobot dataset versus the World Model's visual reconstruction of its predicted latent states.

License
=======

This project is licensed under the MIT License.
