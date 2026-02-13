set shell := ["bash", "-c"]

# 1. Install Nix (Run this first on a fresh RunPod instance)
install-nix:
    curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

# 2. Setup GPU Environment (Handles the RunPod driver "glitch")
setup-gpu:
    @echo "🔧 Mapping RunPod drivers to Nix-visible paths..."
    sudo mkdir -p /run/opengl-driver/lib
    sudo find /usr/lib/x86_64-linux-gnu -name 'libcuda.so*' -exec ln -sf {} /run/opengl-driver/lib/ \;
    @echo "📦 Installing GPU-enabled environment..."
    nix develop --impure --command bash -c "uv venv && uv pip install -e .[gpu]"

# 3. Clean: The "Nuclear Option"
clean:
    @echo "🗑️ Cleaning project artifacts..."
    rm -rf .venv/ build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo "✅ Project cleaned."

# 4. Core Sprint Commands
train args="":
    nix develop --impure --command trident train --data ./push_t

evaluate ckpt="latest.pt":
    nix develop --impure --command trident evaluate --checkpoint {{ckpt}}

test:
    nix develop --impure --command pytest tests/

dev *args:
    nix develop --impure --command {{if args == "" { "bash" } else { args }}}