set shell := ["bash", "-c"]

# Smart detection: Only use 'nix develop' if we aren't already in a Nix shell
nix_cmd := if env_var_or_default("IN_NIX_SHELL", "false") == "false" { "nix develop --impure --command " } else { "" }

# 1. Install Nix (Run this first on a fresh RunPod instance)
install-nix:
    curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

# Initialize GPU Environment with proper pathing
setup-gpu:
    @echo "🔧 Mapping RunPod drivers..."
    sudo mkdir -p /run/opengl-driver/lib
    sudo find /usr/lib/x86_64-linux-gnu -name 'libcuda.so*' -exec ln -sf {} /run/opengl-driver/lib/ \;
    @echo "📦 Initializing GPU environment inside Nix..."
    {{nix_cmd}} bash -c "uv venv && uv pip install -e .[gpu]"
# Utility to verify everything is linked correctly
check-gpu:
    {{nix_cmd}} python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
# 3. Development Installation
dev-install:
    @echo "🛠️ Installing trident-wm in editable mode..."
    {{nix_cmd}} uv pip install -e .[dev,gpu]

# 4. Clean
clean:
    @echo "🗑️ Cleaning project artifacts..."
    rm -rf .venv/ build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo "✅ Project cleaned."

# 5. Core Sprint Commands (Using the installed 'trident' command)
# Usage: just train config=configs/gpu_sprint.yaml device=gpu
train config="configs/config_cpu_debug.yaml" device="cpu":
    {{nix_cmd}} trident train --config {{config}} --device {{device}}

# Usage: just evaluate ckpt=checkpoints/model.ckpt config=configs/gpu_sprint.yaml
evaluate ckpt config="configs/config_gpu_sprint.yaml":
    {{nix_cmd}} trident evaluate --checkpoint {{ckpt}} --config {{config}}

# Usage: just test-shapes config=configs/config_cpu_test.yaml
test-shapes config="configs/config_cpu_test.yaml":
    {{nix_cmd}} trident test-shapes --config {{config}}

test:
    {{nix_cmd}} pytest src/trident_wm/tests/

dev *args:
    {{nix_cmd}} {{ if args == "" { "bash" } else { args } }}
