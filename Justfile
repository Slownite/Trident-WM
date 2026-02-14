set shell := ["bash", "-c"]

# Smart detection: Only use 'nix develop' if we aren't already in a Nix shell
nix_cmd := if env_var_or_default("IN_NIX_SHELL", "false") == "false" { "nix develop --impure --command " } else { "" }

# 1. Install Nix (Run this first on a fresh RunPod instance)
install-nix:
    curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

# 2. Setup GPU Environment (Handles the RunPod driver mapping)
setup-gpu:
    @echo "🔧 Mapping RunPod drivers to Nix-visible paths..."
    sudo mkdir -p /run/opengl-driver/lib
    sudo find /usr/lib/x86_64-linux-gnu -name 'libcuda.so*' -exec ln -sf {} /run/opengl-driver/lib/ \;
    @echo "📦 Initializing GPU environment..."
    {{nix_cmd}} uv venv && {{nix_cmd}} uv pip install -e .[gpu]

# 3. Development Installation (Hardware-aware editable install)
dev-install:
    @echo "🛠️ Installing trident-wm in editable mode with dev extras..."
    {{nix_cmd}} uv pip install -e .[dev,gpu]

# 4. Clean: The "Nuclear Option"
clean:
    @echo "🗑️ Cleaning project artifacts..."
    rm -rf .venv/ build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo "✅ Project cleaned."

# 5. Core Sprint Commands
train args="":
    {{nix_cmd}} trident fit --config configs/default.yaml {{args}}

evaluate ckpt="latest.ckpt":
    {{nix_cmd}} trident test --config configs/default.yaml --ckpt_path {{ckpt}}

test:
    {{nix_cmd}} pytest src/trident_wm/tests/

# 6. Utility Commands
check-gpu:
    {{nix_cmd}} python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Enter the dev shell or run a specific command (Defaults to bash)
dev *args:
    {{nix_cmd}} {{ if args == "" { "bash" } else { args } }}