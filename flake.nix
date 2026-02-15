{
  description = "trident-WM: Perception, Memory, Decoder architecture";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system; 
          config.allowUnfree = true; 
        };
        # Essential libraries for PyTorch and Vision tasks
        runtimeLibs = with pkgs; [ 
          stdenv.cc.cc.lib 
          zlib 
          libGL 
          glib
          ffmpeg 
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [ 
            uv 
            python312 
            pkg-config 
            linuxHeaders 
          ];

          shellHook = ''
            # Activate venv if it exists
            if [ -d ".venv" ]; then source .venv/bin/activate; fi
            
            # Critical: Bridge Nix libraries and RunPod's host GPU drivers
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            
            echo "🔱 trident-WM Dev Shell Active (RTX 6000 Ada Ready)"
          '';
        };
      });
}
