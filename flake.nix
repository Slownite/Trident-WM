{
  description = "trident-WM: Perception, Memory, Controller architecture";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
        hasGpu = pkgs.stdenv.isLinux && (builtins.pathExists /dev/nvidia0);
        runtimeLibs = with pkgs; [ stdenv.cc.cc.lib zlib libGL ffmpeg ]
          ++ (if hasGpu then [ pkgs.linuxPackages.nvidia_x11 ] else []);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [ uv python312 pkg-config ]
            ++ (if hasGpu then [ cudaPackages.cudatoolkit ] else []);

          shellHook = ''
            if [ ! -d ".venv" ]; then uv venv; fi
            source .venv/bin/activate
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}${if hasGpu then ":/run/opengl-driver/lib" else ""}:$LD_LIBRARY_PATH"
            echo "🔱 trident-WM Dev Shell Active"
          '';
        };
      });
}
