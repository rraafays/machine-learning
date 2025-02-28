{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = [
    pkgs.python312
    pkgs.python312Packages.pip
    (pkgs.writeShellScriptBin "notebook" ''
      #!${pkgs.stdenv.shell}
      jupyter-notebook "$@"
    '')
  ];
  shellHook = ''
    python -m venv .venv
    source .venv/bin/activate

    pip install numpy
    pip install pandas
    pip install matplotlib
    pip install seaborn
  '';
}
