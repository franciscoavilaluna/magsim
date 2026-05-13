{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "magnetic-field-env";

  buildInputs = with pkgs; [
    (python3.withPackages (ps: with ps; [
      opencv4
      scipy
      numpy
      matplotlib
      scipy
      tqdm
    ]))

    libGL
    glib
    xorg.libX11
    stdenv.cc.cc.lib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.libGL pkgs.glib pkgs.stdenv.cc.cc.lib ]}:$LD_LIBRARY_PATH"
    echo "Loaded"
  '';
}
