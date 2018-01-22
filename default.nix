# Nix development environment
#
# nix-build -I "nixpkgs=https://goo.gl/wTvE5t"
# nix-shell -I "nixpkgs=https://goo.gl/wTvE5t"
#
with import <nixpkgs> {};
{
  brainbuilder = brainbuilder.overrideDerivation (oldAtr: rec {
    version = "DEV_ENV";
    src = ./.;
  });
}
