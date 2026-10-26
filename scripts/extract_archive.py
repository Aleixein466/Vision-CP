"""Extrae ZIP de forma segura y sin sobrescribir un destino existente."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as zf:
        roots = {Path(n).parts[0] for n in zf.namelist() if Path(n).parts}
        collisions = [r for r in roots if (args.destination / r).exists()]
        if collisions:
            raise SystemExit(f"No se sobrescriben destinos existentes: {collisions}")
        base = args.destination.resolve()
        for member in zf.infolist():
            target = (args.destination / member.filename).resolve()
            if base not in target.parents and target != base:
                raise SystemExit(f"Ruta ZIP insegura: {member.filename}")
        zf.extractall(args.destination)
    print(f"Extraído en {args.destination}")


if __name__ == "__main__":
    main()

