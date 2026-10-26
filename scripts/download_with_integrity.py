"""Descarga reanudable, limitada por tamaño y con manifiesto SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sha256")
    p.add_argument("--max-mb", type=float, default=100.0)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"No se sobrescribe: {args.output}")

    usage = shutil.disk_usage(args.output.parent)
    max_bytes = int(args.max_mb * 1024 * 1024)
    if usage.free < max_bytes + 100 * 1024 * 1024:
        raise SystemExit("Espacio libre insuficiente para el presupuesto y margen de seguridad")

    part = args.output.with_suffix(args.output.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(args.url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(req) as response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            raise SystemExit("El servidor no aceptó reanudación; quite manualmente el .part para reiniciar")
        declared = response.headers.get("Content-Length")
        total = offset + int(declared) if declared else None
        if total and total > max_bytes:
            raise SystemExit(f"Descarga declarada {total / 2**20:.1f} MB > límite {args.max_mb:.1f} MB")
        mode = "ab" if offset else "wb"
        with part.open(mode) as fh:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                fh.write(block)
                if fh.tell() > max_bytes:
                    raise SystemExit("Descarga detenida al superar el límite")

    digest = sha256(part)
    if args.sha256 and digest.lower() != args.sha256.lower():
        raise SystemExit(f"SHA-256 no coincide: {digest}")
    part.replace(args.output)
    manifest = {
        "url": args.url,
        "path": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": digest,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_sha256": args.sha256,
    }
    manifest_path = args.output.parent.parent / "manifests" / f"{args.output.name}.download.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

