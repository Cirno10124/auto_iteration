import os
from pathlib import Path

site_pkgs = Path("venv/Lib/site-packages")  # Windows 路径
# site_pkgs = Path("auto_iteration/venv/lib/python3.11/site-packages")  # Linux 路径
sizes = []
for pkg in site_pkgs.iterdir():
    if pkg.is_dir():
        total = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file())
        sizes.append((pkg.name, total))
sizes.sort(key=lambda x: x[1], reverse=True)
for name, size in sizes[:40]:
    print(f"{name.ljust(30)} {size/1024/1024:.1f} MB")