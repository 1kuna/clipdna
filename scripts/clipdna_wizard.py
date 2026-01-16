#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import os
import random
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ComposeCommand:
    base: list[str]

    def run(self, args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
        cmd = self.base + args
        print(f"$ {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=str(cwd), check=check)


def find_compose() -> ComposeCommand:
    docker = shutil.which("docker")
    if docker:
        result = subprocess.run([docker, "compose", "version"], capture_output=True)
        if result.returncode == 0:
            return ComposeCommand([docker, "compose"])
    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return ComposeCommand([docker_compose])
    raise RuntimeError("Docker Compose not found. Install Docker first.")


def prompt_text(label: str, default: str | None = None, secret: bool = False) -> str:
    prompt = f"{label}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "

    if secret:
        value = getpass.getpass(prompt)
    else:
        value = input(prompt)

    if not value and default is not None:
        return default
    return value


def prompt_required(label: str, default: str | None = None, secret: bool = False) -> str:
    while True:
        value = prompt_text(label, default=default, secret=secret)
        if value:
            return value
        print("Value required.")


def prompt_int(label: str, default: int) -> int:
    while True:
        raw = prompt_text(label, default=str(default))
        try:
            return int(raw)
        except ValueError:
            print("Enter a valid integer.")


def prompt_float(label: str, default: float) -> float:
    while True:
        raw = prompt_text(label, default=str(default))
        try:
            return float(raw)
        except ValueError:
            print("Enter a valid number.")


def confirm(label: str, default_yes: bool = False) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"{label} {suffix}: ").strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes"}


def read_env_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text().splitlines()


def parse_env(lines: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def write_env(path: Path, lines: list[str], updates: dict[str, str]) -> None:
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key, _ = stripped.split("=", 1)
        key = key.strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            updated_keys.add(key)
        else:
            new_lines.append(line)
            updated_keys.add(key)

    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    path.write_text("\n".join(new_lines) + "\n")


def copy_env_if_missing(env_path: Path, example_path: Path) -> None:
    if env_path.exists():
        return
    if not example_path.exists():
        raise FileNotFoundError(str(example_path))
    env_path.write_text(example_path.read_text())


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def pick_random_port() -> int:
    for _ in range(50):
        port = random.randint(49152, 65535)
        if is_port_free(port):
            return port
    return 49321


def check_git_updates(repo_root: Path) -> bool:
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_root, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Not a git repository; skipping update check.")
        return False

    subprocess.run(["git", "fetch", "--all", "--prune"], cwd=repo_root, check=False)
    try:
        upstream = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        print("No upstream branch configured; skipping update check.")
        return False

    counts = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    behind = int(counts.split()[1]) if counts else 0
    if behind > 0:
        print(f"Updates available from {upstream} ({behind} commits behind).")
        return True
    print("Repo is up to date.")
    return False


def apply_git_updates(repo_root: Path) -> None:
    print("Pulling latest changes...")
    subprocess.run(["git", "pull"], cwd=repo_root, check=True)


def ensure_dirs(data_path: Path) -> None:
    for subdir in ["videos/sources", "videos/clips", "index/faiss", "embeddings"]:
        (data_path / subdir).mkdir(parents=True, exist_ok=True)


def run_nas_wizard() -> None:
    print("\nClipDNA Desktop NAS Wizard\n")
    env_path = ROOT / "nas" / ".env"
    example_path = ROOT / "nas" / ".env.example"
    copy_env_if_missing(env_path, example_path)

    lines = read_env_lines(env_path)
    values = parse_env(lines)

    data_path = prompt_required("DATA_PATH", default=values.get("DATA_PATH", "/volume1/clipdna-data"))
    postgres_user = prompt_required("POSTGRES_USER", default=values.get("POSTGRES_USER", "clipdna"))
    postgres_password = prompt_required(
        "POSTGRES_PASSWORD", default=values.get("POSTGRES_PASSWORD"), secret=True
    )
    postgres_db = prompt_required("POSTGRES_DB", default=values.get("POSTGRES_DB", "clipdna_desktop"))

    suggested_port = int(values.get("API_PORT", pick_random_port()))
    api_port = prompt_int("API_PORT (any open port)", default=suggested_port)

    updates = {
        "DATA_PATH": data_path,
        "POSTGRES_USER": postgres_user,
        "POSTGRES_PASSWORD": postgres_password,
        "POSTGRES_DB": postgres_db,
        "API_PORT": str(api_port),
    }

    write_env(env_path, lines, updates)

    ensure_dirs(Path(data_path))

    if check_git_updates(ROOT) and confirm("Pull updates now?", default_yes=True):
        apply_git_updates(ROOT)

    compose = find_compose()
    compose.run(["pull"], cwd=ROOT / "nas", check=False)
    compose.run(["up", "-d", "--build"], cwd=ROOT / "nas")

    print(f"\nNAS API should be available at http://localhost:{api_port}/health\n")


def run_dgx_wizard() -> None:
    print("\nClipDNA Desktop DGX Worker Wizard\n")
    env_path = ROOT / "gpu-node" / ".env"
    example_path = ROOT / "gpu-node" / ".env.example"
    copy_env_if_missing(env_path, example_path)

    lines = read_env_lines(env_path)
    values = parse_env(lines)

    nas_host = prompt_required("NAS_HOST", default=values.get("NAS_HOST", "192.168.1.100"))
    nas_mount_path = prompt_required("NAS_MOUNT_PATH", default=values.get("NAS_MOUNT_PATH", "/mnt/nas"))
    postgres_user = prompt_required("POSTGRES_USER", default=values.get("POSTGRES_USER", "clipdna"))
    postgres_password = prompt_required(
        "POSTGRES_PASSWORD", default=values.get("POSTGRES_PASSWORD"), secret=True
    )
    postgres_db = prompt_required("POSTGRES_DB", default=values.get("POSTGRES_DB", "clipdna_desktop"))
    frame_rate = prompt_float("FRAME_RATE", default=float(values.get("FRAME_RATE", 1)))
    batch_size = prompt_int("BATCH_SIZE", default=int(values.get("BATCH_SIZE", 32)))
    num_workers = prompt_int("NUM_WORKERS", default=int(values.get("NUM_WORKERS", 1)))
    log_level = prompt_required("LOG_LEVEL", default=values.get("LOG_LEVEL", "INFO"))

    updates = {
        "NAS_HOST": nas_host,
        "NAS_MOUNT_PATH": nas_mount_path,
        "POSTGRES_USER": postgres_user,
        "POSTGRES_PASSWORD": postgres_password,
        "POSTGRES_DB": postgres_db,
        "FRAME_RATE": str(frame_rate),
        "BATCH_SIZE": str(batch_size),
        "NUM_WORKERS": str(num_workers),
        "LOG_LEVEL": log_level,
    }

    write_env(env_path, lines, updates)

    if check_git_updates(ROOT) and confirm("Pull updates now?", default_yes=True):
        apply_git_updates(ROOT)

    compose = find_compose()
    compose.run(["pull"], cwd=ROOT / "gpu-node", check=False)
    compose.run(["up", "-d", "--build"], cwd=ROOT / "gpu-node")

    print("\nDGX worker is running.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ClipDNA Desktop setup wizard")
    subparsers = parser.add_subparsers(dest="target", required=True)

    subparsers.add_parser("nas", help="Configure and start NAS services")
    subparsers.add_parser("dgx", help="Configure and start DGX worker")

    args = parser.parse_args()

    if args.target == "nas":
        run_nas_wizard()
    elif args.target == "dgx":
        run_dgx_wizard()
    else:
        parser.error("Unknown target")


if __name__ == "__main__":
    main()
