"""
Build a wheel containing only the selected OptunaHub samplers.

Target samplers:
  BayesianOptimization: carbo, ctpe, hebo, turbo
  EvolutionAlgorithm:   differential_evolution, hype, moead, nsgaii, speaii
  EvolutionStrategy:    implicit_natural_gradient, mocma
  SwarmIntelligence:    grey_wolf_optimization, pso, whale_optimization
  Top-level:            auto_sampler

Usage:
  uv run build_samplers_whl.py
  -> dist/ に .whl が生成されます

Install:
  pip install dist/optunahub_samplers-*.whl
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).parent
SAMPLERS_SRC = REPO_ROOT / "package" / "samplers"
DIST_DIR = REPO_ROOT / "dist"
VERSION = "1.5.0.dev1"

TARGET_SAMPLERS = [
    # BayesianOptimization
    "carbo",
    "ctpe",
    "hebo",
    "turbo",
    "value_at_risk",
    # EvolutionAlgorithm
    "differential_evolution",
    "hype",
    "moead",
    "nsgaii_with_initial_trials",
    "speaii",
    # EvolutionStrategy
    "implicit_natural_gradient",
    "mocma",
    "restart_cmaes",
    "cma_es_refinement",
    # SwarmIntelligence
    "grey_wolf_optimization",
    "pso",
    "whale_optimization",
    # Top-level
    "auto_sampler",
]

# 各サンプラーの requirements.txt から集約した依存関係
# scipy の pinned version (hype/moead で ==1.13.1) を優先
DEPENDENCIES = [
    "optuna>=4.0.0",
    "optunahub",
    "scipy>=1.13.1",
    "torch",
    "cmaes",
    "hebo",
]

# コピー対象から除外するファイル/ディレクトリ
EXCLUDE_NAMES = {
    "tests",
    "images",
    "example.py",
    "example.ipynb",
    "README.md",
    "LICENSE",
    "requirements.txt",
}


def is_python_subpackage(path: Path) -> bool:
    """__init__.py を持つサブディレクトリかどうか判定する."""
    return path.is_dir() and (path / "__init__.py").exists()


def copy_sampler(sampler_name: str, src_dir: Path, dest_dir: Path) -> None:
    """1つのサンプラーをコピーする（Pythonファイルとサブパッケージのみ）."""
    src = src_dir / sampler_name
    dest = dest_dir / sampler_name
    dest.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.name in EXCLUDE_NAMES:
            continue
        if item.name.startswith("."):
            continue

        if item.is_file() and item.suffix == ".py":
            shutil.copy2(item, dest / item.name)
        elif is_python_subpackage(item):
            shutil.copytree(item, dest / item.name)


def create_pyproject_toml(build_dir: Path) -> None:
    """pyproject.toml を生成する."""
    content = f"""\
[build-system]
requires = ["setuptools >= 40.8.0", "wheel"]
build-backend = "setuptools.build_meta"
"""
    (build_dir / "pyproject.toml").write_text(content)


def create_setup_py(build_dir: Path) -> None:
    """setup.py を生成する."""
    deps = "\n".join(f"        {dep!r}," for dep in DEPENDENCIES)
    content = f"""\
from setuptools import find_packages
from setuptools import setup


setup(
    name="optunahub-samplers",
    version={VERSION!r},
    description="Selected samplers from OptunaHub Registry",
    packages=find_packages(include=["optunahub_samplers", "optunahub_samplers.*"]),
    python_requires=">=3.8",
    install_requires=[
{deps}
    ],
)
"""
    (build_dir / "setup.py").write_text(content)


def create_package_init(pkg_dir: Path) -> None:
    """optunahub_samplers/__init__.py を生成する."""
    lines = [
        '"""OptunaHub samplers package."""',
        "",
    ]
    (pkg_dir / "__init__.py").write_text("\n".join(lines))


def build(tmp_dir: Path) -> Path:
    """wheel をビルドして dist/ に配置する."""
    DIST_DIR.mkdir(exist_ok=True)
    build_command = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(DIST_DIR),
        str(tmp_dir),
    ]
    if shutil.which("uv") is not None:
        build_command = [
            "uv",
            "run",
            "--with",
            "build",
            "--with",
            "setuptools",
            "--with",
            "wheel",
            "python",
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(DIST_DIR),
            str(tmp_dir),
        ]

    result = subprocess.run(
        build_command,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Build failed")
    print(result.stdout)
    return DIST_DIR


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="optunahub_samplers_build_") as tmp:
        tmp_dir = Path(tmp)
        pkg_dir = tmp_dir / "optunahub_samplers"
        pkg_dir.mkdir()

        print("Copying samplers...")
        for name in TARGET_SAMPLERS:
            src = SAMPLERS_SRC / name
            if not src.exists():
                print(f"  WARNING: {name} not found at {src}, skipping.")
                continue
            copy_sampler(name, SAMPLERS_SRC, pkg_dir)
            print(f"  + {name}")

        create_package_init(pkg_dir)
        create_pyproject_toml(tmp_dir)
        create_setup_py(tmp_dir)

        print("\nBuilding wheel...")
        out_dir = build(tmp_dir)

    wheels = list(out_dir.glob("optunahub_samplers-*.whl"))
    if wheels:
        print("\nDone! Wheel created:")
        for w in wheels:
            print(f"  {w}")
    else:
        print("\nBuild finished but no wheel found in dist/")


if __name__ == "__main__":
    main()
