"""
MindFlow AI — Launcher
======================
Basta apertar RUN neste arquivo.

O script detecta automaticamente o virtualenv (.venv) do projeto,
re-executa a si mesmo com o Python correto se necessário, e inicia
o runner de webcam.
"""
import sys
import os
import subprocess
from pathlib import Path

# ── Diretório raiz do projeto (onde este arquivo está) ──────────────────────
ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

# ── Se não estamos rodando dentro do venv, relança com o Python correto ─────
def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix or str(ROOT / ".venv") in sys.prefix

if not _in_venv() and VENV_PYTHON.exists():
    print(f"[launcher] Usando venv: {VENV_PYTHON}")
    result = subprocess.run([str(VENV_PYTHON), __file__] + sys.argv[1:])
    sys.exit(result.returncode)

# ── Se o venv não existe ainda, cria e instala as dependências ───────────────
if not VENV_PYTHON.exists():
    print("[launcher] Criando ambiente virtual (.venv)...")
    subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
    print("[launcher] Instalando dependências (pode demorar na 1ª vez)...")
    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )
    # Garante a versão certa do mediapipe
    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "mediapipe==0.10.14"],
        check=True,
    )
    print("[launcher] Pronto! Relançando...\n")
    result = subprocess.run([str(VENV_PYTHON), __file__] + sys.argv[1:])
    sys.exit(result.returncode)

# ── Garante que o pacote local está no path ──────────────────────────────────
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Inicia o runner ──────────────────────────────────────────────────────────
from mindflow_extractor.runners.webcam_runner import run_webcam  # noqa: E402

run_webcam()
