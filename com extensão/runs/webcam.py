import os
import sys
from pathlib import Path

# ── Corrige conflito Qt (cv2 tem Qt embutido que briga com PyQt5) ────────────
# Precisa ser feito ANTES de qualquer import do cv2 ou PyQt5
ROOT = Path(__file__).resolve().parent.parent
PYQT5_PLUGINS = (
    ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
    / "PyQt5" / "Qt5" / "plugins"
)
if PYQT5_PLUGINS.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(PYQT5_PLUGINS)

# Garante Wayland como fallback se XCB falhar
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Lança a GUI ───────────────────────────────────────────────────────────────
from mindflow_extractor.runners.gui_runner import run_gui  # noqa: E402

if __name__ == "__main__":
    run_gui()
