from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / 'assets'
FRUIT_ASSET_DIR = ASSETS_DIR / 'fruits'

DEFAULT_WINDOW_SIZE = (400, 800)
MIN_WINDOW_SIZE = (360, 560)
FPS = 120
