import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def test_imports():
    try:
        from seedvox.model import SeedVoxModel
        from seedvox.utils.tokenizer import CharTokenizer
        from seedvox.utils.text import normalize_text
        from seedvox.modules.mimi import get_mimi_model
        print("✅ All core modules imported successfully!")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
