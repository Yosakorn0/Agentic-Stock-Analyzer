"""
Simple helper to test Hugging Face models used by the Multi-AI scanner.

Run this once to verify that a model:
- can be loaded from your local Hugging Face cache
- can generate a short response

Usage examples (from project root or Agentic-Stock-Analyzer folder):

    python examples/test_hf_models.py --model hf:finance-chat
    python examples/test_hf_models.py --model hf:tinyllama
    python examples/test_hf_models.py --all-shortcuts
"""

import argparse
import os
import sys
from typing import List


def _ensure_path():
    """
    Make sure imports work whether we run from project root or Agentic-Stock-Analyzer.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_path()

try:
    from core.analysis.huggingface_analyzer import (
        HuggingFaceAnalyzer,
        RECOMMENDED_MODELS,
    )
except ImportError as e:
    print("❌ Could not import HuggingFaceAnalyzer. Are you in the right directory?")
    print("   Run from project root or Agentic-Stock-Analyzer folder.")
    print(f"   Details: {e}")
    sys.exit(1)


def resolve_model_name(model: str) -> str:
    """
    Resolve a shortcut like 'hf:finance-chat' or 'hf:tinyllama'
    into a full Hugging Face model id using RECOMMENDED_MODELS.
    """
    name = model
    if name.startswith("hf:"):
        name = name.replace("hf:", "", 1)
    if RECOMMENDED_MODELS and name in RECOMMENDED_MODELS:
        return RECOMMENDED_MODELS[name]
    return name


def test_single_model(model_id: str, ticker: str = "AAPL", use_gpu: bool = False) -> bool:
    """
    Load and quickly test a single Hugging Face model.

    Returns True if load + simple generation work, False otherwise.
    """
    print(f"\n================================================================================")
    print(f"🔍 Testing Hugging Face model: {model_id}")
    print(f"================================================================================")

    try:
        analyzer = HuggingFaceAnalyzer(model_name=model_id, use_gpu=use_gpu, quiet=False)
        if not analyzer.is_loaded:
            print(f"❌ Model {model_id} failed to load (is_loaded=False)")
            return False
    except Exception as e:
        print(f"❌ Exception while loading {model_id}: {str(e)[:200]}")
        return False

    # Prepare a very small dummy context
    stock_info = {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "52_week_high": 220.0,
        "52_week_low": 150.0,
        "pe_ratio": 28.5,
        "market_cap": 3_000_000_000_000,
    }
    technical_signals = {
        "current_price": 190.0,
        "price_change_1d": 0.5,
        "price_change_5d": 2.3,
        "price_change_20d": 5.0,
        "rsi": 45.0,
        "rsi_signal": "neutral",
        "direction": "up",
        "strength": 60.0,
        "ema_cross": "bullish",
        "macd_signal": "bullish",
        "bb_position": "middle",
    }
    price_summary = {
        "current_price": technical_signals["current_price"],
        "price_change_1d": technical_signals["price_change_1d"],
        "price_change_5d": technical_signals["price_change_5d"],
        "price_change_20d": technical_signals["price_change_20d"],
    }

    try:
        print("🧪 Running a tiny test analysis on ticker stocks...")
        result = analyzer.analyze_stock(
            ticker=ticker,
            stock_info=stock_info,
            technical_signals=technical_signals,
            price_data_summary=price_summary,
        )
        rec = result.get("recommendation", "UNKNOWN")
        conf = result.get("confidence", 0)
        print(f"✅ Test completed. Recommendation: {rec}, Confidence: {conf}")
        
        # Cleanup to free memory
        print("🧹 Cleaning up model...")
        analyzer.cleanup()
        print("✅ Cleanup complete")
        
        return True
    except Exception as e:
        print(f"❌ Exception during test analysis for {model_id}: {str(e)[:200]}")
        # Try to cleanup even on error
        try:
            if hasattr(analyzer, 'cleanup'):
                analyzer.cleanup()
        except:
            pass
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Hugging Face models used by the Multi-AI scanner."
    )
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "Model to test. Examples: "
            "hf:finance-chat, hf:tinyllama, mistralai/Mistral-7B-Instruct-v0.2"
        ),
    )
    parser.add_argument(
        "--all-shortcuts",
        action="store_true",
        help="Test all RECOMMENDED_MODELS shortcuts (finance-chat, tinyllama, etc.).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for testing if CUDA is available.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Single ticker symbol to use for the test analysis (default: AAPL).",
    )

    args = parser.parse_args()

    models_to_test: List[str] = []

    if args.all_shortcuts:
        print("🔧 Testing all RECOMMENDED_MODELS shortcuts:")
        for shortcut, full_id in RECOMMENDED_MODELS.items():
            print(f"   - {shortcut} -> {full_id}")
            models_to_test.append(full_id)

    if args.model:
        resolved = resolve_model_name(args.model)
        if resolved not in models_to_test:
            models_to_test.append(resolved)

    if not models_to_test:
        print("⚠️ No models specified.")
        print("   Use --model hf:finance-chat or --all-shortcuts")
        sys.exit(1)

    successes = 0
    for m in models_to_test:
        if test_single_model(m, ticker=args.ticker, use_gpu=args.use_gpu):
            successes += 1

    print("\n================================================================================")
    print(f"✅ Finished testing {len(models_to_test)} model(s)")
    print(f"   Successful: {successes}")
    print(f"   Failed: {len(models_to_test) - successes}")
    print("================================================================================")


if __name__ == "__main__":
    main()


