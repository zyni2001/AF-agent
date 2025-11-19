#!/usr/bin/env python3
"""Simple test to verify the folio-benchmark system works."""

import sys
import os

# Set environment
os.environ['VAMPIRE_PATH'] = '/home/argustest/logic-reasoning-workspace/zhiyu/folio-agent/folio_correction/vampire/build/vampire'
# Note: GEMINI_API_KEY should be set in your environment or .env file
if 'GEMINI_API_KEY' not in os.environ:
    print("Warning: GEMINI_API_KEY not set in environment. Please set it before running tests.")
    print("Example: export GEMINI_API_KEY='your-api-key-here'")
    sys.exit(1)

# Add to path
sys.path.insert(0, '.')

print("\n" + "="*70)
print("FOLIO BENCHMARK - SIMPLE SYSTEM TEST")
print("="*70 + "\n")

# Test 1: Load dataset
print("Test 1: Loading FOLIO validation dataset...")
try:
    from src.folio_utils.dataset import load_validation_dataset
    df = load_validation_dataset(max_examples=3)
    print(f"✓ Successfully loaded {len(df)} examples")
    print(f"  Columns: {list(df.columns)}\n")
except Exception as e:
    print(f"✗ FAILED: {e}\n")
    sys.exit(1)

# Test 2: Vampire execution
print("Test 2: Testing Vampire theorem prover...")
try:
    from src.folio_utils.vampire_runner import run_vampire
    result = run_vampire('fof(test, axiom, p(a)).', time_limit=5)
    print(f"✓ Vampire executed successfully")
    print(f"  Status: {result['status']}\n")
except Exception as e:
    print(f"✗ FAILED: {e}\n")
    sys.exit(1)

# Test 3: Format FOLIO problem
print("Test 3: Formatting FOLIO problem...")
try:
    from src.folio_utils.dataset import format_folio_problem_as_text
    row = df.iloc[0]
    problem_text = format_folio_problem_as_text(row)
    print(f"✓ Successfully formatted problem")
    print(f"  Problem length: {len(problem_text)} characters")
    print(f"  Expected label: {row['label']}\n")
except Exception as e:
    print(f"✗ FAILED: {e}\n")
    sys.exit(1)

# Test 4: Test FOL to TPTP conversion
print("Test 4: Testing FOL to TPTP conversion...")
try:
    from src.folio_utils.fol_to_tptp import make_tptp_with_negated_axiom
    premises = ["p(a)", "q(b)"]
    conclusion = "r(c)"
    tptp = make_tptp_with_negated_axiom(premises, conclusion)
    print(f"✓ Successfully converted to TPTP")
    print(f"  TPTP length: {len(tptp)} characters\n")
except Exception as e:
    print(f"✗ FAILED: {e}\n")
    sys.exit(1)

# Test 5: Full vampire check on FOLIO case
print("Test 5: Full Vampire check on real FOLIO case...")
try:
    from src.folio_utils.vampire_runner import check_folio_with_vampire
    from src.folio_utils.dataset import parse_premises
    
    row = df.iloc[0]
    premises_fol = parse_premises(row['premises-FOL'])
    conclusion_fol = row['conclusion-FOL']
    expected_label = str(row['label']).strip()
    
    result = check_folio_with_vampire(
        premises_fol=premises_fol,
        conclusion_fol=conclusion_fol,
        time_limit=50
    )
    
    predicted = result['predicted_label']
    predicted_str = "True" if predicted is True else "False" if predicted is False else "Uncertain"
    
    print(f"✓ Vampire check completed")
    print(f"  Expected: {expected_label}")
    print(f"  Predicted: {predicted_str}")
    print(f"  Status: {result['vampire_status']}")
    print(f"  Match: {'✓' if predicted_str.lower() == expected_label.lower() else '✗'}\n")
except Exception as e:
    print(f"✗ FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nThe folio-benchmark system is ready to use.")
print("\nNext steps:")
print("  1. Run a quick test: conda activate folio-bench && python main.py quick")
print("  2. Run full evaluation: conda activate folio-bench && python main.py launch")
print("="*70 + "\n")

