#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FOLIO dataset loading utilities.
"""

import os
import pandas as pd
from typing import Optional, List


def parse_premises(premises_fol: str) -> List[str]:
    """Parse premises-FOL string (newline-separated) into list of formulas."""
    if not premises_fol or pd.isna(premises_fol):
        return []
    # Split by newlines and filter empty lines
    formulas = [f.strip() for f in premises_fol.split('\n') if f.strip()]
    return formulas


def _resolve_validation_dataset_path() -> str:
    """Resolve which validation split to use.

    Supported environment variables:
    - FOLIO_DATASET_PATH: absolute or relative path to a CSV file
    - FOLIO_DATASET_VARIANT: one of {"refined", "cleaned", "original", "dev"}
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    explicit_path = os.environ.get("FOLIO_DATASET_PATH")
    if explicit_path:
        dataset_path = os.path.abspath(explicit_path)
        if os.path.exists(dataset_path):
            return dataset_path
        raise FileNotFoundError(f"FOLIO_DATASET_PATH does not exist: {dataset_path}")

    variant = os.environ.get("FOLIO_DATASET_VARIANT", "refined").strip().lower()
    variant_to_path = {
        "refined": os.path.join(project_root, "data", "folio-wiki", "cleaned-FOLIO-by-yifeng.csv"),
        "cleaned": os.path.join(project_root, "data", "folio-wiki", "cleaned-FOLIO-by-yifeng.csv"),
        "original": os.path.join(project_root, "data", "folio-wiki", "dev.csv"),
        "dev": os.path.join(project_root, "data", "folio-wiki", "dev.csv"),
    }

    if variant not in variant_to_path:
        raise ValueError(
            f"Unsupported FOLIO_DATASET_VARIANT={variant!r}. "
            "Use one of: refined, cleaned, original, dev."
        )

    dataset_path = os.path.abspath(variant_to_path[variant])
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Resolved dataset path does not exist: {dataset_path}")
    return dataset_path


def load_validation_dataset(max_examples: Optional[int] = None) -> pd.DataFrame:
    """Load FOLIO validation dataset.
    
    Args:
        max_examples: If provided, only load first N examples
    
    Returns:
        DataFrame with columns: story_id, premises, premises-FOL, conclusion, 
                                 conclusion-FOL, label, example_id
    """
    dataset_path = _resolve_validation_dataset_path()
    
    print(f"Loading FOLIO validation dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    if max_examples:
        df = df.head(max_examples)
    
    print(f"Loaded {len(df)} examples from FOLIO validation set")
    return df


def format_folio_problem_as_text(row: pd.Series) -> str:
    """Format a FOLIO problem as natural language text for agents.
    
    Args:
        row: DataFrame row with premises, conclusion, etc.
    
    Returns:
        Formatted text description of the problem
    """
    premises = row['premises']
    conclusion = row['conclusion']
    
    text = f"""Given the following premises:

{premises}

Does the following conclusion logically follow from these premises?

Conclusion: {conclusion}

Please answer with ONLY one of the following: True, False, or Uncertain

Your answer:"""
    
    return text


def format_folio_problem_for_autoform(row: pd.Series) -> str:
    """Format a FOLIO problem for autoformalization (NL → FOL).
    
    Args:
        row: DataFrame row with premises, conclusion, etc.
    
    Returns:
        Formatted text asking for FOL conversion
    """
    premises = row['premises']
    conclusion = row['conclusion']
    
    text = f"""Convert the following natural language premises and conclusion into First-Order Logic (FOL) format.

Premises:
{premises}

Conclusion:
{conclusion}

Please provide your answer in the following format:

PREMISES-FOL:
<premise1 in FOL>
<premise2 in FOL>
...

CONCLUSION-FOL:
<conclusion in FOL>

Your FOL conversion:"""
    
    return text
