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


def load_validation_dataset(max_examples: Optional[int] = None) -> pd.DataFrame:
    """Load FOLIO validation dataset.
    
    Args:
        max_examples: If provided, only load first N examples
    
    Returns:
        DataFrame with columns: story_id, premises, premises-FOL, conclusion, 
                                 conclusion-FOL, label, example_id
    """
    # Find the dataset file (relative to this module or absolute path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible locations
    possible_paths = [
        # Docker/deployed location (relative to project root)
        # os.path.join(current_dir, '..', '..', 'data', 'folio-wiki', 'cleaned-FOLIO-by-yifeng.csv'),
        os.path.join(current_dir, '..', '..', 'data', 'folio-wiki', 'dev.csv'),
        # Relative to folio-benchmark/
        # os.path.join(current_dir, '..', '..', '..', 'folio_correction', 'validation', 'original_dataset.csv'),
        # Absolute path (fallback)
        # '/home/argustest/logic-reasoning-workspace/zhiyu/folio-agent/folio_correction/validation/original_dataset.csv',
    ]
    
    dataset_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            dataset_path = abs_path
            break
    
    if dataset_path is None:
        raise FileNotFoundError(
            f"Could not find FOLIO validation dataset. Tried:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
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

