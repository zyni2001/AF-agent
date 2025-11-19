#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vampire theorem prover runner for FOLIO problems.
"""

import os
import subprocess
import tempfile
from typing import Optional, List


def run_vampire(tptp_content: str, time_limit: int = 50, vampire_path: str = None) -> dict:
    """Run Vampire on TPTP content and return result.
    
    Args:
        tptp_content: TPTP problem content
        time_limit: Time limit in seconds
        vampire_path: Path to Vampire executable
    
    Returns:
        dict with keys:
            - refutation_found: bool
            - theorem_found: bool
            - status: str (theorem, unsatisfiable, satisfiable, timeout, etc.)
            - is_satisfiable: bool
            - is_unsatisfiable: bool
            - output: str (raw Vampire output)
            - return_code: int
            - has_parse_error: bool
            - parse_error_msg: Optional[str]
    """
    if vampire_path is None:
        vampire_path = os.environ.get('VAMPIRE_PATH', '/usr/local/bin/vampire')
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.p', delete=False) as f:
        f.write(tptp_content)
        temp_file = f.name
    
    try:
        # Run Vampire
        cmd = [
            vampire_path,
            "--input_syntax", "tptp",
            "--time_limit", str(time_limit),
            temp_file
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 2  # Add buffer for overhead
        )
        
        output = result.stdout + result.stderr
        
        # Check for parse errors
        has_parse_error = False
        parse_error_msg = None
        if "parse error" in output.lower() or "User error" in output:
            has_parse_error = True
            for line in output.split('\n'):
                if "parse error" in line.lower() or "User error" in line:
                    parse_error_msg = line.strip()
                    break
        
        # Determine result status
        refutation_found = False
        theorem_found = False
        status = "unknown"
        is_satisfiable = False
        is_unsatisfiable = False
        
        if "Refutation found" in output or "SZS status Theorem" in output:
            refutation_found = True
            theorem_found = True
            is_unsatisfiable = True
            status = "theorem"
        elif "SZS status Unsatisfiable" in output or "SZS status ContradictoryAxioms" in output:
            refutation_found = True
            is_unsatisfiable = True
            status = "unsatisfiable"
        elif "Termination reason: Refutation" in output:
            refutation_found = True
            theorem_found = True
            is_unsatisfiable = True
            status = "theorem"
        elif "SZS status Satisfiable" in output or "satisfiable" in output.lower():
            is_satisfiable = True
            status = "satisfiable"
        elif "Termination reason: RefutationNotFound" in output:
            status = "refutation_not_found"
        elif result.returncode == 4 and not has_parse_error:
            status = "no_refutation"
        else:
            if "time" in output.lower() or result.returncode == 143:
                status = "timeout"
            else:
                status = "unknown"
        
        return {
            "refutation_found": refutation_found,
            "theorem_found": theorem_found,
            "status": status,
            "is_satisfiable": is_satisfiable,
            "is_unsatisfiable": is_unsatisfiable,
            "output": output,
            "return_code": result.returncode,
            "has_parse_error": has_parse_error,
            "parse_error_msg": parse_error_msg
        }
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def check_folio_with_vampire(premises_fol: List[str], conclusion_fol: str, 
                              time_limit: int = 50, vampire_path: str = None) -> dict:
    """Check if conclusion follows from premises using Vampire.
    
    Args:
        premises_fol: List of FOL premise formulas
        conclusion_fol: FOL conclusion formula
        time_limit: Time limit in seconds
        vampire_path: Path to Vampire executable
    
    Returns:
        dict with keys:
            - predicted_label: True/False/None (None = Uncertain)
            - vampire_status: str
            - premises_neg_conclusion_status: str
            - premises_conclusion_status: Optional[str]
    """
    from .fol_to_tptp import make_tptp_with_negated_axiom, make_tptp_text, _clean_formula
    
    # Step 1: Check premises + ~conclusion (refutation-based reasoning)
    tptp_content = make_tptp_with_negated_axiom(premises_fol, conclusion_fol)
    vampire_result = run_vampire(tptp_content, time_limit, vampire_path)
    
    predicted_label = None
    conclusion_status = None
    
    if vampire_result["is_unsatisfiable"] or vampire_result["theorem_found"]:
        # UNSAT: premises + ~conclusion is contradictory → conclusion follows
        predicted_label = True
    elif vampire_result["is_satisfiable"]:
        # SAT: premises + ~conclusion is consistent
        # Need to check if conclusion is definitively False or just Uncertain
        
        # Create TPTP with premises + conclusion (no negation)
        lines = []
        for i, pr in enumerate(premises_fol, 1):
            prf = _clean_formula(pr)
            lines.append(f"fof(p{i}, axiom, {prf}).")
        cf = _clean_formula(f"({conclusion_fol})")
        if not cf.startswith('(') or not cf.endswith(')'):
            cf = f"({cf})"
        lines.append(f"fof(conclusion, axiom, {cf}).")
        tptp_with_conclusion = "\n".join(lines) + "\n"
        
        conclusion_result = run_vampire(tptp_with_conclusion, time_limit, vampire_path)
        conclusion_status = conclusion_result["status"]
        
        if conclusion_result["is_unsatisfiable"] or conclusion_result["theorem_found"]:
            # premises + conclusion is UNSAT → conclusion contradicts premises
            predicted_label = False
        else:
            # premises + conclusion is SAT → cannot determine
            predicted_label = None
    elif vampire_result["status"] in ["refutation_not_found", "no_refutation"]:
        # No refutation found → conclusion does not follow
        predicted_label = False
    else:
        # Timeout or error → uncertain
        predicted_label = None
    
    return {
        "predicted_label": predicted_label,
        "vampire_status": vampire_result["status"],
        "premises_neg_conclusion_status": vampire_result["status"],
        "premises_conclusion_status": conclusion_status,
        "has_parse_error": vampire_result["has_parse_error"],
        "parse_error_msg": vampire_result.get("parse_error_msg"),
    }

