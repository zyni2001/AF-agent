#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, unicodedata
from typing import List, Dict

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to ASCII."""
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')

def fix_var_binding(text: str) -> str:
    """Fix variable binding by replacing lowercase vars with uppercase in quantifier scopes."""
    quantifier_pattern = r'([!?])\s*\[\s*([A-Z][a-zA-Z0-9]*(?:\s*,\s*[A-Z][a-zA-Z0-9]*)*)\s*\]\s*:'
    matches = list(re.finditer(quantifier_pattern, text))
    
    for match in reversed(matches):
        var_list = [v.strip().upper() for v in match.group(2).split(',')]
        start_pos = match.end()
        
        # Find scope
        paren_count = 0
        in_scope = False
        scope_start = start_pos
        scope_end = len(text)
        
        for i in range(start_pos, len(text)):
            if text[i] == '(':
                if not in_scope:
                    in_scope = True
                    scope_start = i
                paren_count += 1
            elif text[i] == ')':
                paren_count -= 1
                if paren_count == 0 and in_scope:
                    scope_end = i + 1
                    break
            elif text[i] in ['&', '|', '=>', '<=>'] and paren_count == 0 and i > start_pos:
                scope_end = i
                break
        
        if not in_scope:
            for i in range(start_pos, len(text)):
                if text[i] in ['&', '|'] and i + 1 < len(text) and text[i+1] in [' ', '&', '|']:
                    scope_end = i
                    break
                elif text[i:i+2] in ['=>', '<=>']:
                    scope_end = i
                    break
        
        scope = text[scope_start:scope_end]
        
        for var_upper in var_list:
            var_lower = var_upper.lower()
            nested_single = rf'[!?]\s*\[\s*{var_upper}\s*\]\s*:'
            nested_multiple = rf'[!?]\s*\[\s*[A-Z][a-zA-Z0-9]*(?:\s*,\s*[A-Z][a-zA-Z0-9]*)*,?\s*{var_upper}\s*(?:,\s*[A-Z][a-zA-Z0-9]*)*\]\s*:'
            nested_matches = list(re.finditer(nested_single, scope))
            nested_matches.extend(re.finditer(nested_multiple, scope))
            
            if not nested_matches:
                scope = re.sub(rf'\b{var_lower}\b', var_upper, scope)
            else:
                nested_scopes = []
                for nested_match in nested_matches:
                    nested_start = nested_match.end()
                    nested_paren_count = 0
                    nested_scope_start = nested_start
                    nested_scope_end = len(scope)
                    
                    while nested_scope_start < len(scope) and scope[nested_scope_start] == ' ':
                        nested_scope_start += 1
                    
                    for i in range(nested_scope_start, len(scope)):
                        if scope[i] == '(':
                            nested_paren_count += 1
                        elif scope[i] == ')':
                            nested_paren_count -= 1
                            if nested_paren_count == 0:
                                nested_scope_end = i + 1
                                break
                        elif scope[i] in ['&', '|'] and nested_paren_count == 0:
                            nested_scope_end = i
                            break
                        elif scope[i:i+2] in ['=>', '<=>'] and nested_paren_count == 0:
                            nested_scope_end = i
                            break
                    
                    nested_scopes.append((nested_scope_start, nested_scope_end))
                
                fixed_scope_parts = []
                last_pos = 0
                
                for nested_start, nested_end in sorted(nested_scopes):
                    if last_pos < nested_start:
                        chunk = scope[last_pos:nested_start]
                        fixed_chunk = re.sub(rf'\b{var_lower}\b', var_upper, chunk)
                        fixed_scope_parts.append(fixed_chunk)
                    fixed_scope_parts.append(scope[nested_start:nested_end])
                    last_pos = nested_end
                
                if last_pos < len(scope):
                    chunk = scope[last_pos:]
                    fixed_chunk = re.sub(rf'\b{var_lower}\b', var_upper, chunk)
                    fixed_scope_parts.append(fixed_chunk)
                
                scope = ''.join(fixed_scope_parts)
        
        text = text[:scope_start] + scope + text[scope_end:]
    
    return text

def combine_chained_quantifiers(s: str) -> str:
    """Combine chained quantifiers: ? [X] ? [Y] -> ? [X, Y] and ! [X] ! [Y] -> ! [X, Y]"""
    # Existential quantifiers
    max_iter = 20
    iter_count = 0
    while re.search(r'\?\s*\[([A-Z][a-zA-Z0-9]*)\]\s*\?\s*\[', s) and iter_count < max_iter:
        iter_count += 1
        match = re.search(r'(\?\s*\[([A-Z][a-zA-Z0-9]*)\]\s*)+?(?=([!\s*\[|[:\(]))', s)
        if match:
            quantifiers = re.findall(r'\?\s*\[([A-Z][a-zA-Z0-9]*)\]', match.group(0))
            if len(quantifiers) <= 1:
                break
            
            seen = set()
            unique_quantifiers = [q for q in quantifiers if not (q in seen or seen.add(q))]
            combined = '? [' + ', '.join(unique_quantifiers) + '] '
            
            after_match = s[match.end():]
            if re.match(r'\s*!\s*\[', after_match):
                combined = combined.strip() + ' : '
            elif re.match(r'\s*[:\(]', after_match):
                combined = combined.strip() + ' :'
            
            if match.group(0).strip() == combined.strip():
                break
            new_s = s[:match.start()] + combined + s[match.end():]
            if new_s == s:
                break
            s = new_s
        else:
            break
    
    # Universal quantifiers
    iter_count = 0
    while re.search(r'!\s*\[([A-Z,\s]+)\]\s*!\s*\[', s) and iter_count < max_iter:
        iter_count += 1
        match = re.search(r'(!\s*\[([A-Z,\s]+)\]\s*)+?(?=(\?\s*\[|[:\(]))', s)
        if match:
            var_lists = re.findall(r'!\s*\[([A-Z,\s]+)\]', match.group(0))
            if len(var_lists) <= 1:
                break
            
            all_vars = []
            for var_list in var_lists:
                all_vars.extend([v.strip().upper() for v in var_list.split(',')])
            
            seen = set()
            unique_vars = [v for v in all_vars if not (v in seen or seen.add(v))]
            
            after_match = s[match.end():]
            if re.match(r'\s*\?\s*\[', after_match):
                combined = '! [' + ', '.join(unique_vars) + '] : '
            elif re.match(r'\s*[:\(]', after_match):
                combined = '! [' + ', '.join(unique_vars) + '] :'
            else:
                combined = '! [' + ', '.join(unique_vars) + '] '
            
            if match.group(0).strip() == combined.strip():
                break
            new_s = s[:match.start()] + combined + s[match.end():]
            if new_s == s:
                break
            s = new_s
        else:
            break
    
    return s

def _convert_to_tptp(s: str) -> str:
    """Convert FOL formula to TPTP/FOF syntax."""
    s = normalize_unicode(s.strip())
    
    # Fix malformed patterns: P(x, A ∧ B) -> P(x, A) ∧ B
    s = re.sub(r'([A-Z][a-zA-Z]*)\s*\(\s*([a-z]+)\s*,\s*([a-zA-Z]+)\s*([∧∨])\s*([A-Z][a-zA-Z]*)\s*\(',
               lambda m: f'{m.group(1).lower()}({m.group(2)}) {m.group(4)} {m.group(5).lower()}(', s)
    
    # Lowercase predicates/functions
    def lower_and_quote_predicate(match):
        ident = match.group(1)
        ident_lower = ident.lower()
        if any(c in ident_lower for c in ['-', '.', '/', '+']):
            return f"'{ident_lower}'"
        return ident_lower
    
    s = re.sub(r'\b([A-Z][a-zA-Z0-9_.-]*)\s*\(', 
               lambda m: lower_and_quote_predicate(m) + '(', s)
    
    # Replace apostrophes in identifiers
    s = re.sub(r"([a-zA-Z0-9.]+)'([a-zA-Z0-9.]+)", r"\1_\2", s)
    s = re.sub(r"([a-z]+\.)'([a-z.]+)'", r"\1_\2", s)
    
    # Fix missing commas: func(xm bolt) -> func(x, bolt)
    s = re.sub(r'([a-zA-Z]+)\(([a-z])([a-z])\s+([a-z]+)', r'\1(\2, \4', s)
    
    # Quote complex identifiers (with hyphens/periods/digits) in arguments
    def quote_complex_args(match):
        pred = match.group(1)
        args = match.group(2)
        if args.startswith("'") and args.endswith("'"):
            return f'{pred}({args})'
        
        def quote_complex_ident(m):
            ident = m.group(0)
            if ident.startswith("'") and ident.endswith("'"):
                return ident
            if ('-' in ident or '.' in ident) and not ident.replace('-', '').replace('.', '').isdigit():
                ident_lower = ident.lower()
                if not (ident_lower.startswith("'") and ident_lower.endswith("'")):
                    return f"'{ident_lower}'"
            return ident
        
        args_quoted = re.sub(r'\b([a-zA-Z0-9][a-zA-Z0-9_.-]+)\b', quote_complex_ident, args)
        return f'{pred}({args_quoted})'
    
    s = re.sub(r'([a-z]+)\(([^)]*)\)', quote_complex_args, s)
    
    # Quote numeric literals in predicate arguments
    def quote_numeric_constant(match):
        pred = match.group(1)
        args = match.group(2)
        if args.startswith("'") and args.endswith("'"):
            return f'{pred}({args})'
        
        def combine_number_pattern(m):
            num1, num2 = m.group(1), m.group(2)
            if num1.isdigit() and num2.isdigit() and len(num2) == 3:
                return num1 + num2
            return m.group(0)
        
        args = re.sub(r'(\d+),\s*(\d{3})\b', combine_number_pattern, args)
        
        def quote_if_not_quoted(m):
            num = m.group(1)
            pos = m.start()
            before = args[:pos]
            if before.count("'") % 2 == 1:
                return num
            num_start, num_end = m.start(), m.end()
            if (num_start > 0 and args[num_start-1] in ['-', '.']) or \
               (num_end < len(args) and args[num_end] in ['-', '.']):
                return num
            return f"'{num}'"
        
        args_quoted = re.sub(r'(?<![a-zA-Z0-9_])\b(\d+)\b(?![a-zA-Z0-9_])', quote_if_not_quoted, args)
        return f'{pred}({args_quoted})'
    
    s = re.sub(r'([a-z]+)\(([^)]*)\)', quote_numeric_constant, s)
    
    # Quote identifiers starting with digits
    def quote_digit_start_ident(match):
        ident = match.group(1)
        if ident.startswith("'") and ident.endswith("'"):
            return ident
        if ident.isdigit():
            return ident
        return f"'{ident.lower()}'"
    
    s = re.sub(r'(?<![-.\w])\b([0-9][a-zA-Z0-9_.-]+)\b(?![-.\w])', quote_digit_start_ident, s)
    
    # Lowercase constants
    def lower_constant(match):
        ident = match.group(1)
        if len(ident) == 1 and ident.isupper():
            return ident
        if ident.startswith("'") and ident.endswith("'"):
            return ident
        ident_lower = ident.lower()
        if any(c in ident_lower for c in ['-', '.', '/', '+']):
            return f"'{ident_lower}'"
        return ident_lower
    
    s = re.sub(r'\b([A-Z][a-z][a-zA-Z0-9_.-]+)\b', lower_constant, s)
    s = re.sub(r'\b([A-Z]{2,}[a-zA-Z0-9_.-]*)\b', lower_constant, s)
    s = re.sub(r'\b([a-z][a-zA-Z0-9_.-]*[A-Z][a-zA-Z0-9_.-]*)\b', 
               lambda m: f"'{m.group(1).lower()}'" if any(c in m.group(1) for c in ['-', '.', '/', '+']) else m.group(1).lower(), s)
    
    # Normalize constants and predicates
    s = re.sub(r'\byr(\d+)\b', lambda m: 'year' + m.group(1), s, flags=re.IGNORECASE)
    s = re.sub(r'\btoppledover\b', 'toppleover', s, flags=re.IGNORECASE)
    
    # Quote identifiers with periods
    def quote_if_needed(ident):
        if any(c in ident for c in ['-', '.', '/', '+']):
            return f"'{ident}'"
        return ident
    
    def quote_unquoted_ids_in_parens(match):
        content = match.group(1)
        def quote_if_unquoted(n):
            ident = n.group(1)
            if ident.startswith("'") and ident.endswith("'"):
                return ident
            pos = n.start()
            if content[:pos].count("'") % 2 == 1:
                return ident
            return f"'{ident}'"
        result = re.sub(r'\b([a-z][a-z0-9_]*\.[a-zA-Z0-9_.-]+)\b', quote_if_unquoted, content)
        return '(' + result + ')'
    
    s = re.sub(r'\(([^()]+)\)', quote_unquoted_ids_in_parens, s)
    s = re.sub(r"(?<!')(?<![a-zA-Z0-9_])([a-z][a-z0-9_]*\.[a-zA-Z0-9_.-]+)(?=[,)])", 
               lambda m: f"'{m.group(1)}'" if not (m.group(1).startswith("'") and m.group(1).endswith("'")) else m.group(1), s)
    
    # Quantifier conversion: ∀x -> ! [X], ∃x -> ? [X]
    s = re.sub(r'∃\s*\(', '∃x (', s)
    s = re.sub(r'∀([a-zA-Z])', lambda m: f'! [{m.group(1).upper()}]', s)
    s = re.sub(r'∃([a-zA-Z])', lambda m: f'? [{m.group(1).upper()}]', s)
    s = re.sub(r'∀\s*([a-zA-Z])', lambda m: f'! [{m.group(1).upper()}]', s)
    s = re.sub(r'∃\s*([a-zA-Z])', lambda m: f'? [{m.group(1).upper()}]', s)
    
    # Fix quantifier syntax
    s = re.sub(r'!\[([a-zA-Z][a-zA-Z0-9]*)\]', lambda m: f'! [{m.group(1).upper()}]', s)
    s = re.sub(r'\?\[([a-zA-Z][a-zA-Z0-9]*)\]', lambda m: f'? [{m.group(1).upper()}]', s)
    s = re.sub(r'!\s*\[([a-zA-Z][a-zA-Z0-9]*)\]', lambda m: f'! [{m.group(1).upper()}]', s)
    s = re.sub(r'\?\s*\[([a-zA-Z][a-zA-Z0-9]*)\]', lambda m: f'? [{m.group(1).upper()}]', s)
    
    # Add colons to quantifiers
    s = re.sub(r'!\s*\[\s*([a-zA-Z][a-zA-Z0-9]*)\s*\]\s*:', lambda m: f'! [{m.group(1).upper()}] :', s)
    s = re.sub(r'\?\s*\[\s*([a-zA-Z][a-zA-Z0-9]*)\s*\]\s*:', lambda m: f'? [{m.group(1).upper()}] :', s)
    s = re.sub(r'(!)\s*\[\s*([a-zA-Z][a-zA-Z0-9]*)\s*\]\s*\(', lambda m: f'{m.group(1)} [{m.group(2).upper()}] : (', s)
    s = re.sub(r'(\?)\s*\[\s*([a-zA-Z][a-zA-Z0-9]*)\s*\]\s*\(', lambda m: f'{m.group(1)} [{m.group(2).upper()}] : (', s)
    
    # Fix mixed quantifiers
    s = re.sub(r'!\s*\[\s*([A-Z,\s]+)\]\s*(\?\s*\[)', lambda m: f'! [{m.group(1).strip()}] : {m.group(2)}', s)
    s = re.sub(r'\?\s*\[\s*([A-Z,\s]+)\]\s*(!\s*\[)', lambda m: f'? [{m.group(1).strip()}] : {m.group(2)}', s)
    s = re.sub(r'(\?\s*\[\s*([A-Z][a-zA-Z0-9]*)\s*\])\s+([a-z])', lambda m: f'{m.group(1)} : {m.group(3)}', s)
    
    # Combine chained quantifiers
    s = combine_chained_quantifiers(s)
    s = re.sub(r'::+', ':', s)
    # Ensure no naked chained quantifier remains (e.g., ?[Y] ?[Z] : ...)
    s = re.sub(r'\?\s*\[\s*([A-Z][A-Za-z0-9]*)\s*\]\s*\?\s*\[\s*([A-Z][A-Za-z0-9]*)\s*\]',
               lambda m: f"? [{m.group(1)}, {m.group(2)}] ", s)
    s = re.sub(r'!\s*\[\s*([A-Z][A-Za-z0-9]*)\s*\]\s*!\s*\[\s*([A-Z][A-Za-z0-9]*)\s*\]',
               lambda m: f"! [{m.group(1)}, {m.group(2)}] ", s)
    
    # Fix variable binding
    s = fix_var_binding(s)
    
    # Unicode operators and arrows
    s = s.replace('—>', '=>').replace('–>', '=>').replace('—', '=>')
    s = s.replace('→', '=>').replace('¬', '~').replace('∨', '|').replace('∧', '&').replace('↔', '<=>')
    # Additional equivalence/arrow unicode variants observed in data
    s = s.replace('⟷', '<=>').replace('⇔', '<=>').replace('⟺', '<=>').replace('≤>', '<=>')
    s = s.replace('≠', ' != ')
    s = re.sub(r'->', '=>', s)
    # Normalize ASCII logical operators occasionally present in data: v for or, ^ for and
    # Only convert when used between formulas (require whitespace or parenthesis context)
    s = re.sub(r'\)\s*v\s*\(', ') | (', s)
    s = re.sub(r'\s+v\s+', ' | ', s)
    s = re.sub(r'\)\s*\^\s*\(', ') & (', s)
    s = re.sub(r'\s+\^\s+', ' & ', s)

    # Fix prioritized patterns
    # Fix misplaced implication inside predicate argument: p((X) => Q) -> ((p(X)) => Q)
    s = re.sub(r'\b([a-z][a-z0-9_]*)\s*\(\s*\(([^)]+)\)\s*=>\s*',
               lambda m: f"(({m.group(1)}({m.group(2)})) => ", s)

    # Fix operator precedence: (~(X=Y) & musician(Y) => love(Y, music)) -> ((~(X=Y) & musician(Y)) => love(Y, music))
    s = re.sub(r'\(~\(([^)]+)\)\s*&\s*([^=]+?)\s*=>\s*([^)]+)\)', r'((~(\1) & \2) => \3)', s)
    s = re.sub(r'&\s*\(~\(([^)]+)\)\s*&\s*([^=]+)\s*=>\s*([^)]+)\)\s*&', r' & (((~(\1) & \2) => \3)) &', s)
    s = re.sub(r'&\s*\(~\(([^)]+)\)\s*&\s*([^=]+)\s*=>\s*([^)]+)\)', r' & ((~(\1) & \2) => \3)', s)
    
    # XOR conversion: (A ⊕ B) -> ((A & ~B) | (~A & B))
    while '⊕' in s:
        xor_pos = s.find('⊕')
        if xor_pos == -1:
            break
        
        if ')) =>' in s[xor_pos:]:
            paren_pos = s.find(')) =>', xor_pos)
            if paren_pos > xor_pos:
                left_paren = -1
                for i in range(xor_pos - 1, -1, -1):
                    if s[i] == '(':
                        paren_depth = 1
                        for j in range(i + 1, len(s)):
                            if s[j] == '(':
                                paren_depth += 1
                            elif s[j] == ')':
                                paren_depth -= 1
                                if paren_depth == 0:
                                    if j > xor_pos and j <= paren_pos:
                                        left_paren = i
                                    break
                        if left_paren >= 0:
                            break
                
                if left_paren >= 0:
                    right_paren = paren_pos + 1
                    inner = s[left_paren+1:right_paren]
                    inner_xor_pos = inner.find('⊕')
                    if inner_xor_pos >= 0:
                        left_part = inner[:inner_xor_pos].strip()
                        right_part = inner[inner_xor_pos+1:].strip()
                        if right_part.endswith(')'):
                            right_part = right_part.rstrip(')').rstrip()
                        replacement = f'(({left_part} & ~{right_part}) | (~{left_part} & {right_part})) =>'
                        s = s[:left_paren] + replacement + s[paren_pos+5:]
                        continue
        
        # Find outermost parentheses containing XOR
        left_paren = -1
        candidates = []
        
        for i in range(xor_pos - 1, -1, -1):
            if s[i] == '(':
                paren_depth = 1
                for j in range(i + 1, len(s)):
                    if s[j] == '(':
                        paren_depth += 1
                    elif s[j] == ')':
                        paren_depth -= 1
                        if paren_depth == 0:
                            if j > xor_pos:
                                candidates.append((i, j))
                            break
            elif s[i] == ')':
                paren_count = 1
                skip_pos = i - 1
                while skip_pos >= 0 and paren_count > 0:
                    if s[skip_pos] == ')':
                        paren_count += 1
                    elif s[skip_pos] == '(':
                        paren_count -= 1
                    skip_pos -= 1
                continue
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            left_paren, _ = candidates[0]
        
        right_paren = -1
        if left_paren >= 0:
            paren_depth = 1
            for i in range(left_paren + 1, min(len(s), left_paren + 200)):
                if s[i] == '(':
                    paren_depth += 1
                elif s[i] == ')':
                    paren_depth -= 1
                    if paren_depth == 0:
                        right_paren = i
                        break
        
        if left_paren >= 0 and right_paren > xor_pos:
            inner = s[left_paren+1:right_paren]
            inner_xor_pos = inner.find('⊕')
            if inner_xor_pos >= 0:
                left_part = inner[:inner_xor_pos].strip()
                right_part = inner[inner_xor_pos+1:].strip()
                replacement = f'(({left_part} & ~{right_part}) | (~{left_part} & {right_part}))'
                
                has_neg = (left_paren > 0 and s[left_paren-1] in ['¬', '~'])
                if has_neg:
                    s = s[:left_paren-1] + s[left_paren-1] + replacement + s[right_paren+1:]
                else:
                    s = s[:left_paren] + replacement + s[right_paren+1:]
                    break
        
        s = s.replace('⊕', '|', 1)
        break
    
    if '⊕' in s:
        s = s.replace('⊕', '|')
    
    if 'XOR' in s:
        s = re.sub(r'\s+XOR\s+', ' | ', s)
    
    # Cleanup
    if s.endswith("."):
        s = s[:-1]
    s = re.sub(r',\s*\)', ')', s)
    s = re.sub(r"([a-z]+\.)'([^']+)'", r"\1_\2", s)
    
    def quote_unquoted_period_ids(match):
        ident = match.group(1)
        if ident.startswith("'") and ident.endswith("'"):
            return ident
        return f"'{ident}'"
    
    s = re.sub(r"(?<!')([a-z][a-z0-9_]*\.[a-zA-Z0-9_.-]+)(?=[,)])", quote_unquoted_period_ids, s)
    
    return s

def _clean_formula(s: str) -> str:
    """Normalize a single FOL formula string for TPTP/FOF."""
    s = _convert_to_tptp(s)
    
    # Fix parentheses balance
    open_parens = s.count('(')
    close_parens = s.count(')')
    
    if close_parens > open_parens:
        extra = close_parens - open_parens
        if s.rstrip().endswith('.'):
            temp = s.rstrip()[:-1].rstrip(')')
            s = temp + ')' * (temp.count('(') - temp.count(')')) + '.'
        else:
            temp = s.rstrip()
            for _ in range(extra):
                if temp.endswith(')'):
                    temp = temp[:-1]
            s = temp + ')' * (open_parens - temp.count('('))
    
    # Fix: ~(...) & ...) => pattern
    if '=>' in s and '~' in s:
        arrow_pos = s.find('=>')
        if arrow_pos > 0:
            before_arrow = s[:arrow_pos].rstrip()
            if before_arrow.endswith('))') and ' & ' in before_arrow:
                open_count = before_arrow.count('(')
                close_count = before_arrow.count(')')
                if before_arrow.startswith('(~') and close_count > open_count:
                    fixed = before_arrow.rstrip(')')
                    s = f"({fixed}){s[arrow_pos:]}"
                elif before_arrow.startswith('~') and not before_arrow.startswith('(~'):
                    s = f"({before_arrow}){s[arrow_pos:]}"
    
    # Wrap if needed
    if not (s.startswith("(") and s.endswith(")")) and not re.match(r'^\s*[!?]\s*\[', s):
        s = f"({s})"
    
    # Final balance check
    open_parens = s.count('(')
    close_parens = s.count(')')
    if open_parens > close_parens:
        missing = open_parens - close_parens
        if s.rstrip().endswith('.'):
            s = s.rstrip()[:-1] + ')' * missing + '.'
        else:
            s = s.rstrip() + ')' * missing
    
    return s

def make_tptp_text(premises: List[str], conclusion: str, emit_conjecture=True) -> str:
    """Build a TPTP string with axioms and a conjecture."""
    lines = []
    for i, pr in enumerate(premises, 1):
        prf = _clean_formula(pr)
        lines.append(f"fof(p{i}, axiom, {prf}).")
    if emit_conjecture:
        cf = _clean_formula(conclusion)
        lines.append(f"fof(goal, conjecture, {cf}).")
    return "\n".join(lines) + "\n"

def make_tptp_with_negated_axiom(premises: List[str], conclusion: str) -> str:
    """Put ¬conclusion as an axiom (no conjecture)."""
    lines = []
    for i, pr in enumerate(premises, 1):
        prf = _clean_formula(pr)
        lines.append(f"fof(p{i}, axiom, {prf}).")
    # Pre-wrap the conclusion to ensure the whole formula is treated as a single unit
    cf = _clean_formula(f"({conclusion})")
    if not cf.startswith('(') or not cf.endswith(')'):
        cf = f"({cf})"
    lines.append(f"fof(neg_goal, axiom, ~{cf}).")
    return "\n".join(lines) + "\n"

def safe_name(s: str) -> str:
    """Sanitize file-friendly identifier."""
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s or "case"

def write_story(outdir: str, story: Dict, emit_neg_axiom: bool):
    sid = safe_name(story.get("id", "story"))
    premises = story.get("premises", [])
    conclusions = story.get("conclusions", [])
    if not premises or not conclusions:
        raise ValueError(f"Story {sid}: missing premises or conclusions.")

    os.makedirs(outdir, exist_ok=True)

    for c in conclusions:
        cid = safe_name(str(c.get("id", "goal")))
        cform = c["formula"]

        conj_text = make_tptp_text(premises, cform, emit_conjecture=True)
        conj_path = os.path.join(outdir, f"{sid}__{cid}.conj.p")
        with open(conj_path, "w") as f:
            f.write(conj_text)

        if emit_neg_axiom:
            neg_text = make_tptp_with_negated_axiom(premises, cform)
            neg_path = os.path.join(outdir, f"{sid}__{cid}.negaxiom.p")
            with open(neg_path, "w") as f:
                f.write(neg_text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", help="Path to JSON (list of stories).")
    ap.add_argument("--outdir", default="./tptp_out", help="Output directory for .p files.")
    ap.add_argument("--emit_negated_axiom", action="store_true",
                    help="Also emit files with ¬conclusion as an axiom.")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Input must be a list of stories or a single story object.")

    for story in data:
        write_story(args.outdir, story, args.emit_negated_axiom)

    print(f"Done. Wrote TPTP files to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
