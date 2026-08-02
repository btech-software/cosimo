"""Robust numeric tokenizer shared by the sanitizer and verification harness.

Handles thousands separators (commas) and decimals so that "1,430,000.00" is
parsed as a single number 1430000.0 rather than being split into [1,430,0.0].
"""
import re

# Full number tokens: optional sign, digit groups (with commas), optional decimal.
TOKEN = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?')

def val(tok):
    return float(tok.replace(',', ''))

def nums(s):
    return [val(t) for t in TOKEN.findall(str(s))]

def has_decimal(tok):
    return '.' in tok

def has_comma(tok):
    return ',' in tok

def fmt(tok, scaled):
    """Format a scaled value, preserving the original token's style."""
    if has_decimal(tok):
        return f"{scaled:,.2f}"
    if has_comma(tok):
        return f"{int(round(scaled)):,}"
    return f"{int(round(scaled))}"

def scale_str(s, factor):
    """Scale every number token in s by factor, preserving surrounding text."""
    def repl(m):
        return fmt(m.group(), val(m.group()) * factor)
    return TOKEN.sub(repl, s)
