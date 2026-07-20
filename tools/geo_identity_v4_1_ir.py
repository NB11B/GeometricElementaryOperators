#!/usr/bin/env python3
"""V4.1 compatibility facade over the native V4.2 fixed-blade engine.

On the V4.2 branch this module no longer patches the accepted legacy backends.
Existing V4.1 corpus tools keep their import surface while using the native
validator, exact polynomial backend, compiler DAG, Python evaluator, and
host/CUDA emitter.
"""
from __future__ import annotations

try:
    import geo_identity_v4_2_compiler as compiler
    import geo_identity_v4_2_exact as exact
except ModuleNotFoundError:
    from tools import geo_identity_v4_2_compiler as compiler
    from tools import geo_identity_v4_2_exact as exact

# Compatibility names used by the accepted V4.1 tests and tools.
if not hasattr(exact, "_validate_expression"):
    exact._validate_expression = exact.validate_expression

parse_fixed_blade = exact.parse_fixed_blade
validate_spec = exact.validate_spec
extract_polynomial = exact.extract_polynomial
load_identity = compiler.load_identity
load_corpus = compiler.load_corpus
evaluate_identity = compiler.evaluate_identity
emit_header = compiler.emit_header
IdentityError = compiler.IdentityError
DiscoveryError = exact.DiscoveryError


def install() -> None:
    """Retained as an idempotent no-op for V4.1 callers."""
    return None
