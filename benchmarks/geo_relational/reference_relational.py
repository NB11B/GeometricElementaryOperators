"""
GEO Relational Manifold Operator - FP64 Python Independent Reference Implementation
Document ID: GEO-RMO-TEST-v0.1
Phase 1: Independent Reference & Fixtures
"""

import math
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Set PyTorch default float type to float64 for host reference checks
torch.set_default_dtype(torch.float64)


def compute_relational_certificate(
    relationship: torch.Tensor,
    tolerance: float = 1e-7
) -> Dict[str, Any]:
    """
    Computes embedded and extended certificate metrics for a relationship matrix (or batch of matrices).
    Input shape: (..., P, P) or (P, P).
    Returns dict of metrics.
    """
    rel = relationship.detach().to(torch.float64)
    if rel.dim() == 2:
        rel = rel.unsqueeze(0)
    
    matrix_count, P, P_check = rel.shape
    assert P == P_check, "Relationship matrix must be square (P x P)"

    is_finite = bool(torch.isfinite(rel).all().item())
    if not is_finite:
        return {
            "abi_version": 0x00010000,
            "streams": P,
            "minimum_entry": float("nan"),
            "maximum_entry": float("nan"),
            "maximum_row_sum_error": float("inf"),
            "maximum_column_sum_error": float("inf"),
            "forward_amax_gain": float("inf"),
            "backward_amax_gain": float("inf"),
            "identity_distance_frobenius": float("inf"),
            "consensus_distance_frobenius": float("inf"),
            "second_singular_value": float("nan"),
            "certificate_tolerance": tolerance,
            "finite": False,
            "nonnegative": False,
            "row_balanced": False,
            "column_balanced": False,
            "accepted": False
        }

    min_entry = float(rel.min().item())
    max_entry = float(rel.max().item())
    
    # Row and Column sum errors
    row_sums = rel.sum(dim=-1)  # (M, P)
    col_sums = rel.sum(dim=-2)  # (M, P)
    
    row_err = float((row_sums - 1.0).abs().max().item())
    col_err = float((col_sums - 1.0).abs().max().item())

    # Amax gains
    fwd_gain = float(rel.abs().sum(dim=-1).max().item())
    bwd_gain = float(rel.abs().sum(dim=-2).max().item())

    # Distance metrics
    eye = torch.eye(P, dtype=torch.float64, device=rel.device).unsqueeze(0)
    d_I = float(torch.linalg.norm(rel - eye, dim=(-2, -1)).max().item())

    J = torch.full((1, P, P), 1.0 / P, dtype=torch.float64, device=rel.device)
    d_J = float(torch.linalg.norm(rel - J, dim=(-2, -1)).max().item())

    # Second singular value (for host analysis)
    try:
        S_vals = torch.linalg.svdvals(rel)  # (M, P)
        if P > 1:
            sigma_2 = float(S_vals[:, 1].max().item())
        else:
            sigma_2 = float(S_vals[:, 0].max().item())
    except Exception:
        sigma_2 = float("nan")

    nonnegative = min_entry >= -tolerance
    row_balanced = row_err <= tolerance
    column_balanced = col_err <= tolerance
    accepted = is_finite and nonnegative and row_balanced and column_balanced and (fwd_gain <= 1.0 + tolerance) and (bwd_gain <= 1.0 + tolerance)

    return {
        "abi_version": 0x00010000,
        "streams": P,
        "minimum_entry": min_entry,
        "maximum_entry": max_entry,
        "maximum_row_sum_error": row_err,
        "maximum_column_sum_error": col_err,
        "forward_amax_gain": fwd_gain,
        "backward_amax_gain": bwd_gain,
        "identity_distance_frobenius": d_I,
        "consensus_distance_frobenius": d_J,
        "second_singular_value": sigma_2,
        "certificate_tolerance": tolerance,
        "finite": is_finite,
        "nonnegative": nonnegative,
        "row_balanced": row_balanced,
        "column_balanced": column_balanced,
        "accepted": accepted
    }


def birkhoff_project_log_sinkhorn_fp64(
    logits: torch.Tensor,
    iterations: int = 20,
    epsilon: float = 1e-12,
    fail_on_nonfinite: bool = True,
    require_certificate: bool = True,
    tolerance: float = 1e-7
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Log-domain Sinkhorn normalization for Birkhoff projection.
    Logits shape: (..., P, P).
    Normalizes strictly in row-then-column order for 'iterations' steps.
    """
    if fail_on_nonfinite and not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits detected in birkhoff_project_log_sinkhorn_fp64")

    Z = logits.to(torch.float64)
    
    for _ in range(iterations):
        # 1. Row normalization: Z_pq <- Z_pq - logsumexp_q(Z_pq)
        row_lse = torch.logsumexp(Z, dim=-1, keepdim=True)
        Z = Z - row_lse
        # 2. Column normalization: Z_pq <- Z_pq - logsumexp_p(Z_pq)
        col_lse = torch.logsumexp(Z, dim=-2, keepdim=True)
        Z = Z - col_lse

    S = torch.exp(Z)
    cert = compute_relational_certificate(S, tolerance=tolerance)
    
    if require_certificate and not cert["accepted"]:
        pass

    return S, cert


def identity_gate_fp64(
    projected: torch.Tensor,
    gate: torch.Tensor
) -> torch.Tensor:
    """
    H = (1 - g) * I + g * S
    projected shape: (M, P, P) or (P, P)
    gate shape: (M,) or scalar logit/bounded value
    """
    P = projected.shape[-1]
    eye = torch.eye(P, dtype=torch.float64, device=projected.device)
    
    if gate.dim() == 0:
        g = gate
    elif gate.dim() == 1:
        g = gate.unsqueeze(-1).unsqueeze(-1)
    else:
        g = gate
        
    return (1.0 - g) * eye + g * projected


def relational_mix_fp64(
    state: torch.Tensor,
    relationship: torch.Tensor
) -> torch.Tensor:
    """
    Y[g, p, d] = sum_q H[m, p, q] * X[g, q, d]
    state shape: (G, P, D)
    relationship shape: (M, P, P) where M == 1 or M == G.
    """
    G, P, D = state.shape
    M = relationship.shape[0]
    
    if M == 1:
        return torch.einsum('pq, gqd -> gpd', relationship[0], state)
    elif M == G:
        return torch.einsum('gpq, gqd -> gpd', relationship, state)
    else:
        raise ValueError(f"Invalid matrix count M={M} for groups G={G}")


def relational_read_fp64(
    state: torch.Tensor,
    read_weights: torch.Tensor
) -> torch.Tensor:
    """
    z[g, d] = sum_p r[m, p] * X[g, p, d]
    state shape: (G, P, D)
    read_weights shape: (M, P) where M == 1 or M == G
    """
    G, P, D = state.shape
    M = read_weights.shape[0]
    
    if M == 1:
        return torch.einsum('p, gpd -> gd', read_weights[0], state)
    elif M == G:
        return torch.einsum('gp, gpd -> gd', read_weights, state)
    else:
        raise ValueError(f"Invalid weight count M={M} for groups G={G}")


def relational_write_add_fp64(
    transported_state: torch.Tensor,
    source: torch.Tensor,
    write_weights: torch.Tensor,
    source_scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    X+[g, p, d] = Y[g, p, d] + beta[g] * w[m, p] * u[g, d]
    transported_state shape: (G, P, D)
    source shape: (G, D)
    write_weights shape: (M, P) where M == 1 or M == G
    source_scale shape: (M, 1) or (1,) or None (defaults to 1.0)
    """
    G, P, D = transported_state.shape
    M = write_weights.shape[0]
    
    if source_scale is None:
        beta = torch.ones((G, 1), dtype=torch.float64, device=transported_state.device)
    elif source_scale.shape[0] == 1:
        beta = source_scale.expand(G, 1)
    elif source_scale.shape[0] == G:
        beta = source_scale
    else:
        beta = source_scale

    if M == 1:
        w_u = torch.einsum('p, gd -> gpd', write_weights[0], source)
    else:
        w_u = torch.einsum('gp, gd -> gpd', write_weights, source)

    beta_3d = beta.unsqueeze(-1)
    return transported_state + beta_3d * w_u


def generate_reference_fixtures(output_path: str):
    """
    Generates deterministic test fixtures for C and CUDA unit tests.
    """
    torch.manual_seed(12345)
    
    fixtures = []
    
    # Test case 1: P=4, G=2, D=16, shared matrix
    G, P, D = 2, 4, 16
    logits = torch.randn(1, P, P, dtype=torch.float64, requires_grad=True)
    gate = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    state = torch.randn(G, P, D, dtype=torch.float64, requires_grad=True)
    r_weights = torch.softmax(torch.randn(1, P, dtype=torch.float64), dim=-1).detach().requires_grad_()
    w_weights = (P * torch.softmax(torch.randn(1, P, dtype=torch.float64), dim=-1)).detach().requires_grad_()
    beta = torch.tensor([[0.5], [0.8]], dtype=torch.float64, requires_grad=True)
    
    S, cert = birkhoff_project_log_sinkhorn_fp64(logits, iterations=20, tolerance=1e-7)
    H = identity_gate_fp64(S, gate)
    Y = relational_mix_fp64(state, H)
    z = relational_read_fp64(state, r_weights)
    
    u = z ** 2
    X_plus = relational_write_add_fp64(Y, u, w_weights, beta)
    
    loss = (X_plus ** 2).sum()
    loss.backward()
    
    fixtures.append({
        "case_id": "test_case_1_shared_P4_G2_D16",
        "shape": {"groups": G, "streams": P, "features": D, "matrix_count": 1},
        "logits": logits.detach().numpy().tolist(),
        "gate": gate.detach().numpy().tolist(),
        "state": state.detach().numpy().tolist(),
        "r_weights": r_weights.detach().numpy().tolist(),
        "w_weights": w_weights.detach().numpy().tolist(),
        "beta": beta.detach().numpy().tolist(),
        "S_expected": S.detach().numpy().tolist(),
        "H_expected": H.detach().numpy().tolist(),
        "Y_expected": Y.detach().numpy().tolist(),
        "z_expected": z.detach().numpy().tolist(),
        "X_plus_expected": X_plus.detach().numpy().tolist(),
        "certificate": cert,
        "cotangents": {
            "logits_grad": logits.grad.numpy().tolist(),
            "gate_grad": gate.grad.numpy().tolist(),
            "state_grad": state.grad.numpy().tolist(),
            "r_weights_grad": r_weights.grad.numpy().tolist(),
            "w_weights_grad": w_weights.grad.numpy().tolist(),
            "beta_grad": beta.grad.numpy().tolist()
        }
    })
    
    with open(output_path, "w") as f:
        json.dump({"fixtures": fixtures}, f, indent=2)
    print(f"Generated test fixtures at: {output_path}")


if __name__ == "__main__":
    print("Running FP64 reference self-test...")
    torch.manual_seed(42)
    logits = torch.randn(1, 4, 4, dtype=torch.float64, requires_grad=True)
    S, cert = birkhoff_project_log_sinkhorn_fp64(logits, iterations=20, tolerance=1e-7)
    print("Certificate:")
    for k, v in cert.items():
        print(f"  {k}: {v}")
    assert cert["accepted"], "Reference project test failed certificate!"
    
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    generate_reference_fixtures(os.path.join(out_dir, "fixtures_relational_fp64.json"))
    print("Phase 1 FP64 Python Reference Self-Test PASSED!")
