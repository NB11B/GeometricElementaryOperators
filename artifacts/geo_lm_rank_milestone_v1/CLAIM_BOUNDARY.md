# Scientific Claim Boundary: GEO LM Rank Milestone v1

## Supported Claims
- **Bigram Superiority**: All 35 tested neural architectures beat the matched bigram baseline across all 5 random seeds.
- **Rank 8 Parity**: GEO rank 8 achieved practical parity with ordinary low rank (median delta -0.0010 BPB, mean delta -0.0050 BPB, 3/5 paired seed wins).
- **Rank 4 Capacity Advantage**: GEO rank 4 won all 5 paired seed comparisons against ordinary low rank (median delta -0.0109 BPB, mean delta -0.0102 BPB).
- **Rank 16 Low-Rank Superiority**: Ordinary low rank won all 5 rank-16 comparisons against GEO rank 16.

## Unsupported Claims (Out of Scope / Unverified)
- Document-level advantage (per-document resolution not measured).
- Source-shifted transfer performance (transfer evaluation not measured).
- Post-hoc systems benchmark results (hardware micro-benchmarks not measured).
- Statistical superiority beyond five paired seeds (directional finding requiring dedicated confirmation).
- Commercial corpus clearance.
