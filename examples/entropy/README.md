# EDT vs Fixed Temperature Comparison

Demonstrates [Entropy-based Dynamic Temperature (EDT)](https://arxiv.org/abs/2403.14541) sampling compared to fixed temperature.

## Run It

```bash
node entropy.mjs
```

## What You'll See

Three prompts comparing fixed T=0.7 vs EDT:

| Prompt Type | Fixed T=0.7 | EDT | Why EDT Helps |
|-------------|-------------|-----|---------------|
| **Factual** "2+2" | Uses T=0.7 (wasteful randomness) | Uses T≈0.04 | Model is confident, don't add noise |
| **Creative** story | T=0.7 (ok) | T varies 0.3-0.9 | Adapts: confident words low T, uncertain words high T |
| **Technical** explanation | Higher entropy | Lower entropy, T≈0.5 | Stays focused on known facts |

## Formula

```
T = T₀ · N^(θ/Entropy)
```

- `T₀=1.0` max temperature
- `N=0.8` base
- `θ=1.5` scale factor
- `Entropy` in nats

## Key Insight

| Entropy | Temperature | Rationale |
|---------|-------------|-----------|
| Low (confident) | Low | Trust the model |
| High (uncertain) | Higher | Explore alternatives |

**Counter-intuitive**: When the model knows the answer, don't add randomness.

## References

- [EDT Paper](https://arxiv.org/abs/2403.14541) - Zhang et al. 2024
- [EAGER](https://arxiv.org/abs/2510.11170) - Entropy-aware inference scaling
- [Locally Typical Sampling](https://arxiv.org/abs/2202.00666) - Information-theoretic approach
