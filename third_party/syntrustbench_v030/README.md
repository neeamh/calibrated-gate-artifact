# Frozen dependency: prior gate implementation, v0.3.0

This folder holds the public release of the prior, fixed-threshold gate that the paper evaluates as
"frozen v0.3" (cited in the paper as the five-dimension gate with provisional fixed thresholds). It is included at commit
`3dd0f7ce088994d84f4925ca7e0949d7e0ecd3d8` so that reviewers do not need network access to install it.

Only package metadata was changed for double-blind review: author names, copyright holders and
repository URLs are redacted in `pyproject.toml` and `LICENSE`. Every file under `src/` is identical
to that commit.

Install with:

    pip install --no-deps third_party/syntrustbench_v030
