# ARE Protocols

## 1. Repository model

ARE is a monorepo that vendors the source of:

- `etaoxing/mineral` into `mineral/`
- `rewarped/rewarped` into `rewarped/`

The outer repository is the source of truth for deployment and day-to-day work. Inner repositories should not keep their own `.git` directories inside ARE.

## 2. Initial import protocol

When building ARE from scratch:

1. Clone the upstream repositories into a temporary location.
2. Copy their working trees into `ARE/mineral` and `ARE/rewarped`.
3. Remove any nested `.git` directories before committing to ARE.
4. Commit the imported state in ARE with a message that records the upstream revisions.

Suggested commit message format:

```text
Import mineral <commit> and rewarped <commit>
```

## 3. Environment setup protocol

Use one environment at the ARE root.

Rules:

- use Python 3.10 unless you have a specific reason to diverge
- install both inner projects in editable mode during development
- keep project-level helper scripts at the ARE root

Recommended command:

```bash
./scripts/setup_env.sh
```

To refresh vendored source from GitHub:

```bash
./scripts/import_upstreams.sh
```

## 4. Editable install protocol

Use editable installs for development because:

- code changes in `mineral/` are picked up immediately
- code changes in `rewarped/` are picked up immediately
- outer-level launch scripts can import both packages without reinstalling

Use non-editable installs only for frozen release images or explicitly reproducible packaging workflows.

## 5. Top-level script protocol

Place orchestration scripts in:

- `scripts/` for reusable shell helpers
- repository root for a small number of deployment entrypoints

Top-level scripts should:

- assume they are run from the ARE root
- activate or verify the local environment
- avoid `cd` into inner projects unless needed
- prefer `python -m ...` over direct file execution

## 6. Updating vendored upstream code

When you want newer upstream code from either project:

1. Clone or fetch the upstream repo outside ARE.
2. Compare the desired upstream revision with the current vendored tree.
3. Copy the updated files into `ARE/mineral` or `ARE/rewarped`.
4. Re-run `./scripts/verify_install.sh`.
5. Commit the update with the upstream revision recorded.

Suggested commit message format:

```text
Update mineral to <commit>
Update rewarped to <commit>
```

## 7. Local modification protocol

When making ARE-specific changes:

- commit changes in the outer ARE repo only
- keep experiment scripts and deployment helpers at the outer level
- document any assumptions that differ from upstream in `README.md` or this protocol file

If a change is generally useful beyond ARE, consider upstreaming it later. ARE should still remain independently runnable even if upstream PRs are not merged.

## 8. Verification protocol

After changing setup or vendored code, verify:

```bash
./scripts/verify_install.sh
```

That script checks that:

- the virtual environment exists
- `mineral` imports
- `rewarped` imports

Add heavier runtime checks only after the environment-specific GPU and simulator dependencies are confirmed for the target machine.
