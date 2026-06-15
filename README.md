# IntraKnot Database

A community algorithm database for the [Alice](https://github.com/Ideogenesis-AI/Alice) project managed by [IntraKnot](https://github.com/Ideogenesis-AI/IntraKnot) — a curated collection of reusable algorithm scripts including observables, model definitions, interaction maps, and post-processing utilities.

## How it works

IntraKnot communicates with this database purely over HTTP. It fetches the `registry.yaml` manifest at the repository root to discover available scripts, then fetches individual files on demand. No git cloning is performed on the client side.

Scripts are identified by **source descriptors** of the form `<db-name>:<file>`:

```
intraknot-database:observables/spin_corr.py
intraknot-database:models/bose_hubbard.py
```

The source descriptor is recorded in each campaign's `algorithm.lock`, so the origin of every managed script is always traceable.

## Quick start

```bash
# Register this database in your IntraKnot project.
iknot database add intraknot-database https://github.com/Ideogenesis-AI/IntraKnot-Database

# Browse available scripts.
iknot database list

# Install a script into the active campaign.
iknot campaign install intraknot-database:observables/spin_corr.py

# Refresh the local manifest when new scripts are published.
iknot database update
```

## Repository structure

```
IntraKnot-Database/
├── registry.yaml          # Manifest cataloguing all available scripts
├── observables/           # Observable measurement scripts
├── models/                # Custom Hamiltonian definitions
└── ...                    # Additional categories as needed
```

## The `registry.yaml` manifest

Every database must provide a `registry.yaml` at its root. IntraKnot fetches this file to discover available scripts and uses the `sha256` field for efficient staleness checks without re-downloading file bodies.

```yaml
description: Community algorithm scripts for IntraKnot

categories:
  observables:
    description: Observable measurement scripts
    scripts:
      - name: spin_corr
        file: observables/spin_corr.py
        description: Spin-spin correlation function ⟨SᵢSⱼ⟩
        sha256: <hex digest of the current file content>

  models:
    description: Custom Hamiltonian definitions
    scripts:
      - name: bose_hubbard
        file: models/bose_hubbard.py
        description: Bose-Hubbard model with interaction and hopping terms
        sha256: <hex digest>
```

## Contributing a script

1. Add the script file under the appropriate category directory (e.g. `observables/`, `models/`). Create a new directory for a new category.
2. Compute the SHA-256 digest of the file:
   ```bash
   sha256sum observables/my_script.py
   ```
3. Add an entry to `registry.yaml` under the correct category, filling in `name`, `file`, `description`, and `sha256`.
4. Open a pull request. Once merged to `primary`, the script is immediately available to any IntraKnot project that runs `iknot database update`.

> **Note:** IntraKnot fetches scripts from the `primary` branch. Make sure your file paths in `registry.yaml` match the actual file locations in the repository.

## Integrity

Every `sha256` field in `registry.yaml` must match the SHA-256 digest of the file it references.  Two mechanisms enforce this:

**Local check (before opening a PR)**

Run the verification utility from the repository root:

```bash
python tools/verify_registry.py
```

It prints `OK`, `MISMATCH`, or `MISSING` for each entry and exits non-zero on any failure.  Recompute a hash after editing a file with:

```bash
sha256sum models/my_script.py
```

**CI gate (on every pull request)**

The `verify-registry` GitHub Actions workflow runs `tools/verify_registry.py` automatically on every PR targeting `primary`.  A failing check blocks the merge.

## License

IntraKnot Database is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license. We encourage you to share any improvements you make back to the community, helping IntraKnot grow and benefit all users. See the [LICENSE](LICENSE) file for the full license text. For more information about GPL-3.0, visit https://www.gnu.org/licenses/gpl-3.0.html
