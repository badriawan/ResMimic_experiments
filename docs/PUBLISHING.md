# Publishing this experiment package

This repository is prepared independently inside `ResMimic/InterPose/custom_data`; it does not change the parent repository's Git history. The publication uses an explicit allowlist so caches, SDKs, local authentication tools, PID state and repeated checkpoints cannot be added accidentally.

## Review the package

```bash
python3 tools/update_publication.py
python3 tools/check_publication.py
git status --short
```

Inspect `publication/files.txt` and `publication/content_manifest.json`. Update the selection intentionally when adding a new result. Review data/asset provenance and [third-party notices](../THIRD_PARTY_NOTICES.md); the code license does not relicense external datasets.

## Authenticate

For creating a new repository through the GitHub CLI, the account owner must complete CLI authentication. Do not paste a password or token into a chat, README or committed file:

```bash
bash tools/github.sh auth login --hostname github.com --git-protocol https --web
bash tools/github.sh auth status
```

The wrapper uses an installed GitHub CLI or the local ignored CLI downloaded from its official release during preparation. On another machine, install [GitHub CLI](https://cli.github.com/) first. Authentication is independent of experiment execution.

For an existing repository, Git authentication supplied by an editor, credential manager or SSH agent may already support pushing even when GitHub CLI is not logged in. After reviewing and committing updates, use the existing configured remote:

```bash
git remote -v
git push origin main
git ls-remote origin refs/heads/main
```

Preserve any existing remote history; do not force-push over someone else's work. The initial `ResMimic_experiments` README commit was retained through a merge before uploading this package.

## Create and push

Once the local allowlisted files are committed, explicitly choose the destination and visibility:

```bash
python3 tools/publish_github.py --repo badriawan/ResMimic_experiments --visibility public
```

Use `--visibility private` for a private repository. The helper requires authentication and a clean reviewed working tree, refuses an unrelated existing destination, creates the GitHub repository, pushes `main` and checks the remote commit SHA. A local Git commit alone is not reported as a successful GitHub publication.

The commands use the official [GitHub CLI repository creation interface](https://cli.github.com/manual/gh_repo_create). Videos and the selected historical checkpoint are ordinary Git files below the [GitHub per-file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github); do not commit all intermediate training checkpoints. Larger future artifacts should be distributed through a deliberate release/LFS workflow.

## Status records

- `publication/github_status.json`: local preparation or repository-creation metadata.
- `publication/last_push_receipt.json`: local mutable receipt generated only after a verified push; excluded from the publication allowlist.
- `publication/validation.json`: preparation checks and their scope, including unexecuted tests.

If credentials or GitHub access are unavailable, the accurate outcome is **local repository prepared; GitHub publication pending authentication**. Do not claim the repository is online until the remote branch check succeeds.
