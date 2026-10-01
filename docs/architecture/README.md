# Explore the architecture

**[Download the interactive HTML](https://github.com/haarikaalla/indic-tokenizer-decoding/raw/refs/heads/main/docs/architecture/tokenizer-architecture.html)**, save it, then open it in a browser. It is self-contained; Archify is not needed to view it. Source-code links require an internet connection.

The repository keeps one overall architecture map, with three representations:

| File | Purpose |
|---|---|
| [tokenizer-architecture.html](tokenizer-architecture.html) | Official Archify viewer with interactive controls |
| [tokenizer-architecture.png](tokenizer-architecture.png) | Static preview for the GitHub README |
| [tokenizer-architecture.json](tokenizer-architecture.json) | Editable typed source used by Archify |

GitHub's file viewer shows HTML source and cannot execute the viewer. Use **Download raw file**, then open the saved file. A PNG cannot provide interactive controls.

## Explore it

| Action | Control |
|---|---|
| Inspect a component and its source references | Click a node |
| Find a component | Search or `/` |
| Explore a directed route | **PATH** or `R` |
| Compare component types | **LENS** or `L` |
| Change theme | Theme switch or `T` |
| Enter presentation mode | Fullscreen button or `F` |
| Save an image | **Export** |
| See available controls | `?` |

## Provenance

The checked-in HTML is preserved byte-for-byte from the successful [official Archify build](https://github.com/haarikaalla/indic-tokenizer-decoding/actions/runs/36851307779), artifact `archify-generated-diagrams` (ID `11155152853`). That workflow checked out `tt-a1i/archify` at `v3.0.1`, rendered the typed JSON with its CLI, and passed the HTML checks before capturing the preview.

The source links are pinned to repository commit `5bb428e9b1c8e031e6d78ef8caa8a9ad26614fc8`. Archify renders an agent-authored specification; it does not independently discover or verify every architectural claim. Source references let readers inspect those claims.

The map covers runtime generation, offline training, file-based model artifacts, and evaluation tooling. Its generic storage symbols refer to files, not a deployed database. Token generation strategies and SentencePiece conversion back to text are separate operations.

## Regenerate

Use an isolated checkout of the official renderer; it remains documentation tooling only:

```bash
git clone --branch v3.0.1 --depth 1 https://github.com/tt-a1i/archify.git /tmp/archify
npm ci --prefix /tmp/archify/archify

# Run from this repository root.
node /tmp/archify/archify/bin/archify.mjs render architecture \
  docs/architecture/tokenizer-architecture.json \
  docs/architecture/tokenizer-architecture.html --repo-root .
node /tmp/archify/archify/bin/archify.mjs check \
  docs/architecture/tokenizer-architecture.html
```

The [documentation workflow](../../.github/workflows/archify-docs.yml) renders and checks this one diagram, captures its PNG, and uploads both files as `archify-generated-diagrams`. After editing the JSON, replace the checked-in HTML and PNG with the outputs from the successful run so downloads and previews stay aligned. Update the provenance above to identify that run.

For new topology or layout, use Archify's `finalize architecture … --repo-root . --quality showcase --json` workflow and review its diagnostics before publishing. The historical build above used `render` and `check`; it is not a full `finalize` receipt.

Archify adds no Python runtime dependency. The diagram cleanup changes documentation only.
