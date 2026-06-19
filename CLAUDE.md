# CLAUDE.md — Instructions for Claude Code

This file is read automatically at the start of every session in this repository.
Use it to set preferences, conventions, and constraints.

---

## Language & Style

- This is a Julia project. All source code is Julia unless otherwise noted.
- Follow existing code style in each file rather than imposing a uniform style
  across the whole codebase.
- Do not add docstrings, comments, or type annotations to code you did not change.
- Do not refactor surrounding code when fixing a bug or adding a feature — keep
  changes focused.
- Do not add or remove any whitespaces except in the lines you are editing already.
- Try to keep lines under 80 characters, with 92 characters the absolute maximum.

---

## Workflow

- Do not commit changes unless explicitly asked to.
- Push to remote when appropriate; ask when in doubt.
- Before editing a file, read it first.
- Prefer editing existing files over creating new ones.

---

## Julia-specific

- Prefer in-place (`!`) variants of functions when performance matters.
- Do not introduce type instabilities. If a change may affect type inference,
  note it explicitly.
- Type annotations are for dispatch only, write code as type agnostic as possible, in particular don't hard-code floating point types
- Do not add dependencies to `Project.toml` without asking first.
- GPU kernels use KernelAbstractions — do not introduce CUDA.jl-specific code
  in shared paths.
- It is ok to use features from the latest Julia versions. If you do, provide brief summaries and comments.

---

## Commit style

- Use short imperative commit messages (e.g. "Remove dead simpletrans.jl").
- Record co-authorship: add a Co-Authored-By trailer when Claude authors or co-authors the commit.
- For pull requests, keep the explanations brief unless prompted to give more details

---

## What NOT to do

- Do not silently change behaviour — if a refactor changes observable output,
  flag it before proceeding.
- Do not delete files without confirmation, even if they appear unused.
- Do not open pull requests without being asked.
