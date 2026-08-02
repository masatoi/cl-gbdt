# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Guidelines

@prompts/repl-driven-development.md
@prompts/common-lisp-expert.md

### How those prompts apply here

Both prompts were imported verbatim from the `cl-mcp` project and describe *its*
environment, not this one. Where they conflict with this file, **this file wins**:

- **cl-mcp tools require a running server.** `repl-driven-development.md` assumes the MCP
  tools (`repl-eval`, `lisp-edit-form`, `clgrep-search`, `run-tests`, …) are always
  available. Here they come from the HTTP server declared in `.mcp.json` — see
  [MCP Setup](#mcp-setup). If the server is down, the tools are absent; fall back to
  `Read`/`Edit`/`Grep`/`Bash` rather than treating its shell-command prohibition as
  binding. The underlying **EXPLORE → EXPERIMENT → PERSIST → VERIFY** loop always
  applies: prototype in the REPL, then persist to `src/`.
- **The test framework is rove**, matching the prompt's assumption, so `run-tests` and
  `(rove:run-test ...)` work as described.
- **FFI is expected.** LightGBM and XGBoost are C libraries; `common-lisp-expert.md`'s
  portable-CL guidance yields to whatever CFFI needs (raw pointers, foreign arrays,
  manual lifetime management). Wrap unsafe foreign calls, don't avoid them.

The style rules in `common-lisp-expert.md` (Google CL Style Guide, 2-space indent,
≤100 columns, lisp-case naming, `;;;`/`;;`/`;` comment hierarchy) do apply.

## Project Overview

`cl-gbdt` is a Common Lisp wrapper library for gradient boosting decision tree
implementations — **LightGBM** and **XGBoost** — exposing a single shared high-level API
over both backends.

**Status: skeleton only.** `src/main.lisp` and `tests/main.lisp` are the generated stubs;
`cl-gbdt.asd` has no dependencies yet. Nothing below describes existing code — design
decisions (FFI vs. subprocess, backend dispatch, data representation) are still open.

## MCP Setup

`.mcp.json` points at a cl-mcp HTTP server on port 3001. Start it from a REPL that has
cl-mcp available and leave it running:

```lisp
(asdf:load-system :cl-mcp)  ; or (ql:quickload :cl-mcp)
(cl-mcp:start-http-server :port 3001)
;; => serving at http://127.0.0.1:3001/mcp; the REPL stays usable
```

The server must be up *before* Claude Code starts, otherwise the `cl-mcp` tools will not
be registered for the session. Source lives at `~/cl-mcp`
(`~/.roswell/local-projects/cl-ai-project/cl-mcp`).

`.mcp.json` is machine-local configuration — keep it out of version control
(see `.gitignore`).

## Testing & Linting

Load and test:

```lisp
(ql:quickload :cl-gbdt)
(asdf:test-system :cl-gbdt)
```

Via the MCP `run-tests` tool, the system name is `cl-gbdt/tests`. Single test from the
REPL: `(rove:run-test 'cl-gbdt/tests/main::some-test)`.

Before committing:

```lisp
(asdf:compile-system :cl-gbdt :force t)  ; surfaces warnings
```

```bash
mallet src/*.lisp
```

## Code Style

- Follow the Google Common Lisp Style Guide
- 2-space indent, ≤100 columns
- Blank line between top-level forms
- Lower-case lisp-case: `my-function`, `*special*`, `+constant+`, `something-p`
- Docstrings required for public functions/classes
- Each file starts with `(in-package ...)`

## Repository Structure

```
src/       Core implementation
tests/     Rove test suites (mirrored naming)
prompts/   System prompts for AI agents (imported from cl-mcp)
```
