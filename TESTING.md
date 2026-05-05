# Testing protocols

HyperSwarmAgents is a framework — most of its value comes from third-party plugins (Sources, Stores, Syncs, Scopes). Tests have to verify both the framework's interfaces and the plugins that implement them. This doc covers both.

## Running the suite

```bash
pip install -e ".[dev]"
pytest                        # full suite
pytest tests/test_X.py -v     # focused
```

Run before every commit. The framework supports Python 3.9+; if you have multiple versions installed, run the suite against each one you care about.

## What every plugin must test

Plugins live in `hyperswarm/sources/`, `hyperswarm/stores/`, `hyperswarm/syncs/`, `hyperswarm/scopes/`. Each plugin file should have a paired test file under `tests/`. The contract differs per extension point — pick the relevant section below.

### Source plugins

A Source captures session state from a runtime. Tests must cover:

1. **`install()` is idempotent.** Calling it twice does not produce duplicate hooks, wrappers, or files.
2. **`capture(raw)` produces a fully-populated `Entry`.** Given a representative raw payload, the returned Entry has runtime, cwd, summary, body, session_id, and timestamp set.
3. **`capture(raw)` is total.** Empty/missing fields in `raw` produce a sensible Entry, not an exception. Most failures during capture should be logged-and-skipped, not raised — a broken hook should never crash the user's session.
4. **No I/O during capture.** `capture()` is called inside hooks and wrappers; it should not write files, open sockets, or shell out. Any persistence is the orchestrator's job (it calls `Store.write`).

Pattern:

```python
def test_capture_handles_missing_fields():
    src = MyRuntimeSource(config={})
    entry = src.capture({"cwd": "/tmp"})  # minimal payload
    assert entry.runtime == "my-runtime"
    assert entry.cwd == "/tmp"
    # other fields default sensibly, no exception
```

### Store plugins

A Store persists Entries. Tests must cover:

1. **Write/read roundtrip.** `store.read(store.write(entry))` returns an equivalent Entry.
2. **`list_since(t)` filtering.** Entries older than `t` are not yielded; entries at or after `t` are.
3. **Empty / missing root behaves cleanly.** `list_since` on an uninitialized store yields nothing, not an exception.
4. **Append-only semantics.** A second write of an Entry with the same logical content produces a new storage id (no in-place updates).

See `tests/test_markdown_store.py` for the reference shape.

### Sync plugins

A Sync moves entries between nodes. Tests should mock the transport (rsync, S3, git) — don't hit real remotes in unit tests:

1. **`push()` returns the count of entries moved.**
2. **`push()` is idempotent.** Pushing twice with no new entries returns 0 the second time.
3. **`pull()` is symmetric to `push()`.** Same contract, opposite direction.
4. **Network failures are surfaced as exceptions, not silent zeros.** A user reading 0 should know it means "nothing to sync," not "transport broken."

### Scope plugins

A Scope tags entries. Tests must cover:

1. **Rule precedence.** If your plugin has rules with priority (like `path_prefix`'s first-match-wins), test that more-specific rules listed first beat generic ones, AND that the reverse order produces the wrong result. The second test guards against accidental sorting changes.
2. **Fallback behavior.** When no rule matches, the configured fallback (or empty string) is returned.
3. **Config robustness.** Empty config, missing fields, malformed entries don't raise.

See `tests/test_path_prefix_scope.py` for the reference shape.

## Integration tests

Cross-plugin tests live in `tests/integration/` (currently empty — landing in Phase 2). They exercise the full orchestrator: Source → Scope → Store → optionally Sync. Keep these few; rely on per-plugin unit tests for breadth.

## Hooks and wrappers

Source plugins that install hooks (Claude Code) or wrappers (Codex) must include:

- A test that verifies `install()` writes the expected file content
- A test that verifies re-running `install()` doesn't duplicate or corrupt that content
- A test that verifies `install()` rolls back cleanly on failure (e.g., if `~/.claude/settings.json` is malformed, don't write garbage)

## Lint

`pip install ruff` and run `ruff check hyperswarm tests` + `ruff format --check hyperswarm tests` before pushing.

## What we don't test

- Real network calls (mock transports)
- Real Claude Code / Codex / OpenClaw runtimes (mock their input shapes)
- Cross-platform filesystem differences (test the platforms you actually run on)

If a test needs a real runtime to be meaningful, it belongs in `tests/integration/` and should be marked `@pytest.mark.integration` with a skip on missing prerequisites.
