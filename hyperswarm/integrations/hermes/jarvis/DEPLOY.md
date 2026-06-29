# Jarvis + NEBOS Hermes memory providers — deploy

These are Hermes memory-provider plugins for Shawn's assistants (built 2026-06-29/30).
Source of truth = this repo; the box runs deployed copies (private box, scp deploy).

## jarvis (personal Hermes)
- Box: `/home/ubuntu/hermes-jarvis/hermes/{__init__.py,plugin.yaml,provider.py,nebos_context.py}`,
  symlinked `~/.hermes/plugins/jarvis`. Config `memory.provider: jarvis` + a `jarvis:` block.
- Recall + write-back over the shawn-corpus REST API (`:8210`, `JARVIS_API_TOKEN` in `~/.hermes/.env`),
  session-start NEBOS company snapshot, and a `coding_leftoff` tool over the local HyperSwarm store.

## nebos (TMN Hermes, `tmn` profile)
- Box: `/home/ubuntu/hermes-nebos/hermes/{__init__.py,plugin.yaml,provider.py}` + `nebos_context.py`
  (a SYMLINK to the jarvis copy on the box — single source). Symlink `~/.hermes/profiles/tmn/plugins/nebos`.
  Config `memory.provider: nebos`.
- Session-start Team Nebula company snapshot (read-only); creds reused from `mcp_servers.nebos`.

## Apply
scp changed files to the box paths above, then `systemctl --user restart hermes-gateway.service`
(personal) / `hermes-gateway-tmn.service` (TMN). Provider/tool-schema changes need a restart.

## Rollback
personal: `include_nebos_context: false` in the `jarvis:` block, or `memory.provider: hyperswarm`
(restore `config.yaml.bak-prejarvis`). tmn: restore `config.yaml.bak-prenebos`. Restart the gateway.
