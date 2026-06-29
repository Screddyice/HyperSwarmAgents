"""Shared helper: assemble a compact Team Nebula company snapshot from NEBOS.

NEBOS is MCP-only for the ``nebos_`` bearer, but ``/api/mcp`` accepts a STATELESS
single JSON-RPC ``tools/call`` POST (no initialize handshake, no session, no SSE)
— so we fan out a few read-only tool calls and assemble the snapshot client-side.
Used by BOTH Hermes instances as a session-start company-context layer
(personal: folded into the Jarvis provider; TMN: the standalone nebos provider).

Creds are reused from the already-configured ``mcp_servers.nebos`` block in
$HERMES_HOME/config.yaml (url + Bearer) — no new secret handling. Fail-soft:
any error yields "" (or a partial block); never raises.
"""
from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def _nebos_creds(hermes_home: str) -> tuple[str | None, str | None]:
    path = os.path.join(hermes_home or os.path.expanduser("~/.hermes"), "config.yaml")
    try:
        import yaml
        data = yaml.safe_load(open(path, encoding="utf-8")) or {}
        nb = (data.get("mcp_servers") or {}).get("nebos") or {}
        url = nb.get("url")
        auth = (nb.get("headers") or {}).get("Authorization", "")
        token = auth.split("Bearer ", 1)[1].strip() if "Bearer " in auth else (auth or None)
        return url, token
    except Exception:
        return None, None


def _call(url: str, token: str, name: str, arguments: dict, timeout: float):
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    # tool result is JSON-stringified into result.content[0].text
    txt = (((d.get("result") or {}).get("content") or [{}])[0]).get("text")
    return json.loads(txt) if txt else None


def _money(n) -> str:
    try:
        return f"${int(n):,}"
    except Exception:
        return str(n)


def company_snapshot(hermes_home: str, *, timeout: float = 4.0, char_cap: int = 1800) -> str:
    """Return a compact Team Nebula company-state block, or "" on failure."""
    url, token = _nebos_creds(hermes_home)
    if not url or not token:
        return ""

    calls = {
        "pipeline": ("pipeline_summary", {}),
        "clients": ("client_list", {"limit": 8}),
        "meetings": ("meeting_list", {"limit": 5}),
        "alerts": ("alert_list", {"active": True}),
    }
    out: dict = {}

    def run(key):
        name, args = calls[key]
        try:
            return key, _call(url, token, name, args, timeout)
        except Exception:
            return key, None

    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            for key, val in ex.map(run, list(calls)):
                out[key] = val
    except Exception:
        return ""

    lines: list[str] = []

    pl = out.get("pipeline") or {}
    if isinstance(pl, dict) and pl:
        by = pl.get("byStage") or {}
        stages = ", ".join(
            f"{s} {v.get('count')}" for s, v in by.items() if isinstance(v, dict)
        )
        lines.append(
            f"Pipeline: {pl.get('totalClients','?')} clients · {_money(pl.get('totalMrr',0))} MRR"
            + (f" (stages: {stages})" if stages else "")
        )

    def _datestr(v) -> str:
        # NEBOS dates may be ISO strings or timestamp-ish objects; coerce safely.
        return v[:10] if isinstance(v, str) else ""

    cl = out.get("clients")
    if isinstance(cl, list) and cl:
        try:
            def health(c):
                try:
                    return float(c.get("health"))
                except Exception:
                    return 999.0
            rows = sorted([c for c in cl if isinstance(c, dict)], key=health)[:6]
            cli = "; ".join(
                f"{c.get('name')} (health {c.get('health','?')}, {c.get('pipelineStage','?')}"
                + (f", {_money(c.get('mrr'))}" if c.get('mrr') else "") + ")"
                for c in rows
            )
            if cli:
                lines.append("Clients: " + cli)
        except Exception:
            pass

    mt = out.get("meetings")
    if isinstance(mt, list) and mt:
        try:
            ms = "; ".join(
                f"{_datestr(m.get('date'))} {m.get('title') or m.get('clientName') or ''}".strip()
                for m in mt[:4] if isinstance(m, dict)
            )
            if ms:
                lines.append("Recent meetings: " + ms)
        except Exception:
            pass

    al = out.get("alerts")
    if isinstance(al, list) and al:
        try:
            tops = "; ".join(
                f"{a.get('clientName') or a.get('clientId','?')}: {a.get('type','alert')}"
                for a in al[:4] if isinstance(a, dict)
            )
            lines.append(f"Open alerts ({len(al)}): {tops}")
        except Exception:
            pass

    if not lines:
        return ""
    block = "Team Nebula — live company state (NEBOS):\n" + "\n".join(f"- {ln}" for ln in lines)
    return block[:char_cap]
