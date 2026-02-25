#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests


DEFAULT_BASE_URL = "https://chatgpt.com"


@dataclass
class Message:
    created_at: Optional[int]
    role: str
    text: str


def _extract_text_parts(content: Any) -> str:
    """
    ChatGPT message 'content' is typically:
      {"content_type":"text","parts":[...]}
    but can also include other shapes. We try to be robust.
    """
    if content is None:
        return ""

    # Common: {"content_type":"text","parts":[...]}
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts if p is not None).strip()

        # Sometimes: {"text": "..."} or nested shapes
        if "text" in content and isinstance(content["text"], str):
            return content["text"].strip()

        # Fallback: stringify
        return json.dumps(content, ensure_ascii=False)

    # Rare: content is already a string/list
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(x) for x in content if x is not None).strip()

    return str(content).strip()


def _collect_messages(conversation: Dict[str, Any]) -> List[Message]:
    """
    Conversation JSON has a 'mapping' dict of nodes.
    Each node can contain a 'message' with author.role and content.
    We collect message nodes and sort by created_at.
    """
    mapping = conversation.get("mapping", {})
    msgs: List[Message] = []

    if not isinstance(mapping, dict):
        return msgs

    for _node_id, node in mapping.items():
        if not isinstance(node, dict):
            continue
        message = node.get("message")
        if not isinstance(message, dict):
            continue

        author = message.get("author", {})
        role = "unknown"
        if isinstance(author, dict):
            role = author.get("role") or role

        content = message.get("content")
        text = _extract_text_parts(content)

        # Skip empty “system-ish” nodes or placeholders
        if not text:
            continue

        created_at = message.get("create_time")
        if created_at is not None:
            try:
                created_at = int(created_at)
            except Exception:
                created_at = None

        msgs.append(Message(created_at=created_at, role=str(role), text=text))

    # Sort by timestamp (None last, stable)
    msgs.sort(key=lambda m: (m.created_at is None, m.created_at or 0))
    return msgs


def _ts_to_iso(ts: Optional[int]) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _to_markdown(title: str, messages: List[Message]) -> str:
    lines: List[str] = []
    lines.append(f"# {title}".strip() if title else "# Conversation")
    lines.append("")
    lines.append(f"_Exported: {datetime.now(tz=timezone.utc).isoformat()}_")
    lines.append("")

    for m in messages:
        role = m.role
        header = role.upper()
        t = _ts_to_iso(m.created_at)
        if t:
            header += f" ({t})"

        lines.append(f"## {header}")
        lines.append("")
        # Keep message text verbatim
        lines.append(m.text.rstrip())
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def fetch_conversation(
    conversation_id: str,
    base_url: str,
    bearer_token: Optional[str],
    cookie: Optional[str],
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Fetch conversation JSON from:
      GET /backend-api/conversation/<id>
    """
    url = urljoin(base_url.rstrip("/") + "/", f"backend-api/conversation/{conversation_id}")

    headers = {
        "User-Agent": "chat-export/1.0",
        "Accept": "application/json",
    }

    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token.strip()}"

    cookies = {}
    if cookie:
        # Allow either "name=value" or a full Cookie header string
        if "=" in cookie and ";" not in cookie:
            name, value = cookie.split("=", 1)
            cookies[name.strip()] = value.strip()
        else:
            headers["Cookie"] = cookie

    r = requests.get(url, headers=headers, cookies=cookies, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def main() -> int:
    p = argparse.ArgumentParser(description="Export a single ChatGPT conversation to Markdown.")
    p.add_argument("conversation_id", help="Conversation ID (the <id> part of /c/<id>)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Default: {DEFAULT_BASE_URL}")
    p.add_argument("--out-md", default=None, help="Markdown output path (default: <id>.md)")
    p.add_argument("--out-json", default=None, help="Optional: also save raw JSON to this path")
    p.add_argument("--bearer", default=os.getenv("CHATGPT_BEARER"), help="Bearer token (or set CHATGPT_BEARER)")
    p.add_argument(
        "--cookie",
        default=os.getenv("CHATGPT_COOKIE"),
        help="Cookie string or single name=value (or set CHATGPT_COOKIE)",
    )
    args = p.parse_args()

    if not args.bearer and not args.cookie:
        print(
            "Error: provide --bearer or --cookie (or env CHATGPT_BEARER / CHATGPT_COOKIE) for auth.",
            file=sys.stderr,
        )
        return 2

    convo = fetch_conversation(
        conversation_id=args.conversation_id,
        base_url=args.base_url,
        bearer_token=args.bearer,
        cookie=args.cookie,
    )

    title = convo.get("title") or ""
    messages = _collect_messages(convo)

    out_md = args.out_md or f"{args.conversation_id}.md"
    md = _to_markdown(title=title, messages=messages)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote Markdown: {out_md} ({len(messages)} messages)")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(convo, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
