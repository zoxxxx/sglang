#!/usr/bin/env python3
"""Benchmark Host KV Pool with deterministic Prefill and Decode DP ranks."""

import argparse
import asyncio
import json
import math
import statistics
import time
import urllib.parse
import uuid
from pathlib import Path

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--bootstrap-host")
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--prefill-dp-rank", type=int, default=0)
    parser.add_argument("--decode-dp-rank", type=int, default=0)
    parser.add_argument("--model", default="default")
    parser.add_argument("--prompt-chars", type=int, default=3007)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(index, 0)]


def make_prompt(index: int, prompt_chars: int) -> str:
    # Put the unique portion first so requests cannot reuse the benchmark body.
    prefix = f"host-kv-fixed-rank request={index} nonce={uuid.uuid4().hex}\n"
    unit = (
        "Measure host centric KV ownership while prefill publishes cache pages and "
        "decode temporarily attaches them from shared host memory. "
    )
    return (prefix + unit * math.ceil(prompt_chars / len(unit)))[:prompt_chars]


async def post_json(
    session: aiohttp.ClientSession, url: str, payload: dict
) -> tuple[int, dict | str, float]:
    started = time.perf_counter()
    async with session.post(url, json=payload) as response:
        raw = await response.read()
        elapsed = time.perf_counter() - started
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = raw.decode("utf-8", errors="replace")
        return response.status, body, elapsed


async def run_one(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    index: int,
    bootstrap_host: str,
) -> dict:
    host_kv_id = f"host-kv-bench-{uuid.uuid4()}"
    bootstrap_room = uuid.uuid4().int & ((1 << 63) - 1)
    common = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": make_prompt(index, args.prompt_chars)}
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": args.bootstrap_port,
        "bootstrap_room": bootstrap_room,
        "host_kv_id": host_kv_id,
    }
    prefill_payload = {**common, "routed_dp_rank": args.prefill_dp_rank}
    decode_payload = {
        **common,
        "routed_dp_rank": args.decode_dp_rank,
        "disagg_prefill_dp_rank": args.prefill_dp_rank,
    }

    total_started = time.perf_counter()
    prefill_status, prefill_body, prefill_s = await post_json(
        session, f"{args.prefill_url.rstrip('/')}/v1/chat/completions", prefill_payload
    )
    if prefill_status >= 400:
        return {
            "index": index,
            "ok": False,
            "stage": "prefill",
            "status": prefill_status,
            "body": prefill_body,
            "prefill_s": prefill_s,
            "total_s": time.perf_counter() - total_started,
            "host_kv_id": host_kv_id,
            "bootstrap_room": bootstrap_room,
        }

    decode_status, decode_body, decode_s = await post_json(
        session, f"{args.decode_url.rstrip('/')}/v1/chat/completions", decode_payload
    )
    total_s = time.perf_counter() - total_started
    usage = decode_body.get("usage", {}) if isinstance(decode_body, dict) else {}
    meta_info = (
        decode_body.get("meta_info", {}) if isinstance(decode_body, dict) else {}
    )
    return {
        "index": index,
        "ok": decode_status < 400,
        "stage": "decode",
        "status": decode_status,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "usage": usage,
        "dp_rank": meta_info.get("dp_rank"),
        "host_kv_id": host_kv_id,
        "bootstrap_room": bootstrap_room,
        **({"body": decode_body} if decode_status >= 400 else {}),
    }


def summarize(samples: list[dict], wall_s: float) -> dict:
    good = [sample for sample in samples if sample["ok"]]
    summary = {
        "requests": len(samples),
        "ok": len(good),
        "failed": len(samples) - len(good),
        "wall_s": wall_s,
        "rps": len(good) / wall_s if wall_s else None,
    }
    for field in ("prefill_s", "decode_s", "total_s"):
        values = [sample[field] for sample in good if field in sample]
        summary[field] = {
            "avg": statistics.fmean(values) if values else None,
            "p50": statistics.median(values) if values else None,
            "p95": percentile(values, 0.95),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


async def main() -> int:
    args = parse_args()
    bootstrap_host = (
        args.bootstrap_host or urllib.parse.urlparse(args.prefill_url).hostname
    )
    if not bootstrap_host:
        raise ValueError("Cannot infer --bootstrap-host from --prefill-url")
    if args.concurrency <= 0 or args.requests <= 0 or args.warmup < 0:
        raise ValueError(
            "concurrency/requests must be positive and warmup non-negative"
        )

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency, 1) * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        warmup_semaphore = asyncio.Semaphore(args.concurrency)

        async def run_warmup(index: int) -> dict:
            async with warmup_semaphore:
                result = await run_one(session, args, index, bootstrap_host)
                print(
                    f"warmup {index + args.warmup + 1}/{args.warmup}: "
                    f"ok={result['ok']} total={result.get('total_s', 0):.3f}s",
                    flush=True,
                )
                return result

        warmup = await asyncio.gather(
            *(run_warmup(index) for index in range(-args.warmup, 0))
        )

        samples: list[dict] = []
        if all(result["ok"] for result in warmup):
            semaphore = asyncio.Semaphore(args.concurrency)

            async def guarded(index: int) -> dict:
                async with semaphore:
                    result = await run_one(session, args, index, bootstrap_host)
                    print(
                        f"request {index + 1}/{args.requests}: ok={result['ok']} "
                        f"prefill={result.get('prefill_s', 0):.3f}s "
                        f"decode={result.get('decode_s', 0):.3f}s "
                        f"total={result.get('total_s', 0):.3f}s",
                        flush=True,
                    )
                    return result

            started = time.perf_counter()
            samples = await asyncio.gather(
                *(guarded(index) for index in range(args.requests))
            )
            wall_s = time.perf_counter() - started
        else:
            wall_s = 0.0

    result = {
        "config": {
            "prefill_url": args.prefill_url,
            "decode_url": args.decode_url,
            "bootstrap_host": bootstrap_host,
            "bootstrap_port": args.bootstrap_port,
            "prefill_dp_rank": args.prefill_dp_rank,
            "decode_dp_rank": args.decode_dp_rank,
            "model": args.model,
            "prompt_chars": args.prompt_chars,
            "max_tokens": args.max_tokens,
            "warmup": args.warmup,
            "requests": args.requests,
            "concurrency": args.concurrency,
        },
        "warmup": warmup,
        "samples": samples,
        "summary": summarize(samples, wall_s),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if samples and all(sample["ok"] for sample in samples) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
