#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import os
from collections import Counter, defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek


# ---- HW1 solution: config, prompt, helpers --------------------------------
MODEL_NAME = "deepseek-v4-flash-vision-exp"   # 题目指定骨干模型
API_BASE = "https://api.deepseek.com"          # DeepSeek 的 OpenAI 兼容端点
N_VOTES = 3            # 自一致性投票：每张小票跑 3 次，取多数
MAX_RETRIES = 2       # 若某张一次都没解析出 JSON，再补跑几轮

SYSTEM_PROMPT = """You are a precise OCR engine for Hong Kong supermarket receipts.
Read the receipt image and return STRICT JSON only (no markdown, no prose, no code fences):

{{
  "subtotal": <number, the SUBTOTAL line: after all discounts, before ROUNDING>,
  "amount_paid": <number, the final amount actually paid: the payment-method line
                 (OCTOPUS / CASH / EPS / CARD / etc.), taken AFTER any ROUNDING>,
  "rounding": <number, the ROUNDING line (negative if it reduced the total); 0 if absent>,
  "discounts": [<number>, ...]
}}

Rules:
- Output ONLY the JSON object. Nothing else.
- All amounts are HKD as plain numbers, e.g. 102.31. No $ sign, no commas.
- "discounts" lists EVERY discount / promotion / coupon / "-$X" / "X% OFF" line,
  each as a POSITIVE magnitude (e.g. 5.39 for a "-$5.39" line).
- If "rounding" or "discounts" are genuinely absent, use 0 / [] respectively.
- subtotal and amount_paid must always be present and non-null.
"""

EXTRACT_TEXT = "Extract the JSON fields from this receipt image. Output JSON only."


def _parse_receipt_json(text: str) -> dict | None:
    """容错解析：去掉 ```json 围栏、抽取第一个 {...}、校验数值字段。"""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    try:
        return {
            "subtotal": float(data["subtotal"]),
            "amount_paid": float(data["amount_paid"]),
            "rounding": float(data.get("rounding", 0) or 0),
            "discounts": [abs(float(x)) for x in data.get("discounts", [])],
        }
    except (KeyError, ValueError, TypeError):
        return None


def _majority(parsed: list[dict]) -> dict:
    """对多次抽取结果按 (subtotal, amount_paid) 取多数，折扣取该组里的中位和。"""
    key = lambda p: (round(p["subtotal"], 2), round(p["amount_paid"], 2))
    best = Counter(key(p) for p in parsed).most_common(1)[0][0]
    winners = [p for p in parsed if key(p) == best]
    n = len(winners)
    disc_sorted = sorted(sum(w["discounts"]) for w in winners)
    disc_med = disc_sorted[n // 2] if disc_sorted else 0.0
    rep = winners[0]
    rep["discounts_sum"] = disc_med
    return rep


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; put it in .env")
    model = ChatDeepSeek(
        model_name=MODEL_NAME,
        api_key=api_key,
        api_base=API_BASE,
        temperature=0.0,
        max_tokens=4096,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", [
            {"type": "image_url", "image_url": {"url": "{image}"}},
            {"type": "text", "text": EXTRACT_TEXT},
        ]),
    ])
    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    data_urls = [image_data_url(p) for p in images]      # 模板已提供此 helper

    # 一次性把 N_VOTES × 张数 全部 batch 出去（并行，最快）
    reqs, owner = [], []
    for i, url in enumerate(data_urls):
        for _ in range(N_VOTES):
            reqs.append({"image": url})
            owner.append(i)
    raws = chain.batch(reqs)

    # 按"属于哪张小票"归组、解析
    grouped: dict[int, list[dict]] = defaultdict(list)
    for i, r in zip(owner, raws):
        _txt = response_text(r)                                  # 优先取 content
        if not _txt:                                             # 推理模型 content 可能空，兜底取 reasoning_content
            _ak = getattr(r, "additional_kwargs", {}) or {}
            _txt = _ak.get("reasoning_content", "") or ""
        p = _parse_receipt_json(_txt)
        if p is not None:
            grouped[i].append(p)

    total_paid = Decimal("0")
    total_without_discount = Decimal("0")

    for i in range(len(data_urls)):
        parsed = grouped.get(i, [])
        retries = 0
        while not parsed and retries < MAX_RETRIES:
            extra = chain.batch([{"image": data_urls[i]}] * N_VOTES)
            for r in extra:
                p = _parse_receipt_json(response_text(r))
                if p is not None:
                    parsed.append(p)
            retries += 1
        if not parsed:
            raise RuntimeError(f"receipt #{i} 无法解析，请检查图片/prompt")

        rec = _majority(parsed)
        paid = Decimal(str(rec["amount_paid"]))
        subtotal = Decimal(str(rec["subtotal"]))
        discounts_sum = Decimal(str(rec["discounts_sum"]))

        # Q1 = 最终实付(取整后)；Q2 = 小计 + 折扣加回(不加 ROUNDING)
        total_paid += paid
        total_without_discount += subtotal + discounts_sum

    q1 = f"HK${total_paid.quantize(Decimal('0.01'))}"
    q2 = f"HK${total_without_discount.quantize(Decimal('0.01'))}"
    return {QUERY_1: q1, QUERY_2: q2}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
