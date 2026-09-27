# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution

### Approach in one sentence

Each receipt image is sent individually to the vision model only to **extract
structured fields** as JSON; all arithmetic (per-receipt subtotals and the two
final totals) is done in Python with `Decimal`, so the model never does math
and the only uncertainty is OCR.

### Chain design (`build_chain`)

- **Model:** `ChatDeepSeek(model_name="deepseek-v4-flash-vision-exp",
  api_base="https://api.deepseek.com", temperature=0)` — the required vision
  Flash model via DeepSeek's OpenAI-compatible endpoint, temperature 0 for
  stable extraction.
- **Chain:** a `ChatPromptTemplate` with a strict *system* message that forces
  the model to reply with **only** a JSON object (no prose, no code fences) and
  a *human* multimodal message that carries the receipt image as a data URL
  (via the provided `image_data_url` helper). The chain is a single
  `model | parser` step — no routing or reflection was needed because a
  well-constrained extraction prompt is already reliable.

### Field schema (per receipt)

```json
{
  "subtotal":   <the SUBTOTAL line, after discounts, before rounding>,
  "amount_paid":<the final payment line, AFTER rounding — Octopus/cash/card>,
  "rounding":   <the ROUNDING line, used only for sanity checks>,
  "discounts":  [<absolute values of every promotion/coupon/member/app/
                 packaging-damage/percentage discount line>]
}
```

### Query logic (`answer_queries`)

For each receipt the chain is invoked; the returned JSON is cleaned (strip
`$`, `,`, `HK`) and parsed into `Decimal`. Then:

- **Q1 (total spent)** = Σ over receipts of `amount_paid`
  (the final payment *after* the rounding line).
- **Q2 (without discount)** = Σ over receipts of `subtotal + Σ(discounts)`.
  Rounding is **never** added back, matching the homework definition.

Both totals are formatted as `HK$X.XX`, each response containing exactly one
number as required by the grader's `parse_single_amount`.

### Reliability techniques

- **Self-consistency:** each image is run `N_VOTES=3` times in parallel via
  `chain.batch`. The `subtotal`/`amount_paid` pair is reduced by majority vote;
  the discount list is reduced by per-slot median. A disagreement triggers one
  extra run. This cancels occasional OCR slips on individual receipts.
- **Retries:** JSON parse failures are retried up to `MAX_RETRIES` times before
  falling back to a zero-field record (so one bad image never aborts the run).
- **Deterministic math:** all sums use `Decimal` to avoid float drift, and Q2's
  "add back every discount, never add back rounding" rule is encoded directly in
  code rather than asked of the model.
- **No hard-coding:** no filenames or public answers appear anywhere; the
  chain answers purely from image content, so it generalises to unseen receipt
  folders used for grading.

### Reproducibility

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=sk-your-key" > .env   # .env is gitignored
python3 hw1.py --image-folder public_test
cat results.csv
```

Expected on `public_test` (see `public_test/ground_truth.json`):

```
query,model_response,correctness
How much money did I spend in total for these bills?,HK$1974.30,correct
How much would I have had to pay without the discount?,HK$2348.20,correct
```

### Pipeline diagram

```
 receipt_i.png
       │  image_data_url()
       ▼
 ┌───────────────────────────────────────────┐
 │  build_chain()                            │
 │  system: "reply JSON only, fields below"  │
 │  human : [image data URL]                 │
 │  model  : deepseek-v4-flash-vision-exp    │
 └───────────────────────────────────────────┘
       │  ×3 parallel (self-consistency)
       ▼
   {subtotal, amount_paid, rounding, discounts[]}
       │  majority vote + median
       ▼
   per-receipt Decimal fields
       │
       ├──► Q1 = Σ amount_paid           ──► "HK$1974.30"
       └──► Q2 = Σ(subtotal + Σdiscounts) ──► "HK$2348.20"
```

