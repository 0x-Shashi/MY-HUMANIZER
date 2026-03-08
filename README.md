# My Humanizer

**6-pass AI text humanization engine** that rewrites AI-generated text to bypass detection tools like GPTZero, ZeroGPT, Turnitin, etc.

> Tested: Takes AI text from **73% detected → 18%** on a 3-paragraph academic essay — 0 patterns remaining, verification passed.

---

## Setup (First Time)

### Requirements

- **Python 3.10+** (check with `python --version`)
- No external dependencies needed for basic mode

### Step 1 — Clone the repo

```bash
git clone https://github.com/0x-Shashi/MY-HUMANIZER.git
cd MY-HUMANIZER
```

### Step 2 — Install enhanced mode (recommended)

This gives much better results with semantic embeddings:

```bash
pip install -r requirements.txt
```

This installs `sentence-transformers`, `faiss-cpu`, and `numpy`. First run will download a small model (~80MB, one time only).

### Step 3 — Run it

```bash
python main.py
```

That's it. You're in interactive mode.

---

## How to Use

### Option A: Interactive Mode (easiest)

```bash
python main.py
```

1. You'll see a `myhumanizer>` prompt
2. Paste your AI-generated text
3. Press **Enter twice** (empty line) to submit
4. Wait a few seconds — the transformed text is printed

```
myhumanizer> Artificial intelligence has fundamentally transformed the landscape of
myhumanizer> modern technology. Moreover, it is crucial to understand...
myhumanizer>

=== TRANSFORMATION RESULTS ===
AI Score: 73% → 18%
Patterns: 39 → 0
Changes: 40

=== OUTPUT ===
Artificial intelligence has at its core transformed the field of modern technology...
```

#### Interactive Commands

| Command | What it does |
|---------|-------------|
| `/analyze <text>` | Check AI score without transforming |
| `/domain academic` | Switch to academic mode (default) |
| `/domain casual` | Switch to casual mode |
| `/creativity 0.8` | Set creativity 0.0–1.0 (higher = more changes) |
| `/stats <text>` | Show burstiness, TTR, connector density |
| `/quit` | Exit |

### Option B: Humanize a File

Write or paste your essay into a `.txt` file, then:

```bash
# Humanize an essay (academic mode is default)
python main.py -f essay.txt

# Save output to a new file
python main.py -f essay.txt -o humanized.txt

# With higher creativity (more aggressive changes)
python main.py -f essay.txt -c 0.8

# With more iteration passes (slower but better)
python main.py -f essay.txt --max-iterations 5
```

### Option C: Quick Inline Text

```bash
python main.py -t "Your AI text goes here in quotes"
```

### Option D: Just Check the AI Score (no changes)

```bash
python main.py --analyze "Your text here"
python main.py --analyze -f essay.txt
```

---

## For College Reports (Academic Mode)

This tool is tuned for **academic writing by default**. When you humanize text, it:

- Replaces AI-favorite words with natural academic vocabulary
- Strips formulaic connectors ("Moreover", "Furthermore", "Additionally")
- Removes filler phrases ("It is important to note that", "It is crucial to understand")
- Adds natural hedging ("arguably", "to some extent", "one could argue")
- Inserts academic-appropriate sentence fragments for burstiness
- Varies sentence length to match human writing patterns
- Runs verification to ensure the output actually passes detection

### Workflow for a College Report

1. **Write your report with ChatGPT/Claude** — don't worry about AI detection yet
2. **Save it** as `report.txt`
3. **Humanize it:**
   ```bash
   python main.py -f report.txt -o report_final.txt --max-iterations 5
   ```
4. **Check the output** — read `report_final.txt` and fix anything that reads weird
5. **Verify** — paste the final text into [GPTZero](https://gptzero.me/) or [ZeroGPT](https://www.zerogpt.com/) to confirm

### Tips for Best Results

- **Longer text = better results** — the engine works best with 200+ words
- **Creativity 0.6–0.8** is the sweet spot for academic writing
- **Read the output** — occasionally a sentence might sound awkward, just manually fix it
- **Run it twice** if the score is still above 25% — copy the output, paste it back in
- **Don't paste the entire paper at once** — do it section by section (intro, body, conclusion) for more variety

---

## All CLI Options

```
python main.py [options]

Options:
  -f, --file FILE         Input file path (.txt)
  -t, --text TEXT         Input text directly in quotes
  -o, --output FILE       Save output to file
  --analyze               Detection only (no transformation)
  -d, --domain DOMAIN     Writing domain: academic, casual, technical, creative, business
  -c, --creativity FLOAT  Creativity level 0.0–1.0 (default: 0.7)
  --max-iterations INT    Max transformation passes (default: 5)
  --stats                 Show detailed statistical profile
  --no-voice              Disable voice/personality injection
  --seed INT              Random seed for reproducible results
  --corpus-dir DIR        Path to human writing corpus for RAG
```

---

## Domains

| Domain | Best for |
|--------|----------|
| `academic` | College reports, essays, research papers (default) |
| `casual` | Blog posts, social media, informal writing |
| `technical` | Documentation, technical reports, code reviews |
| `creative` | Stories, creative writing, personal narratives |
| `business` | Business reports, emails, proposals |

Switch domain:
```bash
python main.py -f essay.txt -d casual
```

---

## What Detectors Actually Measure (and how My Humanizer beats them)

| Metric | AI Text | Human Text | What My Humanizer Does |
|--------|---------|------------|-----------------|
| Burstiness (sentence length variation) | σ = 2–4 (uniform) | σ = 6–10 (varied) | Inserts fragments, varies length |
| Vocabulary richness (TTR) | 0.50–0.65 | 0.65–0.80 | Diverse word replacements |
| Connector density | > 0.20 | < 0.15 | Strips "Moreover", "Furthermore" etc. |
| Perplexity | 10–40 (predictable) | 50–150 (surprising) | Injects less-predictable word choices |
| Pattern frequency | 137+ AI tells | Near zero | Replaces all flagged words/phrases |

---

## Project Structure

```
MY-HUMANIZER/
├── main.py                          # CLI — start here
├── requirements.txt                 # pip install -r requirements.txt
├── core/
│   ├── pipeline.py                  # 6-pass orchestrator (the brain)
│   └── types.py                     # Configuration & domain types
├── detection/
│   ├── detector.py                  # Unified detection engine
│   ├── statistical_analyzer.py      # Burstiness, TTR, perplexity scoring
│   └── patterns/
│       ├── lexical.py               # 140+ word/phrase AI patterns
│       ├── structural.py            # Sentence structure patterns
│       └── communication.py         # Chatbot artifact detection
├── transformation/
│   ├── word_level.py                # Domain-aware word replacement
│   ├── sentence_level.py            # Sentence restructuring
│   ├── burstiness_engine.py         # Sentence length variation
│   └── voice_injector.py            # Human personality injection
├── retrieval/
│   └── retriever.py                 # RAG corpus manager
├── verification/
│   └── verifier.py                  # Adversarial verification loop
└── utils/
    └── nlp.py                       # Sentence splitting utilities
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'sentence_transformers'"**
→ Run `pip install -r requirements.txt`

**First run is slow**
→ It downloads the embedding model (~80MB) once. After that it's cached.

**AI score still high?**
→ Try `--max-iterations 5 -c 0.8` or run the output through a second pass.

**Weird sentence?**
→ The engine occasionally reorders clauses awkwardly. Just manually fix that one sentence.

**Want even better results?**
→ Add human writing samples to `retrieval/corpus/` folder (`.txt` files of real human writing in your style).

---

## License

MIT
