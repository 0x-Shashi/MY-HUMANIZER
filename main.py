"""
My Humanizer — CLI entry point.

Usage:
    python main.py                          # Interactive mode
    python main.py -f input.txt             # Humanize a file
    python main.py -t "text here"           # Humanize inline text
    python main.py -f input.txt -d academic # With domain
    python main.py --analyze "text"         # Detection only (no transform)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import HumanizationPipeline
from core.types import (
    HumanizationConfig,
    Severity,
    WritingDomain,
)
from detection.detector import detect
from retrieval.retriever import HumanCorpusManager


# ── ANSI colors ─────────────────────────────────────────────────────

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def _colored(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"


# ── CLI argument parser ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="my-humanizer",
        description="My Humanizer — 6-pass AI text humanization engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python main.py                           Interactive mode
              python main.py -f essay.txt              Humanize a file
              python main.py -t "your text here"       Humanize inline text
              python main.py -f essay.txt -d academic  Academic domain
              python main.py --analyze "text"          Detection only
              python main.py -f out.txt --stats        Show statistical profile
        """),
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("-f", "--file", type=str, help="Input file path")
    input_group.add_argument("-t", "--text", type=str, help="Input text directly")
    input_group.add_argument(
        "--analyze", type=str,
        help="Analyze text for AI patterns (detection only, no transform)",
    )

    parser.add_argument(
        "-d", "--domain",
        type=str,
        choices=["casual", "academic", "technical", "creative", "business"],
        default="casual",
        help="Writing domain (default: casual)",
    )
    parser.add_argument(
        "-c", "--creativity",
        type=float, default=0.5,
        help="Creativity level 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int, default=3,
        help="Maximum transformation iterations (default: 3)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistical profile",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice/personality injection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible output",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        help="Path to human writing corpus directory",
    )

    return parser


# ── Display functions ────────────────────────────────────────────────

def print_banner():
    banner = f"""
{_colored('╔══════════════════════════════════════════════╗', C.CYAN)}
{_colored('║', C.CYAN)}  {_colored('MY HUMANIZER', C.BOLD + C.WHITE)}  {_colored('v1.0', C.DIM)}                 {_colored('║', C.CYAN)}
{_colored('║', C.CYAN)}  {_colored('6-Pass AI Text Humanization Engine', C.DIM)}        {_colored('║', C.CYAN)}
{_colored('╚══════════════════════════════════════════════╝', C.CYAN)}
"""
    print(banner)


def print_detection_report(result, text: str):
    """Print a detailed detection report."""
    sev_colors = {
        Severity.CRITICAL: C.RED,
        Severity.HIGH: C.YELLOW,
        Severity.MEDIUM: C.MAGENTA,
        Severity.LOW: C.DIM,
    }

    # AI Score bar
    score = result.ai_score
    bar_len = 30
    filled = int(score * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    if score > 0.7:
        bar_color = C.RED
        label = "Likely AI"
    elif score > 0.4:
        bar_color = C.YELLOW
        label = "Mixed signals"
    else:
        bar_color = C.GREEN
        label = "Likely human"

    print(f"\n  {_colored('AI Score:', C.BOLD)} {_colored(bar, bar_color)} {score:.0%} ({label})")

    # Pattern summary
    counts = result.pattern_count_by_severity
    total = len(result.patterns)
    print(f"\n  {_colored('Patterns Found:', C.BOLD)} {total}")
    for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
        count = counts.get(sev, 0)
        if count > 0:
            color = sev_colors[sev]
            print(f"    {_colored(f'● {sev.value.upper()}:', color)} {count}")

    # Top patterns
    if result.patterns:
        print(f"\n  {_colored('Top Issues:', C.BOLD)}")
        shown = set()
        for p in sorted(result.patterns, key=lambda x: [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW].index(x.severity)):
            if p.pattern_id in shown:
                continue
            shown.add(p.pattern_id)
            if len(shown) > 8:
                break
            color = sev_colors.get(p.severity, C.DIM)
            matched = p.matched_text[:40]
            print(f"    {_colored('→', color)} \"{matched}\" — {p.message}")

    # Statistical profile
    stats = result.stats
    print(f"\n  {_colored('Statistical Profile:', C.BOLD)}")
    print(f"    Perplexity:        {stats.perplexity:>8.1f}  {'✓' if 40 < stats.perplexity < 150 else '✗ (human: 40-150)'}")
    print(f"    Burstiness (σ):    {stats.burstiness:>8.1f}  {'✓' if stats.burstiness > 5.0 else '✗ (human: 6-10, AI: 2-4)'}")
    print(f"    Entropy:           {stats.entropy:>8.2f}")
    print(f"    Vocabulary (TTR):  {stats.vocabulary_richness:>8.2f}  {'✓' if stats.vocabulary_richness > 0.62 else '✗ (human: 0.65-0.80)'}")
    print(f"    Connector density: {stats.connector_density:>8.2f}  {'✓' if stats.connector_density < 0.18 else '✗ (human: <0.15)'}")
    print(f"    Repetition:        {stats.repetition_score:>8.3f}")

    # Score breakdown
    if result.breakdown:
        print(f"\n  {_colored('Score Breakdown:', C.BOLD)}")
        for key, value in result.breakdown.items():
            bar_mini = "█" * int(value * 15) + "░" * (15 - int(value * 15))
            print(f"    {key:>12}: {bar_mini} {value:.2f}")


def print_transformation_report(result):
    """Print transformation results."""
    before = result.detection_before.ai_score
    after = result.detection_after.ai_score
    reduction = before - after

    print(f"\n  {_colored('═══ TRANSFORMATION RESULTS ═══', C.BOLD + C.CYAN)}")
    print(f"  AI Score: {_colored(f'{before:.0%}', C.RED)} → {_colored(f'{after:.0%}', C.GREEN)}"
          f"  ({_colored(f'-{reduction:.0%}', C.GREEN)} reduction)")
    print(f"  Changes:  {result.change_count}")
    print(f"  Iterations: {result.iterations}")

    # Before/after stats comparison
    sb = result.detection_before.stats
    sa = result.detection_after.stats
    print(f"\n  {_colored('Metric Comparison:', C.BOLD)}")
    print(f"    {'Metric':<20} {'Before':>10} {'After':>10} {'Target':>12}")
    print(f"    {'─'*54}")
    print(f"    {'Burstiness (σ)':<20} {sb.burstiness:>10.1f} {sa.burstiness:>10.1f} {'6-10':>12}")
    print(f"    {'Vocabulary (TTR)':<20} {sb.vocabulary_richness:>10.2f} {sa.vocabulary_richness:>10.2f} {'0.65-0.80':>12}")
    print(f"    {'Connector density':<20} {sb.connector_density:>10.2f} {sa.connector_density:>10.2f} {'<0.15':>12}")
    print(f"    {'Perplexity':<20} {sb.perplexity:>10.1f} {sa.perplexity:>10.1f} {'40-150':>12}")


def progress_callback(message: str, iteration: int, max_iterations: int):
    """Print progress updates."""
    if max_iterations > 0:
        pct = iteration / max_iterations
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        print(f"  {_colored(bar, C.BLUE)} {message}", end="\r")
    else:
        print(f"  {_colored('→', C.BLUE)} {message}")


# ── Interactive mode ─────────────────────────────────────────────────

def interactive_mode():
    """Run interactive humanization mode."""
    print_banner()
    print(f"  {_colored('Interactive Mode', C.BOLD)} — paste your text below.")
    print(f"  {_colored('Commands:', C.DIM)} /analyze, /domain <name>, /creativity <0-1>, /quit")
    print(f"  {_colored('End input:', C.DIM)} empty line + Enter\n")

    config = HumanizationConfig()

    # Try to load corpus
    corpus = HumanCorpusManager()
    loaded = corpus.load_corpus()
    if loaded > 0:
        config.enable_rag = True
        print(f"  {_colored(f'Loaded {loaded} corpus documents for RAG', C.GREEN)}")

    pipeline = HumanizationPipeline(
        corpus_manager=corpus if config.enable_rag else None,
        on_progress=progress_callback,
    )

    while True:
        print(f"\n{_colored('myhumanizer', C.CYAN)}> ", end="")
        lines = []
        try:
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_colored('Goodbye!', C.DIM)}")
            break

        text = "\n".join(lines).strip()
        if not text:
            continue

        # Handle commands
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/quit" or cmd == "/exit":
                print(f"  {_colored('Goodbye!', C.DIM)}")
                break

            elif cmd == "/analyze":
                if len(parts) < 2:
                    print(f"  {_colored('Usage: /analyze <text>', C.YELLOW)}")
                    continue
                result = detect(parts[1])
                print_detection_report(result, parts[1])
                continue

            elif cmd == "/domain":
                if len(parts) < 2:
                    print(f"  Current domain: {_colored(config.domain.value, C.CYAN)}")
                    continue
                try:
                    config.domain = WritingDomain(parts[1].lower())
                    print(f"  Domain set to: {_colored(config.domain.value, C.GREEN)}")
                except ValueError:
                    print(f"  {_colored('Invalid domain. Options: casual, academic, technical, creative, business', C.RED)}")
                continue

            elif cmd == "/creativity":
                if len(parts) < 2:
                    print(f"  Current creativity: {_colored(str(config.creativity), C.CYAN)}")
                    continue
                try:
                    val = float(parts[1])
                    config.creativity = max(0.0, min(1.0, val))
                    print(f"  Creativity set to: {_colored(str(config.creativity), C.GREEN)}")
                except ValueError:
                    print(f"  {_colored('Invalid value. Use a number between 0.0 and 1.0', C.RED)}")
                continue

            elif cmd == "/stats":
                if len(parts) < 2:
                    print(f"  {_colored('Usage: /stats <text>', C.YELLOW)}")
                    continue
                result = detect(parts[1])
                print_detection_report(result, parts[1])
                continue

            else:
                print(f"  {_colored(f'Unknown command: {cmd}', C.RED)}")
                continue

        # Humanize the text
        print()
        result = pipeline.run(text, config)

        print_transformation_report(result)

        print(f"\n  {_colored('═══ HUMANIZED TEXT ═══', C.BOLD + C.GREEN)}\n")
        # Wrap output nicely
        for para in result.transformed_text.split("\n\n"):
            wrapped = textwrap.fill(para, width=78, initial_indent="  ", subsequent_indent="  ")
            print(wrapped)
            print()


# ── File mode ────────────────────────────────────────────────────────

def file_mode(args):
    """Process a file or inline text."""
    # Get input text
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        text = filepath.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        print("Error: No input provided. Use -f, -t, or run without args for interactive mode.", file=sys.stderr)
        sys.exit(1)

    # Build config
    config = HumanizationConfig(
        domain=WritingDomain(args.domain),
        creativity=args.creativity,
        max_iterations=args.max_iterations,
    )
    if args.no_voice:
        config.voice_profile = "none"

    # Load corpus if specified
    corpus = None
    if args.corpus_dir:
        corpus = HumanCorpusManager(Path(args.corpus_dir))
        loaded = corpus.load_corpus()
        if loaded > 0:
            config.enable_rag = True

    # Run pipeline
    pipeline = HumanizationPipeline(
        corpus_manager=corpus,
        on_progress=progress_callback,
    )
    result = pipeline.run(text, config, seed=args.seed)

    # Output
    print()
    print_transformation_report(result)

    if args.stats:
        print_detection_report(result.detection_after, result.transformed_text)

    output_text = result.transformed_text

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"\n  {_colored(f'Output written to: {args.output}', C.GREEN)}")
    else:
        print(f"\n  {_colored('═══ HUMANIZED TEXT ═══', C.BOLD + C.GREEN)}\n")
        print(output_text)


# ── Analysis mode ────────────────────────────────────────────────────

def analyze_mode(text: str, show_stats: bool = False):
    """Detection-only mode."""
    print_banner()
    result = detect(text)
    print_detection_report(result, text)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.analyze:
        analyze_mode(args.analyze, args.stats)
    elif args.file or args.text:
        file_mode(args)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
