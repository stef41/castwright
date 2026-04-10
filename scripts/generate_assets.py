"""Generate SVG assets for README."""
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def gen_generation_report():
    console = Console(record=True, width=95)

    table = Table(title="Generation Report — castwright run #47")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    rows = [
        ("Seed examples", "25"),
        ("Target count", "2,000"),
        ("Generated (raw)", "2,312"),
        ("Filtered (low quality)", "187"),
        ("Filtered (duplicates)", "94"),
        ("Filtered (too similar to seed)", "31"),
        ("Final examples", Text("2,000", style="green bold")),
        ("Provider", "gpt-4o-mini"),
        ("Total tokens used", "4.2M"),
        ("Generation cost", "$0.63"),
        ("Cost per example", "$0.000315"),
        ("Elapsed time", "8m 42s"),
        ("Avg quality score", Text("0.91", style="green bold")),
    ]
    for metric, value in rows:
        table.add_row(metric, value)

    console.print(table)
    svg = console.export_svg(title="castwright — generation report")
    Path("assets/generation_report.svg").write_text(svg)
    print(f"  generation_report.svg: {len(svg)//1024}KB")


def gen_quality_distribution():
    console = Console(record=True, width=95)

    table = Table(title="Quality Distribution — 2,000 generated examples")
    table.add_column("Quality Bucket", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Pct", justify="right")
    table.add_column("Distribution", min_width=30)

    rows = [
        ("0.95 – 1.00  Excellent", "412", "20.6%", Text("█████████████████████" , style="green bold")),
        ("0.90 – 0.95  Good", "687", "34.4%", Text("██████████████████████████████████" , style="green")),
        ("0.85 – 0.90  Acceptable", "534", "26.7%", Text("███████████████████████████" , style="yellow")),
        ("0.80 – 0.85  Marginal", "241", "12.1%", Text("████████████" , style="yellow")),
        ("0.70 – 0.80  Low", "98", "4.9%", Text("█████" , style="red")),
        ("< 0.70       Rejected", "28", "1.4%", Text("█" , style="red dim")),
    ]
    for bucket, count, pct, dist in rows:
        table.add_row(bucket, count, pct, dist)

    console.print(table)
    svg = console.export_svg(title="castwright — quality distribution")
    Path("assets/quality_distribution.svg").write_text(svg)
    print(f"  quality_distribution.svg: {len(svg)//1024}KB")


if __name__ == "__main__":
    gen_generation_report()
    gen_quality_distribution()
    print("Done.")
