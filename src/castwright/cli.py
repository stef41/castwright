"""Command-line interface for castwright."""

from __future__ import annotations

import sys

try:
    import click
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False

try:
    from rich.console import Console
    _console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    _console = None  # type: ignore[assignment]


def _build_cli():  # type: ignore[no-untyped-def]
    if not _HAS_CLICK:
        return None

    from castwright._types import GenerationConfig, OutputFormat
    from castwright.generate import generate, load_seeds, save_results
    from castwright.providers import MockProvider

    @click.group()
    @click.version_option(package_name="castwright")
    def cli() -> None:
        """castwright -- generate synthetic instruction-tuning data."""

    @cli.command()
    @click.argument("seed_file", type=click.Path(exists=True))
    @click.option("-n", "--count", default=10, type=int, help="Number of examples to generate.")
    @click.option("-m", "--model", default="gpt-4o-mini", help="Model to use.")
    @click.option("-o", "--output", required=True, help="Output JSONL file.")
    @click.option("-t", "--temperature", default=0.9, type=float, help="Sampling temperature.")
    @click.option("-f", "--format", "fmt", default="alpaca",
                  type=click.Choice(["alpaca", "sharegpt", "openai"]),
                  help="Output format.")
    @click.option("--provider", default="openai",
                  type=click.Choice(["openai", "anthropic", "mock"]),
                  help="LLM provider.")
    @click.option("--api-key", default=None, help="API key (overrides env var).")
    @click.option("--base-url", default=None, help="Custom API base URL.")
    @click.option("--diversity", default=0.7, type=float, help="Diversity factor (0.0-1.0).")
    def gen(
        seed_file: str,
        count: int,
        model: str,
        output: str,
        temperature: float,
        fmt: str,
        provider: str,
        api_key: str | None,
        base_url: str | None,
        diversity: float,
    ) -> None:
        """Generate synthetic data from seed examples."""
        seeds = load_seeds(seed_file)
        click.echo(f"Loaded {len(seeds)} seed examples")

        # Create provider
        if provider == "mock":
            llm = MockProvider()
        elif provider == "openai":
            from castwright.providers import OpenAIProvider
            llm = OpenAIProvider(model=model, api_key=api_key, base_url=base_url)
        elif provider == "anthropic":
            from castwright.providers import AnthropicProvider
            llm = AnthropicProvider(model=model, api_key=api_key)
        else:
            click.echo(f"Unknown provider: {provider}", err=True)
            raise SystemExit(1)

        config = GenerationConfig(
            n=count,
            model=model,
            temperature=temperature,
            diversity_factor=diversity,
            output_format=OutputFormat(fmt),
        )

        click.echo(f"Generating {count} examples with {model}...")
        result = generate(seeds, llm, config)

        save_results(result, output, OutputFormat(fmt))
        click.echo(
            f"Generated {result.n_generated}, filtered {result.n_filtered}, "
            f"saved {len(result.examples)} to {output}"
        )
        if result.total_input_tokens > 0:
            click.echo(
                f"Tokens used: {result.total_input_tokens:,} input, "
                f"{result.total_output_tokens:,} output"
            )

    @cli.command()
    @click.argument("seed_file", type=click.Path(exists=True))
    def preview(seed_file: str) -> None:
        """Preview seed examples."""
        seeds = load_seeds(seed_file)
        for i, seed in enumerate(seeds):
            click.echo(f"\n--- Seed {i + 1} ---")
            click.echo(f"Instruction: {seed.instruction}")
            if seed.input:
                click.echo(f"Input: {seed.input}")
            click.echo(f"Output: {seed.output[:200]}{'...' if len(seed.output) > 200 else ''}")

    return cli


cli = _build_cli()


def main() -> None:
    if cli is None:
        print(
            "The CLI requires extra dependencies. Install with:\n"
            "  pip install castwright[cli]",
            file=sys.stderr,
        )
        sys.exit(1)
    cli()


if __name__ == "__main__":
    main()
