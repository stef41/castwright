"""Tests for castwright.cli."""


import pytest

try:
    from click.testing import CliRunner
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False

from castwright.cli import _build_cli

pytestmark = pytest.mark.skipif(not _HAS_CLICK, reason="click not installed")


@pytest.fixture
def cli():
    c = _build_cli()
    assert c is not None
    return c


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def seed_file(tmp_path):
    f = tmp_path / "seeds.jsonl"
    f.write_text(
        '{"instruction": "Explain the concept of recursion in programming", "output": "Recursion is a programming technique where a function calls itself to solve a problem by breaking it into smaller subproblems."}\n'
        '{"instruction": "What is the difference between a stack and a queue?", "output": "A stack follows LIFO (Last In First Out) while a queue follows FIFO (First In First Out)."}\n'
    )
    return str(f)


class TestCLIGen:
    def test_gen_mock_provider(self, cli, runner, seed_file, tmp_path):
        out = tmp_path / "output.jsonl"
        result = runner.invoke(cli, [
            "gen", seed_file,
            "-n", "3",
            "-o", str(out),
            "--provider", "mock",
        ])
        assert result.exit_code == 0, result.output
        assert "Loaded 2 seed examples" in result.output
        assert "saved" in result.output.lower() or "Generated" in result.output

    def test_gen_creates_output(self, cli, runner, seed_file, tmp_path):
        out = tmp_path / "output.jsonl"
        result = runner.invoke(cli, [
            "gen", seed_file,
            "-n", "2",
            "-o", str(out),
            "--provider", "mock",
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_gen_custom_format(self, cli, runner, seed_file, tmp_path):
        out = tmp_path / "output.jsonl"
        result = runner.invoke(cli, [
            "gen", seed_file,
            "-n", "2",
            "-o", str(out),
            "--provider", "mock",
            "-f", "sharegpt",
        ])
        assert result.exit_code == 0, result.output

    def test_gen_openai_format(self, cli, runner, seed_file, tmp_path):
        out = tmp_path / "output.jsonl"
        result = runner.invoke(cli, [
            "gen", seed_file,
            "-n", "2",
            "-o", str(out),
            "--provider", "mock",
            "-f", "openai",
        ])
        assert result.exit_code == 0, result.output

    def test_gen_with_temperature(self, cli, runner, seed_file, tmp_path):
        out = tmp_path / "output.jsonl"
        result = runner.invoke(cli, [
            "gen", seed_file,
            "-n", "2",
            "-o", str(out),
            "--provider", "mock",
            "-t", "0.5",
        ])
        assert result.exit_code == 0, result.output

    def test_gen_with_diversity(self, cli, runner, seed_file, tmp_path):
        out = tmp_path / "output.jsonl"
        result = runner.invoke(cli, [
            "gen", seed_file,
            "-n", "2",
            "-o", str(out),
            "--provider", "mock",
            "--diversity", "0.3",
        ])
        assert result.exit_code == 0, result.output

    def test_gen_nonexistent_seed_file(self, cli, runner, tmp_path):
        out = tmp_path / "output.jsonl"
        result = runner.invoke(cli, [
            "gen", "/nonexistent/seeds.jsonl",
            "-n", "2",
            "-o", str(out),
            "--provider", "mock",
        ])
        assert result.exit_code != 0


class TestCLIPreview:
    def test_preview(self, cli, runner, seed_file):
        result = runner.invoke(cli, ["preview", seed_file])
        assert result.exit_code == 0, result.output
        assert "Seed 1" in result.output
        assert "Seed 2" in result.output
        assert "recursion" in result.output.lower()

    def test_preview_nonexistent(self, cli, runner):
        result = runner.invoke(cli, ["preview", "/nonexistent/seeds.jsonl"])
        assert result.exit_code != 0


class TestCLIGroup:
    def test_help(self, cli, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "castwright" in result.output

    def test_gen_help(self, cli, runner):
        result = runner.invoke(cli, ["gen", "--help"])
        assert result.exit_code == 0
        assert "--count" in result.output or "-n" in result.output

    def test_preview_help(self, cli, runner):
        result = runner.invoke(cli, ["preview", "--help"])
        assert result.exit_code == 0
