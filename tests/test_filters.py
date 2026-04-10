"""Tests for castwright.filters."""

from castwright._types import GeneratedExample
from castwright.filters import (
    DEFAULT_FILTERS,
    _check_balanced_formatting,
    _check_min_length,
    _check_no_meta_talk,
    _check_not_empty,
    _check_not_refusal,
    _check_not_repetitive,
    deduplicate_generated,
    filter_examples,
)


def _ex(instruction: str = "Valid question here", output: str = "A valid detailed answer") -> GeneratedExample:
    return GeneratedExample(instruction=instruction, output=output)


# --- Individual filters ---


class TestCheckNotEmpty:
    def test_valid(self):
        assert _check_not_empty(_ex()) is True

    def test_empty_instruction(self):
        assert _check_not_empty(_ex(instruction="")) is False

    def test_empty_output(self):
        assert _check_not_empty(_ex(output="")) is False

    def test_whitespace_only_instruction(self):
        assert _check_not_empty(_ex(instruction="   ")) is False

    def test_whitespace_only_output(self):
        assert _check_not_empty(_ex(output="  \n  ")) is False

    def test_both_empty(self):
        assert _check_not_empty(_ex(instruction="", output="")) is False


class TestCheckMinLength:
    def test_long_enough(self):
        assert _check_min_length(_ex(instruction="This is a long enough question")) is True

    def test_too_short(self):
        assert _check_min_length(_ex(instruction="Hi")) is False

    def test_exactly_10_chars(self):
        assert _check_min_length(_ex(instruction="1234567890")) is True

    def test_9_chars(self):
        assert _check_min_length(_ex(instruction="123456789")) is False

    def test_whitespace_stripped(self):
        assert _check_min_length(_ex(instruction="   short   ")) is False


class TestCheckNotRepetitive:
    def test_normal_text(self):
        ex = _ex(output="The quick brown fox jumps over the lazy dog")
        assert _check_not_repetitive(ex) is True

    def test_highly_repetitive(self):
        ex = _ex(output="word word word word word word word word word word")
        assert _check_not_repetitive(ex) is False

    def test_short_text_passes(self):
        ex = _ex(output="hi hi hi")
        assert _check_not_repetitive(ex) is True  # fewer than 5 words

    def test_some_repetition_ok(self):
        ex = _ex(output="the cat sat on the mat with the cat nearby")
        assert _check_not_repetitive(ex) is True


class TestCheckNotRefusal:
    def test_normal(self):
        assert _check_not_refusal(_ex(output="Here is the explanation...")) is True

    def test_sorry_cant(self):
        assert _check_not_refusal(_ex(output="Sorry, I can't help with that")) is False

    def test_im_sorry(self):
        assert _check_not_refusal(_ex(output="I'm sorry, I'm unable to assist")) is False

    def test_as_an_ai(self):
        assert _check_not_refusal(_ex(output="As an AI language model, I cannot...")) is False

    def test_unfortunately(self):
        assert _check_not_refusal(_ex(output="Unfortunately, I cannot help with that")) is False

    def test_sorry_midsentence_ok(self):
        # "sorry" mid-sentence shouldn't trigger
        assert _check_not_refusal(_ex(output="The algorithm is sorry to report errors when...")) is True

    def test_i_am_unable(self):
        assert _check_not_refusal(_ex(output="I am unable to process that request")) is False


class TestCheckNoMetaTalk:
    def test_normal(self):
        assert _check_no_meta_talk(_ex()) is True

    def test_here_is_an_example(self):
        assert _check_no_meta_talk(_ex(output="Here's an example instruction for you")) is False

    def test_as_requested(self):
        assert _check_no_meta_talk(_ex(output="As requested, here it is")) is False

    def test_ill_generate(self):
        assert _check_no_meta_talk(_ex(output="I'll generate something now")) is False

    def test_here_are_examples(self):
        assert _check_no_meta_talk(_ex(instruction="Here are 5 examples for training")) is False

    def test_example_in_content_ok(self):
        # "example" in normal content shouldn't be caught by all patterns
        ex = _ex(output="For example, consider the following approach")
        assert _check_no_meta_talk(ex) is True


class TestCheckBalancedFormatting:
    def test_no_code_blocks(self):
        assert _check_balanced_formatting(_ex()) is True

    def test_balanced_code_block(self):
        ex = _ex(output="Here:\n```python\nprint('hi')\n```")
        assert _check_balanced_formatting(ex) is True

    def test_unbalanced_code_block(self):
        ex = _ex(output="Here:\n```python\nprint('hi')")
        assert _check_balanced_formatting(ex) is False

    def test_multiple_balanced(self):
        ex = _ex(output="```a```\nand\n```b```")
        assert _check_balanced_formatting(ex) is True

    def test_three_code_fences(self):
        ex = _ex(output="```\nfirst\n```\nsome text\n```\nunclosed")
        assert _check_balanced_formatting(ex) is False


# --- filter_examples ---


class TestFilterExamples:
    def test_good_examples_pass(self):
        examples = [
            _ex(instruction="Explain quantum computing", output="Quantum computing uses qubits..."),
            _ex(instruction="What is machine learning?", output="Machine learning is a branch of AI..."),
        ]
        result = filter_examples(examples)
        assert len(result) == 2

    def test_bad_examples_filtered(self):
        examples = [
            _ex(instruction="", output="A"),  # empty instruction
            _ex(instruction="Hi", output="A"),  # too short
            _ex(instruction="Valid question here?", output="I'm sorry, I can't help"),  # refusal
            _ex(instruction="Valid question here?", output="word word word word word word word word word word"),  # repetitive
        ]
        result = filter_examples(examples)
        assert len(result) == 0

    def test_custom_filter(self):
        examples = [
            _ex(instruction="Short one here?", output="Short"),
            _ex(instruction="Long one question?", output="A" * 200),
        ]

        def long_output(ex: GeneratedExample) -> bool:
            return len(ex.output) > 50

        result = filter_examples(examples, filters=[long_output])
        assert len(result) == 1

    def test_default_filters_list(self):
        assert len(DEFAULT_FILTERS) == 6

    def test_empty_input(self):
        assert filter_examples([]) == []


# --- deduplicate_generated ---


class TestDeduplicateGenerated:
    def test_removes_duplicates(self):
        examples = [
            _ex(instruction="What is X?", output="A1"),
            _ex(instruction="What is X?", output="A2"),
            _ex(instruction="What is Y?", output="A3"),
        ]
        result = deduplicate_generated(examples)
        assert len(result) == 2

    def test_case_insensitive(self):
        examples = [
            _ex(instruction="What is X?", output="A1"),
            _ex(instruction="what is x?", output="A2"),
        ]
        result = deduplicate_generated(examples)
        assert len(result) == 1

    def test_strips_whitespace(self):
        examples = [
            _ex(instruction="  What is X?  ", output="A1"),
            _ex(instruction="What is X?", output="A2"),
        ]
        result = deduplicate_generated(examples)
        assert len(result) == 1

    def test_filters_against_existing(self):
        examples = [
            _ex(instruction="Already exists", output="A1"),
            _ex(instruction="Brand new question?", output="A2"),
        ]
        result = deduplicate_generated(examples, existing_instructions={"Already exists"})
        assert len(result) == 1
        assert result[0].instruction == "Brand new question?"

    def test_empty_input(self):
        assert deduplicate_generated([]) == []

    def test_no_existing(self):
        examples = [_ex(instruction="Q1", output="A"), _ex(instruction="Q2", output="B")]
        result = deduplicate_generated(examples, existing_instructions=None)
        assert len(result) == 2

    def test_preserves_first_occurrence(self):
        examples = [
            _ex(instruction="Same Q", output="First answer"),
            _ex(instruction="Same Q", output="Second answer"),
        ]
        result = deduplicate_generated(examples)
        assert result[0].output == "First answer"
