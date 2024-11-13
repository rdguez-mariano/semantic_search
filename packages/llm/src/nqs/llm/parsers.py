import re
from typing import Dict, List, Optional, TypeVar, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    BaseOutputParser,
    MarkdownListOutputParser,
)

T = TypeVar("T")


class FirstLineExclusiveValueOutputParser(BaseOutputParser[T]):
    """Custom boolean parser."""

    mapping: Dict[T, List[str]] = {}

    def parse(self, text: str) -> T:
        cleaned_text = [
            s.strip().lower() for s in text.split("\n") if s.strip() != ""
        ]
        targets = cleaned_text[0].split()
        for return_value, written_values in self.mapping.items():
            for wv in written_values:
                for subt in targets:
                    if wv.lower() in subt:
                        return return_value
        raise OutputParserException(
            f"this parser expected output {targets=} to have one of :"
            f"{self.mapping.values()}"
            f"Received {text=} -> {targets=}."
        )

    @property
    def _type(self) -> str:
        return "first_line_exclusive_value_output_parser"


class FirstLineExclusiveBoolOutputParser(
    FirstLineExclusiveValueOutputParser[bool]
):
    mapping: Dict[bool, List[str]] = {
        True: ["yes", "true", "oui", "vrai"],
        False: ["no", "false", "non", "faux"],
    }


class FrenchMarkdownListOutputParser(MarkdownListOutputParser):
    def get_format_instructions(
        self, examples: List[str] = ["foo", "bar", "baz"]
    ) -> str:
        examples_str = "\n- ".join(examples)
        return (
            "Votre réponse doit être une liste markdown, "
            f"ex:\n`- {examples_str}`"
        )


class OutputAndSourceParser(BaseOutputParser):
    """Custom Output and Source parser."""

    trigger_line: Optional[str] = None

    def __init__(self, trigger_line: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_line = trigger_line

    def parse(self, text: str) -> List[Union[str, int]]:
        triggered = False
        docrefs = []
        for lid, line in enumerate(reversed(text.splitlines(keepends=True))):
            if line == self.trigger_line:
                triggered = True
                break
            if re.match(r".+\d+$", line):
                strnum = re.findall(r"\d+$", line)[0]
                docrefs.append(int(strnum))
        if not triggered:
            docrefs = []
            lid = 0
        docrefs = docrefs[::-1]
        output = "".join(text.splitlines(keepends=True)[: -lid - 1])

        return [output] + docrefs

    @property
    def _type(self) -> str:
        return "first_line_exclusive_value_output_parser"
