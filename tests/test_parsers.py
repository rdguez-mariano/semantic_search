import numpy as np
import pytest

from nqs.llm.parsers import (
    FirstLineExclusiveBoolOutputParser,
    OutputAndSourceParser,
)


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("yes", True),
        ("oui", True),
        ("no", False),
        ("non", False),
    ],
)
def test_yes_or_no_parser(text, expected_result):
    assert FirstLineExclusiveBoolOutputParser().invoke(text) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("yes", False),
        ("oui", False),
        ("no", True),
        ("non", True),
    ],
)
def test_yes_or_no_parser_consistency(text, expected_result):
    assert FirstLineExclusiveBoolOutputParser().invoke(text) != expected_result


@pytest.mark.parametrize(
    "expected_result, text",
    [
        (
            [0, 1, 2],
            "Les tranches marginales d'imposition sur le revenu sont les suivantes :\n- 0 % pour les revenus jusqu'à 10 025 €\n- 11 % pour les revenus entre 10 026 et 26 070 €\n- 30 % pour les revenus entre 26 071 et 74 545 €\n- 41 % pour les revenus entre 74 546 et 160 336 €\n- 45 % pour les revenus supérieurs à 160 336 €\nDocuments utilisés:\n- Document 0\n- Document 1\n- Document 2",  # noqa
        ),
        (
            [2, 0, 1],
            "Les tranches marginales d'imposition sur le revenu sont les suivantes :\n- 0 % pour les revenus jusqu'à 10 025 €\n- 11 % pour les revenus entre 10 026 et 26 070 €\n- 30 % pour les revenus entre 26 071 et 74 545 €\n- 41 % pour les revenus entre 74 546 et 160 336 €\n- 45 % pour les revenus supérieurs à 160 336 €\nDocuments utilisés:\n- Document 2\n- Document 0\n- Document 1",  # noqa
        ),
        (
            [],
            "Les tranches marginales d'imposition sur le revenu sont les suivantes :\n- 0 % pour les revenus jusqu'à 10 025 €\n- 11 % pour les revenus entre 10 026 et 26 070 €\n- 30 % pour les revenus entre 26 071 et 74 545 €\n- 41 % pour les revenus entre 74 546 et 160 336 €\n- 45 % pour les revenus supérieurs à 160 336 €\nDocuments utilisés:\n",  # noqa
        ),
        (
            [],
            "Les tranches marginales d'imposition sur le revenu sont les suivantes :\n- 0 % pour les revenus jusqu'à 10 025 €\n- 11 % pour les revenus entre 10 026 et 26 070 €\n- 30 % pour les revenus entre 26 071 et 74 545 €\n- 41 % pour les revenus entre 74 546 et 160 336 €\n- 45 % pour les revenus supérieurs à 160 336 €\n",  # noqa
        ),
    ],
)
def test_output_and_source_parser(expected_result, text):
    outputs = OutputAndSourceParser("Documents utilisés:\n").invoke(text)
    refs = outputs[1:]
    print(refs)
    assert len(refs) == len(expected_result) and np.all(
        np.array(refs) == np.array(expected_result)
    )
