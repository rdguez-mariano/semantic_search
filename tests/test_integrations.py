from nqs.st_app.utils import generate_answer


def test_generation():
    response = generate_answer("qui a chanté avec Slimane ?")

    assert (
        response.answer
        and response.answer.split() != ""
        and len(response.source_docs) > 0
    )
