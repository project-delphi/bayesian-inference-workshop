import hashlib

from workshop.m02_diagnostic import quiz_answers

HASHES = {
    "q1": "de5bd8bd9e3b0d55",
    "q2": "751c8ea895fd8bcd",
    "q3": "6bfe40d993db6cd7",
    "q4": "c884cc22968b23e4",
    "q5": "1866a4112f73f8fd",
    "q6": "20be3f8962f206ba",
    "q7": "5d04fda517a486f6",
    "q8": "ef3a31d9b0ab9ce5",
}


def _h(q, a):
    return hashlib.sha256(f"{q}:{a}".encode()).hexdigest()[:16]


def test_step5_quiz():
    answers = quiz_answers()
    wrong = [q for q, h in HASHES.items() if _h(q, str(answers.get(q, "")).strip().lower()) != h]
    assert not wrong, f"Incorrect or missing answers: {wrong}. See the review list on the Module 2 page."
