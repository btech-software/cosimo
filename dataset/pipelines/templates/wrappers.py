"""
Question-type wrappers. Each wraps an existing Calculation template fn(rng, seq)
into another question_type while keeping the same computed answer, distractors
(where applicable), reasoning trace, and preference pair.

Deterministic: they consume no RNG draws beyond those the wrapped fn already
consumes, so the stored answer and trace remain byte-identical to the
recomputation gate in verification/verify_all.py.
"""


def wrap_vignette(fn):
    """Turn a Calculation template into an item-set (vignette) question."""
    def t(rng, seq):
        rich = fn(rng, seq)
        meta = dict(rich["meta"])
        meta["question_type"] = "Vignette"
        q = (f"Scenario (item-set). A {meta['subtopic']} setting.\n"
             f"The following exhibit applies to the question below.\n"
             f"Exhibit: {rich['question']}")
        out = {"meta": meta, "question": q, "answer": rich["answer"],
               "distractors": rich["distractors"],
               "reasoning_trace": rich["reasoning_trace"]}
        if rich.get("flawed"):
            out["flawed"] = rich["flawed"]
        return out
    return t


def wrap_cr(fn):
    """Turn a Calculation template into a constructed-response question."""
    def t(rng, seq):
        rich = fn(rng, seq)
        meta = dict(rich["meta"])
        meta["question_type"] = "Constructed Response"
        q = (f"Constructed response. {rich['question']}\n"
             f"Show your work and state the final value to the nearest whole unit.")
        out = {"meta": meta, "question": q, "answer": rich["answer"],
               "distractors": [],
               "reasoning_trace": rich["reasoning_trace"]}
        if rich.get("flawed"):
            out["flawed"] = rich["flawed"]
        return out
    return t


def wrap_mcq(fn):
    """Turn a Calculation template into a 4-option multiple-choice question.

    The correct answer is stored as "X. <answer>" so verification's nums()
    still extracts the same numeric value. `distractors` keeps the 3 wrong
    numeric values (lettered options are shown in the question), preserving
    axis 4. Options are shuffled deterministically via rng.r (reproducible
    under the recomputation gate's seq=0).
    """
    def t(rng, seq):
        rich = fn(rng, seq)
        meta = dict(rich["meta"])
        meta["question_type"] = "MCQ"
        ans = rich["answer"]
        ds = list(rich.get("distractors") or [])
        options = [ans] + ds[:3]
        letters = ["A", "B", "C", "D"][:len(options)]
        order = list(range(len(options)))
        rng.shuffle(order)                  # deterministic: uses rng.r
        correct_pos = order.index(0)        # option 0 is the correct answer
        lines = [f"{letters[i]}. {options[order[i]]}" for i in range(len(order))]
        q = f"Multiple choice. {rich['question']}\nOptions:\n" + "\n".join(lines)
        out = {"meta": meta, "question": q,
               "answer": f"{letters[correct_pos]}. {ans}",
               "distractors": ds[:3],
               "reasoning_trace": rich["reasoning_trace"]}
        if rich.get("flawed"):
            out["flawed"] = rich["flawed"]
        return out
    return t
