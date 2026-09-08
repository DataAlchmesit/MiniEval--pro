"""
Proves two gaps in check_attribution() using the real function — no NLI
model needed, this only tests the attribution pre-check itself.

Run with:
    python test_attribution_gaps.py

Each case prints what the guard actually returned, so you can see the gap
with real output before touching attribution.py.
"""

from minieval_pro.scorers.attribution import check_attribution


CASES = [
    {
        "label": "baseline_my_brother (should already work)",
        "source": "My brother is a lawyer and he helped me with the contract.",
        "fact": "The user is a lawyer.",
        "expect_third_party": True,
        "note": "Confirms the existing 'my <relation>' pattern still fires. "
                "Control case — should already pass today.",
    },
    {
        "label": "gap_1_her_brother",
        "source": "Her brother is a lawyer and he helped with the contract.",
        "fact": "The user is a lawyer.",
        "expect_third_party": True,
        "note": "Same structure as the baseline, just 'her' instead of "
                "'my'. The regex only matches literal 'my', so this is "
                "predicted to wrongly return third_party=False, "
                "confidence=high — meaning the fact goes straight to the "
                "NLI model with no attribution warning at all.",
    },
    {
        "label": "gap_1_their_neighbour",
        "source": "Their neighbour's cat keeps getting into the garden.",
        "fact": "The user owns a cat.",
        "expect_third_party": True,
        "note": "Same shape as the README's documented 'my neighbour's "
                "cat' case, just with 'their' instead of 'my'.",
    },
    {
        "label": "gap_3_plain_third_person_no_reporting_verb",
        "source": "She works as a lawyer downtown.",
        "fact": "The user is a lawyer.",
        "expect_third_party": True,
        "note": "Plain declarative third-person sentence, no 'my', no "
                "reporting verb like 'said' or 'mentioned'. The pronoun "
                "check (THIRD_PERSON_SUBJECTS) is currently only run "
                "inside the reporting-verb branch, so this case never "
                "reaches it at all — predicted to fall through to the "
                "final else and return third_party=False.",
    },
]


def run():
    print("=" * 70)
    print("Attribution guard gap check (pre-fix baseline)")
    print("=" * 70)

    for case in CASES:
        result = check_attribution(case["source"], case["fact"])
        got_third_party = result.third_party
        matches_expectation = got_third_party == case["expect_third_party"]

        status = "OK (caught)" if matches_expectation else "GAP (missed)"
        print(f"\n[{status}] {case['label']}")
        print(f"  source: \"{case['source']}\"")
        print(f"  fact:   \"{case['fact']}\"")
        print(f"  third_party={got_third_party}  confidence={result.confidence}  "
              f"detected={result.detected}")
        print(f"  explanation: {result.explanation}")
        if not matches_expectation:
            print(f"  note: {case['note']}")


if __name__ == "__main__":
    run()