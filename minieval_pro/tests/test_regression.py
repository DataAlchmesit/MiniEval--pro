"""
Regression suite for gate.check().

Rebuilt against the real gate.py and attribution.py (previous version was
written before those files were available and used guessed field names).

Field names match the real GateDecision: verdict is "STORE", "REVIEW", or
"REJECT". Each case's expected value is labeled either CONFIRMED (we've
seen this exact or equivalent pair scored in a real run already) or
PREDICTED (a hypothesis — run it and check, don't assume it's right).

Run with:
    python test_regression.py
"""

from dataclasses import dataclass
from minieval_pro.gate import MemoryGate


@dataclass
class Case:
    label: str
    source: str
    fact: str
    expected: str            # "STORE", "REVIEW", or "REJECT"
    basis: str                # "CONFIRMED" or "PREDICTED"
    note: str = ""


CASES = [

    # --- Faithful, direct — should STORE ---
    Case(
        label="entailed_peanut_allergy",
        source="I am allergic to peanuts.",
        fact="The user is allergic to peanuts.",
        expected="STORE",
        basis="CONFIRMED",
        note="Scored 0.99 faithful when used as an existing-fact pair in "
             "the adjudicate() suite. Same pair here, run through check() "
             "directly.",
    ),

    # --- Implied — should STORE, does today ---
    Case(
        label="implied_moved_to_chennai",
        source="I moved to Chennai last week.",
        fact="The user lives in Chennai.",
        expected="STORE",
        basis="CONFIRMED",
        note="Scored 0.92 faithful as the incoming pair in adjudicate() "
             "case 2. Confirms simple 'moved to X' framing works — unlike "
             "the Delhi/Bangalore case below, which adds a 'moved FROM' "
             "clause.",
    ),

    # --- Known documented failure — kept in deliberately ---
    Case(
        label="implied_delhi_to_bangalore_KNOWN_FAILURE",
        source="I moved from Delhi to Bangalore last month.",
        fact="The user lives in Bangalore.",
        expected="STORE",
        basis="CONFIRMED",
        note="README documents this scoring 0.774 contradiction instead "
             "of supported — a real, open, undocumented-fix limitation. "
             "Expected to FAIL (actual will be REJECT). Kept in the suite "
             "on purpose so this regression is never silently lost.",
    ),

    # --- Contradiction — should REJECT ---
    Case(
        label="contradicts_peanut_hallucination",
        source="I had a great salad for lunch.",
        fact="The user loves eating peanuts.",
        expected="REJECT",
        basis="CONFIRMED",
        note="This exact pair scored 0.00 (contradicts) as the incoming "
             "fact in adjudicate() case 3, correctly triggering BLOCK "
             "there. Predicted to REJECT here via check() for the same "
             "reason.",
    ),

    # --- Unsupported/unrelated — should REVIEW, not REJECT ---
    Case(
        label="unsupported_unrelated_topic",
        source="I had a salad for lunch today.",
        fact="The user is a doctor.",
        expected="REVIEW",
        basis="PREDICTED",
        note="Tests the relatedness guard: an unrelated pair the NLI "
             "model might force into 'contradicts' should be caught by "
             "min_relatedness and downgraded to REVIEW rather than "
             "REJECTed outright. Not yet run for real — confirm.",
    ),

    # --- Attribution — now fixed, all four confirmed via test_attribution_gaps.py ---
    Case(
        label="attribution_my_brother",
        source="My brother is a lawyer and he helped me with the contract.",
        fact="The user is a lawyer.",
        expected="REVIEW",
        basis="CONFIRMED",
        note="check_attribution() confirmed third_party=True, "
             "confidence=high via test_attribution_gaps.py. This runs it "
             "through the full check() pipeline to confirm the REVIEW "
             "verdict actually surfaces end to end.",
    ),
    Case(
        label="attribution_her_brother_FIXED_GAP",
        source="Her brother is a lawyer and he helped with the contract.",
        fact="The user is a lawyer.",
        expected="REVIEW",
        basis="CONFIRMED",
        note="Before the attribution.py fix, this returned "
             "third_party=False, confidence=high — a silent false STORE "
             "risk. Now confirmed caught at the attribution-check level; "
             "this case confirms it also surfaces correctly through the "
             "full gate.",
    ),
    Case(
        label="attribution_their_neighbour_FIXED_GAP",
        source="Their neighbour's cat keeps getting into the garden.",
        fact="The user owns a cat.",
        expected="REVIEW",
        basis="CONFIRMED",
        note="Same fix as above, confirmed at the attribution-check "
             "level. Confirming here through the full check() pipeline.",
    ),
    Case(
        label="attribution_she_works_as_lawyer_FIXED_GAP",
        source="She works as a lawyer downtown.",
        fact="The user is a lawyer.",
        expected="REVIEW",
        basis="CONFIRMED",
        note="The gap_3 fix — plain third-person sentence, no 'my', no "
             "reporting verb. Confirmed caught at the attribution-check "
             "level; confirming here end to end.",
    ),

    # --- Watch case: possible false positive from the widened pronoun check ---
    Case(
        label="watch_pronoun_not_about_subject",
        source="I told her about the new apartment I found.",
        fact="The user found a new apartment.",
        expected="STORE",
        basis="PREDICTED",
        note="'Her' here is who the user was talking TO, not who the "
             "fact is about. This is the exact risk flagged when the "
             "attribution fix widened the pronoun check: it may now "
             "wrongly flag this as third-party attribution and return "
             "REVIEW instead of STORE. If this fails, it's a real, newly "
             "introduced false positive — worth knowing, not assuming "
             "away.",
    ),

]


def run_suite():
    gate = MemoryGate(quiet=True)
    results = []

    for case in CASES:
        decision = gate.check(source=case.source, fact=case.fact)
        actual = decision.verdict
        passed = (actual == case.expected)
        results.append((case, actual, passed, decision))

    return results


def print_report(results):
    total = len(results)
    passed = sum(1 for _, _, ok, _ in results if ok)
    confirmed_failures = [
        (c, a) for c, a, ok, _ in results if not ok and c.basis == "CONFIRMED"
    ]
    predicted_checks = [
        (c, a, ok) for c, a, ok, _ in results if c.basis == "PREDICTED"
    ]

    print("=" * 70)
    print("MiniEval check() regression suite")
    print("=" * 70)

    for case, actual, ok, decision in results:
        mark = "PASS" if ok else "FAIL"
        tag = f"[{case.basis}]"
        print(f"\n[{mark}] {tag} {case.label}")
        print(f"  source: \"{case.source}\"")
        print(f"  fact:   \"{case.fact}\"")
        print(f"  expected={case.expected}  actual={actual}  "
              f"faithfulness={decision.faithfulness:.2f} ({decision.label})")
        if not ok:
            print(f"  note: {case.note}")

    print("\n" + "-" * 70)
    print(f"Total: {total}   Passed: {passed}   Failed: {total - passed}")
    print("-" * 70)

    if confirmed_failures:
        print("\nCONFIRMED known failures (expected to fail, tracked on purpose):")
        for c, a in confirmed_failures:
            print(f"  - {c.label}: expected {c.expected}, got {a}")

    print("\nPREDICTED cases — read these carefully, they were hypotheses:")
    for c, a, ok in predicted_checks:
        status = "held up" if ok else "DID NOT hold up — new finding"
        print(f"  - {c.label}: {status} (expected {c.expected}, got {a})")


if __name__ == "__main__":
    results = run_suite()
    print_report(results)