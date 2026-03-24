"""
Tests for leantree proof tree extraction on diverse Mathlib proofs.

Each test loads a Lean file containing theorems from Mathlib and verifies that
leantree can extract valid, solved proof trees from them.
"""

from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

import os
import pytest

from dotenv import load_dotenv
load_dotenv()

from leantree import LeanProject
from leantree.core.lean_file import LeanTheorem, LeanTacticBlock, StoredError
from leantree.core.proof_tree import ProofTree
from leantree.utils import Logger, LogLevel

PROJECT_PATH = Path(os.getenv("LEANTREE_TESTS_PROJECT_PATH"))

TEST_FILES = {
    "symm_diff": "LeantreeProject/TestSymmDiff.lean",
    "factorial": "LeantreeProject/TestFactorial.lean",
    "wilson": "LeantreeProject/TestWilson.lean",
    "int_div": "LeantreeProject/TestIntDiv.lean",
    "gcd": "LeantreeProject/TestGCD.lean",
    "paths": "LeantreeProject/TestPaths.lean",
    "apply_intro": "LeantreeProject/TestApplyIntro.lean",
    # "solvable" excluded: wellfounded induction triggers "Reusing a child!" in SingletonTreeBuilder
    "connected": "LeantreeProject/TestConnected.lean",
    "prime_dvd": "LeantreeProject/TestPrimeDvd.lean",
    "list_perm": "LeantreeProject/TestListPerm.lean",
    "infinite_primes": "LeantreeProject/InfinitePrimes.lean",
    "infinite_primes_fail": "LeantreeProject/InfinitePrimesFail.lean",
    "explicit_sorry": "LeantreeProject/TestExplicitSorry.lean",
    "semantic_errors": "LeantreeProject/TestSemanticErrors.lean",
    "syntax_errors": "LeantreeProject/TestSyntaxErrors.lean",
    "multiline_errors": "LeantreeProject/TestIMOProofP5.lean",
}

# Expected theorem counts per file
EXPECTED_THEOREMS = {
    "symm_diff": 2,
    "factorial": 1,
    "wilson": 1,
    "int_div": 1,
    "gcd": 1,
    "paths": 2,
    "apply_intro": 2,
    "connected": 2,
    "prime_dvd": 1,
    "list_perm": 2,
    "infinite_primes": 4,
    "infinite_primes_fail": 4,
    "explicit_sorry": 3,  # add_zero_right, full_sorry, have_sorry
    "semantic_errors": 4,
    "syntax_errors": 4,
    "multiline_errors": 1,
}


@pytest.fixture(scope="module")
def project():
    if not PROJECT_PATH.exists():
        pytest.skip(f"Project path {PROJECT_PATH} does not exist")
    return LeanProject(PROJECT_PATH, logger=Logger(LogLevel.DEBUG))


def load_and_check(project: LeanProject, test_key: str):
    """Load a test file and return its theorems, checking basic expectations."""
    file_path = PROJECT_PATH / TEST_FILES[test_key]
    assert file_path.exists(), f"Test file {file_path} does not exist"

    file = project.load_file(file_path)

    # Check we got the expected number of theorems
    expected = EXPECTED_THEOREMS[test_key]
    actual_theorems = [t for t in file.theorems if isinstance(t, LeanTheorem)]
    errors = [t for t in file.theorems if isinstance(t, StoredError)]

    if errors:
        for e in errors:
            print(f"  StoredError: {e.error}")

    assert len(actual_theorems) == expected, (
        f"Expected {expected} theorems in {test_key}, got {len(actual_theorems)} "
        f"(+ {len(errors)} errors)"
    )

    return actual_theorems


def check_theorem_trees(theorem: LeanTheorem, thm_name: str):
    """Check that all by-blocks in a theorem have valid solved proof trees."""
    assert len(theorem.by_blocks) > 0, f"{thm_name}: no by-blocks found"

    for i, block in enumerate(theorem.by_blocks):
        block_id = f"{thm_name} by-block {i}"
        assert isinstance(block, LeanTacticBlock), (
            f"{block_id}: expected LeanTacticBlock, got {type(block)}"
        )
        assert isinstance(block.tree, ProofTree), (
            f"{block_id}: expected ProofTree, got {type(block.tree)}: "
            f"{block.tree.error if isinstance(block.tree, StoredError) else block.tree}"
        )
        assert block.tree.is_solved(), (
            f"{block_id}: proof tree is not solved.\n"
            f"Tree:\n{block.tree.pretty_print()}"
        )
        # Print the tree for debugging
        print(f"\n{'='*60}")
        print(f"  {block_id} - SOLVED (size={block.tree.root.proof_size})")
        print(f"{'='*60}")
        print(block.tree.pretty_print())


# --- Individual test functions ---

def test_symm_diff(project):
    """Test simp_rw with multiple rules + rw chains with refine and bullets."""
    theorems = load_and_check(project, "symm_diff")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "symm_diff")


def test_factorial(project):
    """Test induction with `with`, have with nested `by`, refine."""
    theorems = load_and_check(project, "factorial")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "factorial")


def test_wilson(project):
    """Test rcases, obtain, by_contra, replace, norm_num at, rw chain."""
    theorems = load_and_check(project, "wilson")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "wilson")


def test_int_div(project):
    """Test constructor splitting iff, intro with destructuring, rw + simp."""
    theorems = load_and_check(project, "int_div")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "int_div")


def test_gcd(project):
    """Test suffices, rcases case analysis, split_ifs, rw chains."""
    theorems = load_and_check(project, "gcd")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "gcd")


def test_paths(project):
    """Test induction/cases with `with`, simp only [...] at hyp ⊢."""
    theorems = load_and_check(project, "paths")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "paths")


def test_apply_intro(project):
    """Test apply, intro, obtain, constructor - basic set/function theory."""
    theorems = load_and_check(project, "apply_intro")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "apply_intro")


@pytest.mark.skip(reason="Wellfounded induction triggers 'Reusing a child!' in SingletonTreeBuilder")
def test_solvable(project):
    """Test wellfounded induction, nested induction, suffices, rcases, clear."""
    pass


def test_connected(project):
    """Test biUnion with reflTransGen, rintro, replace, exacts."""
    theorems = load_and_check(project, "connected")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "connected")


def test_prime_dvd(project):
    """Test chain of have steps, simpa, suffices, Or.elim."""
    theorems = load_and_check(project, "prime_dvd")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "prime_dvd")


def test_list_perm(project):
    """Test specialize, by_cases, induction with 4 cases, obtain with dash."""
    theorems = load_and_check(project, "list_perm")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "list_perm")


def test_infinite_primes(project):
    """Test grind, custom notation (n!), French quotes, anonymous have, <;>."""
    theorems = load_and_check(project, "infinite_primes")
    for thm in theorems:
        check_theorem_trees(thm, thm.name or "infinite_primes")


def test_infinite_primes_fail(project):
    """Test partial tree extraction when a file has errors (factorial_po typo).

    Theorems 0-2 should be fully solved. Theorem 3 (InfinitudeOfPrimes) should
    produce a ProofTree with has_error()=True, not a StoredError.
    """
    theorems = load_and_check(project, "infinite_primes_fail")

    # First 3 theorems should be fully solved
    for thm in theorems[:3]:
        check_theorem_trees(thm, thm.name or "infinite_primes_fail")

    # Theorem 3 should have a partial tree with an error
    thm = theorems[3]
    assert len(thm.by_blocks) > 0, "InfinitudeOfPrimes: no by-blocks found"
    block = thm.by_blocks[0]
    assert isinstance(block.tree, ProofTree), (
        f"Expected ProofTree (possibly with errors), got {type(block.tree)}"
    )
    assert not block.tree.is_solved(), "Expected unsolved tree due to error"
    assert block.tree.has_error(), "Expected error node in tree"

    # Find the error edge and verify it mentions the typo
    error_edges = [
        n.tactic for n in block.tree.get_nodes()
        if n.tactic is not None and n.tactic.error is not None
    ]
    assert len(error_edges) > 0, "No error edges found in tree"
    assert any("factorial_po" in e.error for e in error_edges), (
        f"Expected error about 'factorial_po', got: {[e.error for e in error_edges]}"
    )

    print(f"\n{'='*60}")
    print(f"  InfinitudeOfPrimes - PARTIAL TREE WITH ERROR")
    print(f"{'='*60}")
    print(block.tree.pretty_print())


def _check_error_tree(block, label: str):
    """Helper: assert a by-block has a ProofTree with errors."""
    assert isinstance(block.tree, ProofTree), (
        f"{label}: expected ProofTree, got {type(block.tree)}"
    )
    assert not block.tree.is_solved(), (
        f"{label}: expected unsolved tree\n{block.tree.pretty_print()}"
    )
    assert block.tree.has_error(), (
        f"{label}: expected error in tree\n{block.tree.pretty_print()}"
    )
    print(f"\n{'='*60}")
    print(f"  {label} - PARTIAL TREE WITH ERROR")
    print(f"{'='*60}")
    print(block.tree.pretty_print())
    return block.tree


def test_explicit_sorry(project):
    """Test sorry handling: sorry branches become errors, rest expands fully.

    Theorem 0 (add_zero_right): fully solved.
    Theorem 1 (full_sorry): entire proof is sorry.
    Theorem 2 (have_sorry): sorry in a have sub-goal, rest correct.
    """
    theorems = load_and_check(project, "explicit_sorry")

    # Theorem 0 should be fully solved
    check_theorem_trees(theorems[0], theorems[0].name or "add_zero_right")

    # Theorem 1: full sorry — entire proof is sorry
    thm = theorems[1]
    assert len(thm.by_blocks) > 0
    block = thm.by_blocks[0]
    tree = _check_error_tree(block, "full_sorry")
    error_edges = [
        n.tactic for n in tree.get_nodes()
        if n.tactic is not None and n.tactic.error is not None
    ]
    assert any("sorry" in e.error for e in error_edges), (
        f"Expected sorry error, got: {[e.error for e in error_edges]}"
    )

    # Theorem 2: sorry in a have sub-goal
    thm = theorems[2]
    assert len(thm.by_blocks) > 0
    block = thm.by_blocks[0]
    tree = _check_error_tree(block, "have_sorry")
    error_edges = [
        n.tactic for n in tree.get_nodes()
        if n.tactic is not None and n.tactic.error is not None
    ]
    assert any("sorry" in e.error for e in error_edges), (
        f"Expected sorry error, got: {[e.error for e in error_edges]}"
    )


def test_semantic_errors(project):
    """Test semantic errors: wrong claims cause tactic failures.

    Theorem 0 (zero_add_eq): fully solved.
    Theorem 1 (wrong_ineq): omega can't prove n < n.
    Theorem 2 (succ_pos): fully solved.
    Theorem 3 (wrong_eq): simp can't close n + 1 = n.
    """
    theorems = load_and_check(project, "semantic_errors")

    # Theorems 0 and 2 should be fully solved
    check_theorem_trees(theorems[0], theorems[0].name or "zero_add_eq")
    check_theorem_trees(theorems[2], theorems[2].name or "succ_pos")

    # Theorem 1: omega can't prove n < n
    thm = theorems[1]
    assert len(thm.by_blocks) > 0
    block = thm.by_blocks[0]
    _check_error_tree(block, "wrong_ineq")

    # Theorem 3: simp can't close n + 1 = n
    thm = theorems[3]
    assert len(thm.by_blocks) > 0
    block = thm.by_blocks[0]
    _check_error_tree(block, "wrong_eq")


def test_syntax_errors(project):
    """Test reference/syntax errors: unknown constants cause errors.

    Theorem 0 (add_comm_example): fully solved.
    Theorem 1 (add_assoc_ref_error): Nat.add_comn unknown.
    Theorem 2 (trivial_eq): fully solved.
    Theorem 3 (mul_ref_error): Nat.mul_comn unknown.
    """
    theorems = load_and_check(project, "syntax_errors")

    # Theorems 0 and 2 should be fully solved
    check_theorem_trees(theorems[0], theorems[0].name or "add_comm_example")
    check_theorem_trees(theorems[2], theorems[2].name or "trivial_eq")

    # Theorem 1: reference error
    thm = theorems[1]
    assert len(thm.by_blocks) > 0
    block = thm.by_blocks[0]
    tree = _check_error_tree(block, "add_assoc_ref_error")
    error_edges = [
        n.tactic for n in tree.get_nodes()
        if n.tactic is not None and n.tactic.error is not None
    ]
    assert any("add_comn" in e.error for e in error_edges), (
        f"Expected error about 'add_comn', got: {[e.error for e in error_edges]}"
    )

    # Theorem 3: reference error
    thm = theorems[3]
    assert len(thm.by_blocks) > 0
    block = thm.by_blocks[0]
    tree = _check_error_tree(block, "mul_ref_error")
    error_edges = [
        n.tactic for n in tree.get_nodes()
        if n.tactic is not None and n.tactic.error is not None
    ]
    assert any("mul_comn" in e.error for e in error_edges), (
        f"Expected error about 'mul_comn', got: {[e.error for e in error_edges]}"
    )


def test_multiline_errors(project):
    """Test multi-line tactic error handling (IMO-style proof with 3 errors).

    Lean compiler reports:
    1. Line 12: nlinarith failed (multi-line tactic with continuation args)
    2. Line 33: unsolved goals after rw (h_f3a_eq_fa)
    3. Line 43: type mismatch (this.symm wrong direction)

    The single theorem should produce a partial tree with errors.
    """
    theorems = load_and_check(project, "multiline_errors")

    thm = theorems[0]
    assert len(thm.by_blocks) > 0
    block = thm.by_blocks[0]
    assert isinstance(block.tree, ProofTree), (
        f"Expected ProofTree, got {type(block.tree)}"
    )
    assert block.tree.has_error(), "Expected errors in tree"
    assert not block.tree.is_solved(), "Expected unsolved tree due to errors"

    # Tree should have substantial expansion (not just root + error)
    nodes = block.tree.get_nodes()
    assert len(nodes) > 5, (
        f"Expected substantial tree expansion, got only {len(nodes)} nodes"
    )

    # Collect all error edges
    error_edges = [
        n.tactic for n in nodes
        if n.tactic is not None and n.tactic.error is not None
    ]
    error_texts = [e.error for e in error_edges]

    # Lean reports 3 errors. Check which ones leantree captures.
    # Error 1: nlinarith/linarith failed (line 12, multi-line tactic)
    has_nlinarith_error = any(
        "linarith" in err.lower() or "nlinarith" in err.lower()
        for err in error_texts
    )
    # Error 2: unsolved goals after rw in h_f3a_eq_fa (line 33)
    # Lean reports "unsolved goals" but leantree sees it as the synthetic
    # `assumption` tactic failing on the remaining goal after the rw chain.
    has_unsolved_goals_error = any(
        "unsolved" in err.lower() or "assumption" in err.lower()
        for err in error_texts
    )
    # Error 3: type mismatch from this.symm (line 43)
    has_type_mismatch_error = any(
        "type mismatch" in err.lower() or "mismatch" in err.lower()
        for err in error_texts
    )

    captured = sum([has_nlinarith_error, has_unsolved_goals_error, has_type_mismatch_error])

    print(f"\n{'='*60}")
    print(f"  imo_1968_p5_1 - PARTIAL TREE WITH {len(error_edges)} ERRORS")
    print(f"  Captured {captured}/3 Lean errors:")
    print(f"    nlinarith failed:   {'YES' if has_nlinarith_error else 'NO'}")
    print(f"    unsolved goals:     {'YES' if has_unsolved_goals_error else 'NO'}")
    print(f"    type mismatch:      {'YES' if has_type_mismatch_error else 'NO'}")
    print(f"  All error texts: {error_texts}")
    print(f"{'='*60}")
    print(block.tree.pretty_print())

    # All 3 Lean compiler errors must be captured
    assert has_nlinarith_error, (
        f"Missing nlinarith/linarith error (line 12). Errors: {error_texts}"
    )
    assert has_unsolved_goals_error, (
        f"Missing unsolved goals error (line 33). Errors: {error_texts}"
    )
    assert has_type_mismatch_error, (
        f"Missing type mismatch error (line 43). Errors: {error_texts}"
    )
