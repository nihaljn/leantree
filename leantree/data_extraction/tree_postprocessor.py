import re

import leantree.utils
from leantree.repl_adapter.data import SingletonProofTree, SingletonProofTreeNode, SingletonProofTreeEdge
from leantree.file_span import FileSpan, FilePosition
from leantree.repl_adapter.ast_parser import LeanASTObject, LeanASTArray


class ProofTreePostprocessor:
    @classmethod
    def transform_proof_tree(cls, tree: SingletonProofTree, source_text: str | None = None):
        def visitor(node: SingletonProofTreeNode):
            assert node.tactic is not None, "Transforming an unsolved node."
            if node.tactic.is_synthetic():
                return

            # Note: the order is important here, because tacticStrings are being modified.
            cls._replace_nested_tactics_with_sorries(node)
            cls._decompose_unresolved_nested_by(node)
            cls._remove_by_sorry_in_have(node)
            cls._transform_with_cases(node, source_text)
            cls._transform_case_tactic(node)
            cls._transform_simp_rw(node)
            cls._transform_rw(node)

            node.tactic.tactic_string = leantree.utils.remove_empty_lines(leantree.utils.remove_comments(
                node.tactic.tactic_string
            ))

        cls._merge_intro_arguments(tree)
        cls._add_missing_assumption_tactics(tree)
        cls._fix_incomplete_using_clause(tree, source_text)
        tree.traverse_preorder(visitor)

    # Tactic keywords that match the bare identifier regex but should NOT be merged into `intro`.
    # These are single-word tactics that could appear as the next step after `intro`.
    _TACTIC_KEYWORDS = frozenset({
        "omega", "rfl", "assumption", "trivial", "contradiction", "simp", "ring",
        "norm_num", "linarith", "decide", "exact", "apply", "constructor", "rw",
        "cases", "induction", "intro", "intros", "refl", "ext", "funext", "congr",
        "norm_cast", "push_cast", "aesop", "tauto", "Abel", "group", "positivity",
        "polyrith", "field_simp", "ring_nf", "simp_all", "grind", "gcongr",
        "refine", "have", "let", "obtain", "rcases", "suffices", "show",
        "specialize", "revert", "clear", "rename", "subst", "left", "right",
        "exfalso", "absurd", "push_neg", "by_contra", "by_cases",
        "use", "existsi", "inhabit", "choose", "lift",
    })

    @classmethod
    def _merge_intro_arguments(cls, tree: SingletonProofTree):
        """Merge bare identifier tactic steps that follow `intro` into a single `intro` tactic.

        In newer Lean versions, the REPL splits `intro n h1` into two proof steps:
        one for `intro n` and one for `h1`. This method detects and merges them.
        """
        def visitor(node: SingletonProofTreeNode):
            if node.tactic is None:
                return
            tactic_str = node.tactic.tactic_string.strip()
            if not tactic_str.startswith("intro"):
                return
            # Check if the single child is a bare identifier (continuation of intro)
            children = node.tactic.goals_after
            while (
                len(children) == 1
                and len(node.tactic.spawned_goals) == 0
                and children[0].tactic is not None
                and not children[0].tactic.is_synthetic()
            ):
                child = children[0]
                child_tactic = leantree.utils.remove_empty_lines(
                    leantree.utils.remove_comments(child.tactic.tactic_string)
                ).strip()
                # A bare identifier that is NOT a known tactic keyword
                if (re.match(r'^[a-zA-Z_]\w*$', child_tactic)
                        and child_tactic not in cls._TACTIC_KEYWORDS):
                    # Merge: append the identifier to the intro tactic
                    node.tactic.tactic_string = tactic_str + " " + child_tactic
                    tactic_str = node.tactic.tactic_string.strip()
                    # Extend span to cover the merged child
                    if node.tactic.span is not None and child.tactic.span is not None:
                        node.tactic.span = FileSpan(
                            node.tactic.span.start,
                            child.tactic.span.finish,
                        )
                    # Adopt grandchildren
                    node.tactic.goals_after = child.tactic.goals_after
                    node.tactic.spawned_goals = child.tactic.spawned_goals
                    for grandchild in child.tactic.all_children():
                        grandchild.parent = node
                    children = node.tactic.goals_after
                else:
                    break

        tree.traverse_preorder(visitor)

    @classmethod
    def _add_missing_assumption_tactics(cls, tree: SingletonProofTree):
        if tree.is_solved():
            return

        # e.g. `by` or `suffices` tactics seem to transform the spawned goal to a state where the goal trivially follows
        # from a hypothesis, but then no `assumption` tactic follows. Unfortunately it is not enough to compare the goal
        # type with all hypotheses syntactically - consider "x ≠ 1" and "¬x = 1".
        # This fix seems reckless, but the resulting trees are later verified for correctness.
        def visitor(node: SingletonProofTreeNode):
            if node.tactic is None:
                node.set_edge(SingletonProofTreeEdge.create_synthetic(
                    tactic_string="assumption",
                    goal_before=node.goal,
                    goals_after=[],
                    spawned_goals=[],
                ))

        tree.traverse_preorder(visitor)

    @classmethod
    def _fix_incomplete_using_clause(cls, tree: SingletonProofTree, source_text: str | None = None):
        """Fix tactic strings ending with `using` that are missing their argument.

        In Lean 4.29, the REPL sometimes splits `simpa [...] using <term>` into two steps:
        the `simpa [...] using` part (with a span that ends before the term) and a separate
        child node for the term. This produces an invalid tactic like `simpa [...] using `
        that the REPL rejects. Fix by reading the missing argument from the source text
        and appending it.
        """
        if source_text is None:
            return

        def visitor(node: SingletonProofTreeNode):
            if node.tactic is None or node.tactic.is_synthetic():
                return
            ts = node.tactic.tactic_string.rstrip()
            if not ts.endswith('using'):
                return
            # The tactic ends with `using` but no argument follows.
            # Read the argument from the source text after the span.
            if node.tactic.span is None:
                return
            span_end = node.tactic.span.finish.offset
            # Find the argument: skip whitespace after span, then read the identifier/term
            rest = source_text[span_end:]
            stripped = rest.lstrip(' ')
            if not stripped or stripped[0] == '\n':
                return
            # Read until whitespace or newline
            arg = ''
            for ch in stripped:
                if ch in ' \t\n\r':
                    break
                arg += ch
            if arg:
                node.tactic.tactic_string = ts + ' ' + arg

        tree.traverse_preorder(visitor)

    # https://lean-lang.org/doc/reference/latest//Tactic-Proofs/Tactic-Reference/#cases
    @classmethod
    def _transform_with_cases(cls, node: SingletonProofTreeNode, source_text: str | None = None):
        tactic_str = node.tactic.tactic_string
        cases_match = re.search(r"^(cases\s+[^\n]+)with\s+", tactic_str)
        induction_match = re.search(r"^(induction\s+[^\n]+)with\s+", tactic_str)
        if cases_match:
            constructors = cls._extract_cases_constructors(node, source_text)
            match = cases_match
        elif induction_match:
            constructors = cls._extract_induction_constructors(node, source_text)
            match = induction_match
        else:
            return

        assert len(constructors) == len(node.tactic.spawned_goals),\
            "Different number of constructors and spawned goal for cases/induction."
        intermezzo_nodes = []
        for constructor, child in zip(constructors, node.tactic.spawned_goals):
            # We do not want to synthesize the state before the renaming of constructor variables, so we leave that to
            # Lean during tree verification.
            intermezzo_node = SingletonProofTreeNode.create_synthetic(
                parent=node,
            )
            # The `case` tactic handles renaming of inaccessible hypotheses.
            intermezzo_node.set_edge(SingletonProofTreeEdge.create_synthetic(
                tactic_string=f"case {constructor}",
                goal_before=intermezzo_node.goal,
                spawned_goals=[child],
                goals_after=[],
            ))
            intermezzo_nodes.append(intermezzo_node)
        node.tactic.spawned_goals = intermezzo_nodes
        node.tactic.tactic_string = match.group(1)

        # Alternatively, we could use the explicit `rename_i` tactic in each branch to not depend on Mathlib.
        # https://lean-lang.org/doc/reference/latest/Tactic-Proofs/The-Tactic-Language/#rename_i

        # Another idea would be to use the cases' tactic from Mathlib.
        # https://leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/Cases.html#Mathlib.Tactic.cases'
        # However, cases' doesn't work because not all argument names in the constructors of "cases ... with" need to be specified,
        # so the constructor arguments names and the cases' arguments would be misaligned. We could align them by using "_"
        # in `cases'`, but for that we would need to know the number of arguments for each constructor (which is not visible
        # from the AST)

    # Note that `case` tactics are still present in the tree because they handle variable renaming (not just goal selection).
    @classmethod
    def _transform_case_tactic(cls, node: SingletonProofTreeNode):
        # https://lean-lang.org/doc/reference/latest//Tactic-Proofs/The-Tactic-Language/#case
        tactic_str = node.tactic.tactic_string
        pattern = r"case'?[ \t]+([^\n]+?)[ \t]+=>"
        match = re.match(pattern, tactic_str)
        if not match:
            return

        # Note `case'` doesn't force the goal to be solved immediately, but `case` seems to work as well in the REPL.
        new_tactic = f"case {match.group(1)}"
        node.tactic.tactic_string = new_tactic

    # TODO: e.g. `have` doesn't need the `:= by sorry` - without it, it correctly spawns a goal
    # TODO: expand the spans by any whitespaces at the sides
    @classmethod
    def _replace_nested_tactics_with_sorries(cls, node: SingletonProofTreeNode):
        ancestors = [n for n in node.get_subtree_nodes() if n != node]
        # By blocks are present e.g. in
        # https://lean-lang.org/doc/reference/latest//Tactic-Proofs/The-Tactic-Language/#have
        # and replacing the is in accordance with the examples in the official repo. See e.g.:
        # https://github.com/leanprover-community/repl/blob/master/test/name_generator.in
        # By blocks are also in any number of other places, like `exact sum_congr rfl fun x _ ↦ by ac_rfl`.
        sub_spans = []
        for ancestor in ancestors:
            if ancestor.tactic is None:
                continue
            if not ancestor.tactic.is_synthetic() and node.tactic.span.contains(ancestor.tactic.span):
                sub_spans.append(ancestor.tactic.span.relative_to(node.tactic.span.start))

        # Extend sub_spans to cover trailing content that belongs to the same tactic
        # but isn't covered by the REPL's reported span. This happens when tactic arguments
        # (e.g. `hpd3` in `simpa [...] using hpd3`) are separate tree nodes with synthetic
        # spans that don't contribute to sub_spans.
        if sub_spans:
            sub_spans = cls._extend_spans_for_trailing_content(sub_spans, node.tactic.tactic_string)

        if sub_spans:
            sub_spans = FileSpan.merge_contiguous_spans(
                sub_spans,
                node.tactic.tactic_string,
                lambda inbetween: len(inbetween.strip()) == 0,
            )
            new_tactic = FileSpan.replace_spans(
                base_string=node.tactic.tactic_string,
                replacement="sorry",
                spans=sub_spans,
            )
            node.tactic.tactic_string = new_tactic

    # Pattern to find `by <tactic>` in a tactic string that wasn't decomposed.
    # The REPL normally decomposes nested `by` blocks into separate proof steps,
    # but when the nested tactic has an error Lean doesn't report it.
    _NESTED_BY_RE = re.compile(
        r'\bby\s+'           # `by` keyword followed by whitespace
        r'(?!sorry\b)'       # NOT followed by `sorry` (already decomposed)
        r'(.+)'              # the nested tactic text (greedy — take everything)
    )

    @classmethod
    def _decompose_unresolved_nested_by(cls, node: SingletonProofTreeNode):
        """Detect `by <tactic>` in a tactic string that wasn't decomposed by the
        REPL (typically because the nested tactic errored).  Replace with
        `by sorry` and create a synthetic spawned-goal child so the tree builder
        will replay it separately, producing the correct two-branch structure."""
        tactic_str = node.tactic.tactic_string

        # Only process tactics that still have a non-sorry `by` block
        m = cls._NESTED_BY_RE.search(tactic_str)
        if m is None:
            return

        nested_tactic = m.group(1).strip()
        if not nested_tactic:
            return

        # Check if there are already spawned_goals that cover this —
        # if so, _replace_nested_tactics_with_sorries already handled it.
        if node.tactic.spawned_goals:
            return

        # Replace `by <tactic>` with `by sorry`
        by_start = m.start()
        node.tactic.tactic_string = tactic_str[:by_start] + "by sorry"

        # Create a synthetic child node for the nested tactic.  During replay
        # the tree builder will send `by sorry` and get the sub-goal back, then
        # try the nested tactic on that sub-goal.
        child_edge = SingletonProofTreeEdge.create_synthetic(
            tactic_string=nested_tactic,
            goal_before=None,
            spawned_goals=[],
            goals_after=[],
        )
        child_node = SingletonProofTreeNode.create_synthetic(parent=node, tactic=child_edge)
        node.tactic.spawned_goals.append(child_node)

    # Opening/closing bracket pairs for balanced scanning.
    _OPEN_BRACKETS = set('([{⟨«')
    _CLOSE_BRACKETS = set(')]}⟩»')
    _BRACKET_PAIRS = {'(': ')', '[': ']', '{': '}', '⟨': '⟩', '«': '»'}

    @classmethod
    def _find_tactic_end(cls, text: str, start: int) -> int:
        """Find the end of a tactic starting at ``start``.

        Uses two rules that mirror Lean's whitespace-sensitive parser:

        1. **Same line**: scan forward tracking bracket depth. The tactic
           extends through balanced brackets but stops at an unmatched
           closing bracket (which belongs to the parent expression) or a
           comma at depth 0 (sibling in a tuple/anonymous constructor).
        2. **Continuation lines**: subsequent lines that are indented
           strictly further than the start column are part of the same
           tactic.
        """
        line_start = text.rfind('\n', 0, start) + 1
        start_col = start - line_start

        # --- Phase 1: scan the first line with bracket tracking ---
        first_nl = text.find('\n', start)
        if first_nl == -1:
            first_nl = len(text)

        depth = 0
        end = start
        while end < first_nl:
            ch = text[end]
            if ch in cls._OPEN_BRACKETS:
                depth += 1
            elif ch in cls._CLOSE_BRACKETS:
                if depth > 0:
                    depth -= 1
                else:
                    # Unmatched close bracket — belongs to parent.
                    break
            elif ch == ',' and depth == 0:
                # Comma at top level — sibling argument, not our tactic.
                break
            end += 1

        # If we have open brackets, the tactic must continue on the next
        # line(s) to close them.  Fall through to phase 2 in that case.
        if depth == 0:
            return len(text[start:end].rstrip()) + start

        # We stopped at end-of-first-line with open brackets.
        end = first_nl + 1

        # --- Phase 2: consume indented continuation lines ---
        while end < len(text):
            next_nl = text.find('\n', end)
            if next_nl == -1:
                next_nl = len(text)
            line = text[end:next_nl]
            stripped = line.lstrip(' ')
            if stripped == '' or stripped == '\n':
                end = next_nl + 1
                continue
            line_indent = len(line) - len(stripped)
            if line_indent <= start_col:
                break
            end = next_nl + 1
        return len(text[:end].rstrip())

    @classmethod
    def _extend_spans_for_trailing_content(cls, sub_spans: list[FileSpan], tactic_string: str) -> list[FileSpan]:
        """Extend sub_spans to cover trailing content that the REPL's span missed.

        Two strategies, applied per-span — whichever extends further wins:

        1. **Indentation/bracket lookahead** (``_find_tactic_end``): handles
           multi-line tactics where the REPL span covers only the first line
           (e.g. ``nlinarith [arg1,\\n  arg2]``).
        2. **Gap consumption**: extends to the next span boundary, stopping at
           closing delimiters.  This handles structural gaps like
           ``| refl =>`` between ``induction h with`` and the first case body.
        """
        result = []
        sorted_spans = sorted(sub_spans, key=lambda s: s.start)
        for i, span in enumerate(sorted_spans):
            # Upper boundary: don't extend past the start of the next span.
            if i + 1 < len(sorted_spans):
                boundary = sorted_spans[i + 1].start.offset
            else:
                boundary = len(tactic_string)

            # Strategy 1: indentation/bracket-based tactic end.
            indent_end = min(cls._find_tactic_end(tactic_string, span.start.offset), boundary)

            # Strategy 2: gap consumption — extend to boundary, but only stop
            # at a closing bracket that is unmatched (belongs to the parent).
            gap_end = span.finish.offset
            trailing = tactic_string[span.finish.offset:boundary]
            stripped = trailing.strip()
            if stripped and stripped[0] not in cls._CLOSE_BRACKETS:
                depth = 0
                gap_end = boundary
                for j in range(span.finish.offset, boundary):
                    ch = tactic_string[j]
                    if ch in cls._OPEN_BRACKETS:
                        depth += 1
                    elif ch in cls._CLOSE_BRACKETS:
                        if depth > 0:
                            depth -= 1
                        else:
                            gap_end = j
                            break

            new_end = max(indent_end, gap_end)
            if new_end > span.finish.offset:
                result.append(FileSpan(span.start, FilePosition(new_end)))
            else:
                result.append(span)
        return result

    @classmethod
    def _remove_by_sorry_in_have(cls, node: SingletonProofTreeNode):
        # In newer Lean versions, stripping `:= by sorry` from `have` can cause parse errors
        # (e.g. when custom notation like `n !` is involved). Keep the full tactic string.
        pass

    @classmethod
    def _extract_constructors_from_ast(cls, node: SingletonProofTreeNode, keyword: str) -> list[str] | None:
        """Try to extract constructor names from AST (works for Lean <4.29).

        Returns None if the AST structure doesn't match expectations.
        """
        try:
            ast_node = node.tactic.ast.root
            expected_args = 4 if keyword == "cases" else 5
            assert isinstance(ast_node, LeanASTObject) and len(ast_node.args) == expected_args
            alts_array = ast_node.args[expected_args - 1]
            assert isinstance(alts_array, LeanASTArray) and len(alts_array.items) == 1
            alts_node = alts_array.items[0]
            assert (
                isinstance(alts_node, LeanASTObject) and
                alts_node.type == "Tactic.inductionAlts" and
                len(alts_node.args) == 3
            )
            alts = alts_node.args[2]
            assert isinstance(alts, LeanASTArray)

            constructors = []
            for alt in alts.items:
                assert isinstance(alt, LeanASTObject) and alt.type == "Tactic.inductionAlt"
                constructor_tokens = alt.args[0].get_tokens()
                assert constructor_tokens[0] == "|"
                constructor = " ".join(constructor_tokens[1:])
                constructors.append(constructor)
            return constructors
        except (AssertionError, AttributeError, IndexError):
            return None

    @classmethod
    def _extract_constructors_from_source(cls, node: SingletonProofTreeNode, source_text: str) -> list[str] | None:
        """Extract constructor names and binder names from source file text.

        In Lean 4.29+, the AST for induction/cases no longer contains the inductionAlts.
        Instead, we read the source file and parse `| constructor binders... =>` patterns
        that follow the `with` keyword.
        """
        if node.tactic.span is None:
            return None

        # Read source text from after the tactic span (the `with` keyword) to find alternatives.
        # The tactic span covers just the header (e.g., `cases q with`).
        # The alternatives follow on subsequent lines.
        after_with = source_text[node.tactic.span.finish.offset:]

        # Find all `| constructor_name [binders...] =>` patterns
        # This regex matches: | <words separated by spaces> =>
        alt_pattern = re.compile(r'\|\s*(.+?)\s*=>')
        matches = alt_pattern.findall(after_with)

        if not matches:
            return None

        # Only take as many matches as there are spawned goals
        num_goals = len(node.tactic.spawned_goals)
        if len(matches) < num_goals:
            return None

        # Strip `@` prefix from constructor names — `| @tail j k _ hjk ih =>` uses
        # explicit matching in `induction`/`cases` with `with`, but the `case` tactic
        # doesn't support the `@` prefix.
        result = [m.lstrip('@').lstrip() for m in matches[:num_goals]]
        return result

    @classmethod
    def _extract_constructors_from_goal_tags(cls, node: SingletonProofTreeNode) -> list[str]:
        """Last-resort fallback: extract just constructor names from spawned goal tags.

        This doesn't include binder names, so variable renaming won't happen.
        Only use when source text and AST are both unavailable.
        """
        constructors = []
        for spawned in node.tactic.spawned_goals:
            tag = spawned.goal.tag if spawned.goal else None
            if not tag:
                tag = "anonymous"
            constructors.append(tag)
        return constructors

    # TODO: deduplicate?
    @classmethod
    def _extract_cases_constructors(cls, node: SingletonProofTreeNode, source_text: str | None = None) -> list[str]:
        result = cls._extract_constructors_from_ast(node, "cases")
        if result is not None:
            return result
        if source_text is not None:
            result = cls._extract_constructors_from_source(node, source_text)
            if result is not None:
                return result
        return cls._extract_constructors_from_goal_tags(node)

    @classmethod
    def _extract_induction_constructors(cls, node: SingletonProofTreeNode, source_text: str | None = None) -> list[str]:
        result = cls._extract_constructors_from_ast(node, "induction")
        if result is not None:
            return result
        if source_text is not None:
            result = cls._extract_constructors_from_source(node, source_text)
            if result is not None:
                return result
        return cls._extract_constructors_from_goal_tags(node)

    @classmethod
    def _transform_simp_rw(cls, node: SingletonProofTreeNode):
        match = re.match(r"simp_rw \[([^\n]+)]( at [^\n]+)?", node.tactic.tactic_string)
        if not match:
            return
        assert len(node.tactic.spawned_goals) == 0, "`simp_rw` has spawned goals"

        rules_list = match.group(1)
        at_clause = match.group(2) or ""

        def simp_only(rule: str) -> str:
            return f"simp only [{rule}]{at_clause}"

        rules = [rule.strip() for rule in rules_list.split(",")]
        assert len(rules) > 0, "No rules in a `simp_rw`"
        if len(rules) == 1:
            return

        node.tactic.tactic_string = simp_only(rules[0])
        goals_after = node.tactic.goals_after
        curr_node = node
        for rule in rules[1:]:
            child = SingletonProofTreeNode.create_synthetic(
                parent=curr_node,
            )
            child.set_edge(SingletonProofTreeEdge.create_synthetic(
                tactic_string=simp_only(rule),
                goal_before=child.goal,
                spawned_goals=[],
                goals_after=[],  # Will be filled in.
            ))
            curr_node.tactic.goals_after = [child]
            child.parent = curr_node

            curr_node = child
        curr_node.tactic.goals_after = goals_after
        for g in goals_after:
            g.parent = curr_node

    # @classmethod
    # def _transform_exacts(cls, node: SingletonProofTreeNode):
    #     match = re.match(r"exacts \[([^\n]+)]", node.tactic.tactic_string.strip())
    #     if not match:
    #         return
    #     print(node.goal)
    #     print()
    #     print(f"tactic: {node.tactic.tactic_string}")
    #     print()
    #     for g in node.parent.tactic.spawned_goals:
    #         print(f"parent spawned: {g.goal}")
    #         print()
    #     for g in node.parent.tactic.goals_after:
    #         print(f"parent after: {g.goal}")
    #         print()
    #     print("------")
    #
    #     terms = match.group(1).split(",")
    #     assert len(node.parent.tactic.all_children()) == len(terms),\
    #         "`exacts` has different number of terms then open goals"
    #     term_idx = [
    #         i for i, child in enumerate(node.parent.tactic.all_children())
    #         if child.goal.semantic_equals(node.goal, ignore_metavars=True)
    #     ]
    #     assert len(term_idx) == 1, "Ambiguous or duplicated open goals for `exacts`"
    #
    #     node.tactic.tactic_string = f"exact {terms[term_idx[0]].strip()}"

    @classmethod
    def _transform_rw(cls, node: SingletonProofTreeNode):
        if node.tactic.tactic_string.strip() == "rw [rfl]":
            node.tactic.tactic_string = "rfl"
