import numpy as np
import pytest

from modelrouter.data import routerbench
from modelrouter.data.splits import leave_family_out, random_split


def test_parse_turns_and_router_text():
    turns = routerbench.parse_turns("['TASK: solve these. ' , 'How many apples?']")
    assert turns == ["TASK: solve these. ", "How many apples?"]
    assert routerbench.parse_turns("not a list") == ["not a list"]
    text = routerbench.router_text(["x" * 1000, "the real question"])
    assert text.endswith("the real question")
    assert len(text) == routerbench.ROUTER_TEXT_HEAD_CHARS + 2 + len("the real question")


def test_family_of():
    assert routerbench.family_of("mmlu-college-physics") == "mmlu"
    assert routerbench.family_of("grade-school-math") == "gsm8k"
    assert routerbench.family_of("mtbench-math") == "mtbench"
    assert routerbench.family_of("chinese_idioms") == "misc"


def test_random_split_is_stratified_and_disjoint(synthetic):
    split = random_split(synthetic, seed=1)
    n = len(synthetic)
    assert len(split.train) + len(split.val) + len(split.test) == n
    for fam in ("alpha", "beta"):
        share_all = np.mean(synthetic.family == fam)
        share_test = np.mean(synthetic.family[split.test] == fam)
        assert abs(share_all - share_test) < 0.02
    assert random_split(synthetic, seed=1).test.tolist() == split.test.tolist()


def test_leave_family_out_hides_the_family(synthetic):
    split = leave_family_out(synthetic, "alpha")
    assert set(synthetic.family[split.test]) == {"alpha"}
    assert "alpha" not in set(synthetic.family[split.train]) | set(synthetic.family[split.val])
    with pytest.raises(ValueError, match="no rows"):
        leave_family_out(synthetic, "gamma")


def test_subset_and_select_models(synthetic):
    sub = synthetic.subset([0, 5, 7]).select_models(["large", "small"])
    assert len(sub) == 3
    assert sub.models == ("large", "small")
    np.testing.assert_array_equal(sub.quality[:, 0], synthetic.quality[[0, 5, 7], 2])


@pytest.mark.data
def test_prepared_routerbench_shape(routerbench):
    assert len(routerbench) == 36478
    assert len(routerbench.models) == 11
    assert np.all((routerbench.quality >= 0) & (routerbench.quality <= 1))
    assert np.all(routerbench.cost > 0)
    gsm = routerbench.family == "gsm8k"
    # Re-graded GSM8K is exact match, so labels are binary.
    assert set(np.unique(routerbench.quality[gsm])) <= {0.0, 1.0}
