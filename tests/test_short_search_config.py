from experiments.benchmark_acceptance_policy import policy_steps


def test_fixed_short_search_preserves_default():
    assert policy_steps(False) == {'componentwise': 8, 'binary-monotone': 8}
    assert policy_steps(True) == {'componentwise': 8, 'binary-monotone': 4}
