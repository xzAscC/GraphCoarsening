import copy
import unittest

from experiments.audit_acceptance_policy import audit_trace
from experiments.audit_frozen_baselines import sigmoid


class AcceptanceAuditTests(unittest.TestCase):
    def fixture(self, policy='binary-monotone'):
        trace = []
        for index, (keep, delete) in enumerate(((-.2, -5.), (.2, -3.))):
            r, d = min(sigmoid(keep), sigmoid(1.)), sigmoid(delete)
            trace.append({'step': index, 'acceptance_policy': policy, 'proposals': 36 if index else 0,
                          'accepted': bool(index), 'retained_logit': keep, 'removed_logit': delete,
                          'capped_retained_probability': r, 'removed_probability': d, 'objective': r-d})
        return {'method': 'Swap-'+policy, 'search_trace': trace, 'full_logit': 1.,
                'retained_logit': .2, 'removed_logit': -3., 'evaluated_proposals': 36}

    def test_policy_specific_continuous_guards(self):
        audit_trace(self.fixture())
        with self.assertRaises(ValueError):
            audit_trace(self.fixture('componentwise'))

    def test_bad_counts_arithmetic_and_final_state_rejected(self):
        for field, value in [('evaluated_proposals', 35), ('retained_logit', .4)]:
            row = self.fixture() | {field: value}
            with self.assertRaises(ValueError):
                audit_trace(row)
        row = copy.deepcopy(self.fixture())
        row['search_trace'][1]['objective'] = .9
        with self.assertRaises(ValueError):
            audit_trace(row)


if __name__ == '__main__':
    unittest.main()
