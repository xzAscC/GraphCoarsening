import unittest

import torch

from src.explainers.support_refinement import acceptance_mask


class AcceptancePolicyTests(unittest.TestCase):
    def test_binary_repair_can_trade_already_successful_deleted_confidence(self):
        # Example reflects a positive original prediction: retained confidence
        # crosses 0.5 while deleted confidence rises but remains far below 0.5.
        current = tuple(torch.tensor(x) for x in (.4583, .4590, .0007, False, True))
        proposed = tuple(torch.tensor([x]) for x in (.5122, .5216, .0094, True, True))
        self.assertFalse(acceptance_mask(proposed, current, 1e-7).item())
        self.assertTrue(acceptance_mask(proposed, current, 1e-7, 'binary-monotone').item())

    def test_objective_and_binary_guards_remain_required(self):
        current = tuple(torch.tensor(x) for x in (.2, .7, .5, True, False))
        values = tuple(torch.tensor(x) for x in ([.3, .3, .2, .3], [.4, .8, .7, .8],
                                                [.1, .5, .5, .5], [False, True, True, True],
                                                [True, True, False, False]))
        self.assertEqual(acceptance_mask(values, current, 1e-7, 'binary-monotone').tolist(), [False, True, False, True])
        for i in range(4):
            scalar = tuple(x[i] for x in values)
            self.assertEqual(acceptance_mask(scalar, current, 1e-7, 'binary-monotone').item(),
                             acceptance_mask(values, current, 1e-7, 'binary-monotone')[i].item())
        with self.assertRaises(ValueError):
            acceptance_mask(values, current, 1e-7, 'unknown')


if __name__ == '__main__':
    unittest.main()
