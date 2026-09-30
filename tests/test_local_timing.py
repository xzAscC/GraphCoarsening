import unittest

import torch

from experiments.audit_local_gcn import benchmark_pair


class LocalTimingTests(unittest.TestCase):
    def test_alternating_order_and_warmups(self):
        calls = []
        records = benchmark_pair(lambda: calls.append('full'), lambda: calls.append('local'),
                                 'cpu', 4, warmups=1)
        self.assertEqual(calls, ['full', 'local', 'full', 'local', 'local', 'full',
                                 'full', 'local', 'local', 'full'])
        self.assertEqual([r['order'] for r in records],
                         [['full', 'local'], ['local', 'full']] * 2)
        for record in records:
            for name in ('full', 'local'):
                self.assertGreaterEqual(record[name]['wall_seconds'], 0)
                self.assertIsNone(record[name]['peak_extra_allocated_bytes'])

    def test_invalid_repeat_counts(self):
        for count in (0, 1, 3, -2):
            with self.assertRaises(ValueError):
                benchmark_pair(lambda: None, lambda: None, 'cpu', count)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_incremental_peak_allocation(self):
        records = benchmark_pair(lambda: torch.ones(100000, device='cuda'),
                                 lambda: torch.ones(1000, device='cuda'), 'cuda', 2)
        for record in records:
            self.assertGreater(record['full']['peak_extra_allocated_bytes'],
                               record['local']['peak_extra_allocated_bytes'])


if __name__ == '__main__':
    unittest.main()
