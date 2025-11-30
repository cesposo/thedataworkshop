import unittest
import torch

from dist_llm_train.sync.parameter_server import ParameterServer


class TestRobustAggregation(unittest.TestCase):
    def test_trimmed_mean(self):
        ps = ParameterServer(initial_model_state={'w': torch.zeros(3)})

        # Push gradients from 5 workers on param 'w'
        grads = [
            {'w': torch.tensor([1.0, 1.0, 1.0])},
            {'w': torch.tensor([1.0, 1.0, 1.0])},
            {'w': torch.tensor([100.0, 100.0, 100.0])},  # outlier to be trimmed
            {'w': torch.tensor([2.0, 2.0, 2.0])},
            {'w': torch.tensor([2.0, 2.0, 2.0])},
        ]
        for i, g in enumerate(grads):
            ps.push_gradients(f'w{i}', g)

        # Trim 20% => k=0 (since int ratio* n/2); use larger ratio to ensure trimming 1 from both ends
        ok = ps.aggregate_and_update(learning_rate=1.0, rule='trimmed_mean', trim_ratio=0.6)
        self.assertTrue(ok)
        params = ps.get_parameters()['parameters']
        # Mean of trimmed grads (drop one lowest=1 and one highest=100) among 5 -> remaining [1,2,2]
        # avg = 5/3 ≈ 1.666..., model -= lr * avg => -1.666...
        self.assertTrue(torch.allclose(params['w'], torch.tensor([-1.6666666, -1.6666666, -1.6666666]), atol=1e-4))


if __name__ == '__main__':
    unittest.main()

