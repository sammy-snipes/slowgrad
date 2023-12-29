import unittest
import torch
from slowgrad.engine import SlowgradVar


class TestAddition(unittest.TestCase):
    def test_scalar_addition(self):
        self.assertTrue(
            torch.allclose((SlowgradVar(-1) + SlowgradVar(1)).data, torch.tensor(0))
        )
        self.assertTrue(
            torch.allclose((SlowgradVar(2) + SlowgradVar(3)).data, torch.tensor(5))
        )
        self.assertTrue(
            torch.allclose((SlowgradVar(-5) + SlowgradVar(-6)).data, torch.tensor(-11))
        )

    def test_vector_addition(self):
        self.assertTrue(
            torch.allclose(
                (SlowgradVar([1, 2]) + SlowgradVar([3, 4])).data, torch.tensor([4, 6])
            )
        )
        self.assertTrue(
            torch.allclose(
                (SlowgradVar([[-1], [1]]) + SlowgradVar([[1], [-1]])).data,
                torch.tensor([[0], [0]]),
            )
        )

    def test_matrix_addition(self):
        self.assertTrue(
            torch.allclose(
                (
                    SlowgradVar([[1, 2, 3], [4, 5, 6]])
                    + SlowgradVar([[3, 2, 1], [6, 5, 4]])
                ).data,
                torch.tensor([[4, 4, 4], [10, 10, 10]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                (SlowgradVar([[[[[5]]]]]) + SlowgradVar([[[[[6]]]]])).data,
                torch.tensor([[[[[11]]]]]),
            )
        )

    def test_broadcast_addition(self):
        self.assertTrue(
            torch.allclose(
                (SlowgradVar(5) + SlowgradVar([[1, 2], [3, 4]])).data,
                torch.tensor([[6, 7], [8, 9]]),
            )
        )

        self.assertTrue(
            torch.allclose(
                (SlowgradVar([[-1, 1], [1, -1]]) + SlowgradVar(10)).data,
                torch.tensor([[9, 11], [11, 9]]),
            )
        )

        self.assertTrue(
            torch.allclose(
                (SlowgradVar([[1, 2], [3, 4]]) + SlowgradVar([5, -5])).data,
                torch.tensor([[6, -3], [8, -1]]),
            )
        )

        self.assertTrue(
            torch.allclose(
                (SlowgradVar([[1, 2], [3, 4]]) + SlowgradVar([[5, -5]])).data,
                torch.tensor([[6, -3], [8, -1]]),
            )
        )

    def test_self_sum(self):
        self.assertEqual(SlowgradVar([1, 2, 3]).sum().data, 6)
        self.assertEqual(SlowgradVar([[-1, -2, -3], [1, 2, 3]]).sum().data, 0)
        self.assertEqual(
            SlowgradVar([[[1, 2], [3, 4]], [[-2, -7], [-3, 0]]]).sum().data, -2
        )


if __name__ == "__main__":
    unittest.main()
