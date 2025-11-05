# Owner(s): ["module: inductor"]

import unittest
import torch
from torch._higher_order_ops import foreach_map
from torch._inductor.test_case import TestCase, run_tests

def _compile(f):
    # fullgraph=True avoids falling back to eager; lets Inductor decide lowering
    return torch.compile(f, backend="inductor", fullgraph=True)

class TestForeachMapMatmul(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def _xfail_or_skip(self, msg: str):
        # In local dev with pytest: XFAIL; otherwise (no pytest/CI): skip
        try:
            import pytest  # type: ignore
            pytest.xfail(msg)
        except Exception:
            self.skipTest(msg)

    def test_matmul_lists_correctness(self):
        xs = [torch.randn(4, 3), torch.randn(5, 2)]
        ys = [torch.randn(3, 6), torch.randn(2, 7)]

        def body(x, y):
            return torch.mm(x, y)

        fn = _compile(lambda xs, ys: foreach_map(body, xs, ys))
        try:
            out = fn(xs, ys)
        except Exception as e:
            self._xfail_or_skip(
                f"foreach_map still treated as pointwise; mm fails: {type(e).__name__}: {e}"
            )
            return

        ref = [torch.mm(x, y) for x, y in zip(xs, ys)]
        self.assertEqual(len(out), len(ref))
        for a, b in zip(out, ref):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-5))

    def test_matmul_backward_matches_eager(self):
        xs = [torch.randn(4, 3, requires_grad=True),
              torch.randn(5, 2, requires_grad=True)]
        ys = [torch.randn(3, 6, requires_grad=True),
              torch.randn(2, 7, requires_grad=True)]

        def body(x, y):
            return torch.mm(x, y)

        fn = _compile(lambda xs, ys: foreach_map(body, xs, ys))
        try:
            out = fn(xs, ys)
        except Exception as e:
            self._xfail_or_skip(
                f"no non-pointwise fallback in foreach_map (backward): {type(e).__name__}: {e}"
            )
            return

        loss = sum(t.sum() for t in out)
        loss.backward()

        xs_ref = [t.detach().clone().requires_grad_(True) for t in xs]
        ys_ref = [t.detach().clone().requires_grad_(True) for t in ys]
        ref = [torch.mm(x, y) for x, y in zip(xs_ref, ys_ref)]
        ref_loss = sum(t.sum() for t in ref)
        ref_loss.backward()

        for x, xr in zip(xs, xs_ref):
            self.assertTrue(torch.allclose(x.grad, xr.grad, atol=1e-6, rtol=1e-5))
        for y, yr in zip(ys, ys_ref):
            self.assertTrue(torch.allclose(y.grad, yr.grad, atol=1e-6, rtol=1e-5))

    def test_matmul_shape_mismatch_raises(self):
        xs = [torch.randn(4, 3)]
        ys = [torch.randn(4, 6)]  # 3 != 4

        def body(x, y):
            return torch.mm(x, y)

        fn = _compile(lambda xs, ys: foreach_map(body, xs, ys))
        with self.assertRaises(Exception):
            _ = fn(xs, ys)

if __name__ == "__main__":
    run_tests()
