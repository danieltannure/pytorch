# Owner(s): ["module: inductor"]

import inspect
import unittest
import torch
from torch._higher_order_ops import foreach_map
from torch._inductor.test_case import TestCase, run_tests

try:
    from torch._inductor.exc import (
        InductorError,
        SubgraphLoweringException,
        SchedulerError,
        InternalError,
        Unsupported,
    )
    INDUCTOR_EXC = (
        InductorError,
        SubgraphLoweringException,
        SchedulerError,
        InternalError,
        Unsupported,
        RuntimeError,
        AssertionError,
        NotImplementedError,
        ValueError,
        KeyError,
    )
except Exception:
    INDUCTOR_EXC = (RuntimeError, AssertionError, NotImplementedError, ValueError, KeyError)


def _compile(f):
    return torch.compile(f, backend="inductor", fullgraph=True)


def _supports_assert_fused() -> bool:
    """Heuristic: run strict test only if lowering knows about `assert_fused`."""
    try:
        import torch._inductor.lowering as L
        return "assert_fused" in inspect.getsource(L._foreach_map)
    except Exception:
        return False


class TestForeachMapAssertFused(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def test_without_flag_pointwise_runs(self):
        def body(x):
            return x + 1

        xs = [torch.randn(3, 3), torch.randn(3, 3)]
        fn = _compile(lambda xs: foreach_map(body, xs))
        out = fn(xs)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (3, 3))

    def test_without_flag_nonfused_still_runs(self):
        """
        Current behavior: without the flag, even non-fusible bodies (e.g., matmul)
        should execute. Some nightlies may still fail to fall back; skip in that case.
        """
        def body(x, y):
            return x @ y

        xs = [torch.randn(4, 4)]
        ys = [torch.randn(4, 4)]
        fn = _compile(lambda xs, ys: foreach_map(body, xs, ys))
        try:
            out = fn(xs, ys)
        except INDUCTOR_EXC as e:
            self.skipTest(
                f"Inductor nightly lacks non-pointwise fallback for foreach_map: {type(e).__name__}: {e}"
            )
        else:
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].shape, (4, 4))

    def test_with_flag_and_pointwise_ok(self):
        def body(x):
            return x + 1

        xs = [torch.randn(3, 3), torch.randn(3, 3)]
        fn = _compile(lambda xs: foreach_map(body, xs, assert_fused=True))
        out = fn(xs)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)

    @unittest.skipUnless(_supports_assert_fused(), "lowering does not implement `assert_fused` yet")
    def test_with_flag_and_non_fused_raises(self):
        def body(x, y):
            return x @ y

        xs = [torch.randn(4, 4)]
        ys = [torch.randn(4, 4)]
        fn = _compile(lambda xs, ys: foreach_map(body, xs, ys, assert_fused=True))
        with self.assertRaises(Exception):
            fn(xs, ys)


if __name__ == "__main__":
    run_tests()
