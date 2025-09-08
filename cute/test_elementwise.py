import cutlass
import elementwise as elt
import pytest
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack


class TestElementwise:
    def setup_method(self):
        cutlass.cuda.initialize_cuda_context()
        self._A = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        self._B = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        self._C = torch.zeros(2048, 2048, device="cuda", dtype=torch.float16)

        self.A = from_dlpack(self._A)
        self.B = from_dlpack(self._B)
        self.C = from_dlpack(self._C)

    def test_naive_elementwise_1d(self):
        compiled = cute.compile(
            elt.naive_elementwise_1d, self.A, self.B, self.C, elt.add
        )
        compiled(self.A, self.B, self.C)
        assert torch.allclose(self._C, self._A + self._B)

    def test_elementwise_1d_tiled(self):
        compiled = cute.compile(
            elt.elementwise_1d_tiled, self.A, self.B, self.C, elt.add
        )
        compiled(self.A, self.B, self.C)
        assert torch.allclose(self._C, self._A + self._B)

    def test_elementwise_1d_tiled_tv_layout(self):
        elems_per_thread = 8 * 8
        num_threads_per_block = 256
        A = elt.reshape_torch_Tensor_for_tv_layout(
            self._A, elems_per_thread, num_threads_per_block
        )
        B = elt.reshape_torch_Tensor_for_tv_layout(
            self._B, elems_per_thread, num_threads_per_block
        )
        C = elt.reshape_torch_Tensor_for_tv_layout(
            self._C, elems_per_thread, num_threads_per_block
        )

        _A = from_dlpack(A)
        _B = from_dlpack(B)
        _C = from_dlpack(C)

        compiled = cute.compile(
            elt.elementwise_1d_tiled_tv_layout, _A, _B, _C, elems_per_thread, 8, elt.add
        )
        compiled(_A, _B, _C)
        assert torch.allclose(C, A + B)


if __name__ == "__main__":
    pytest.main()
