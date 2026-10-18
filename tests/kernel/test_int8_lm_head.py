import pytest
import torch
import torch.nn.functional as F

from minisgl.kernel.triton.int8_lm_head import int8_lm_head, quantize_weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 4, 16])
def test_int8_lm_head_shapes_and_peak(m):
    torch.manual_seed(0)
    k, n = 256, 4096
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    x = weight[0:1].repeat(m, 1).contiguous()
    int8_weight, scale = quantize_weight(weight)
    out = int8_lm_head(x, weight, int8_weight, scale, refine_k=64)
    ref = F.linear(x, weight)
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.argmax(dim=-1), ref.argmax(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_int8_lm_head_top5_overlap():
    torch.manual_seed(1)
    k, n = 256, 4096
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    int8_weight, scale = quantize_weight(weight)
    out = int8_lm_head(x, weight, int8_weight, scale, refine_k=64)
    ref = F.linear(x, weight)
    ref_top5 = ref.topk(5, dim=-1).indices
    out_top5 = out.topk(5, dim=-1).indices
    overlap = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ref_top5, out_top5))
    assert overlap >= int(0.95 * 5 * x.shape[0])
