import torch

from tmp.GRUC_0813_wSDR_drb_only_pre100ms_ep66_p2_ncnn import test_inference as ti1
from tmp.GRUC_0813_wSDR_drb_only_pre100ms_ep66_p2_pnnx import test_inference as ti2


def cos_sim(a, b):
    print(a.shape, b.shape)
    a = torch.squeeze(a)
    b = torch.squeeze(b)
    a_norm = torch.linalg.norm(a)
    b_norm = torch.linalg.norm(b)
    cos = torch.sum(a * b) / (a_norm * b_norm)
    return cos


out1 = ti1()
out2 = ti2()
res = [cos_sim(a, b) for a, b in zip(out1, out2)]
print(res)
...
