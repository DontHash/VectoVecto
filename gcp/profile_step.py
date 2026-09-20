"""profile_step.py — measure where training time goes on the L4 (run on VM)."""
import sys
import time

import torch

sys.path.insert(0, "/opt/srwork")
from rrdbnet import RRDBNet
from unet_discriminator import UNetDiscriminatorSN, RaGANLoss
from perceptual_loss import VGGPerceptualLoss

torch.backends.cudnn.benchmark = True
dev = "cuda"
g = RRDBNet(3, 3, 64, 23, 32, 4).to(dev).train()
d = UNetDiscriminatorSN(3, 64).to(dev).train()
p = VGGPerceptualLoss().to(dev).eval()
gan = RaGANLoss()
optg = torch.optim.Adam(g.parameters(), 1e-4, betas=(0.9, 0.99))
optd = torch.optim.Adam(d.parameters(), 1e-4, betas=(0.9, 0.99))


def timeit(fn, n=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


for patch, batch in [(96, 6), (64, 8), (128, 3)]:
    amp = torch.bfloat16
    lr = torch.rand(batch, 3, patch, patch, device=dev)
    hr = torch.rand(batch, 3, patch * 4, patch * 4, device=dev)

    def g_fwd_only():
        with torch.autocast("cuda", dtype=amp):
            sr = g(lr)
            _ = torch.nn.functional.l1_loss(sr.float(), hr)

    def g_step_full():
        with torch.autocast("cuda", dtype=amp):
            sr = g(lr)
            l1 = torch.nn.functional.l1_loss(sr.float(), hr)
            lp = p(sr.float(), hr)
            loss = l1 + lp + 0.1 * gan.gen_loss(d(hr), d(sr.float()))
        optg.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        optg.step()

    def g_step_nopercep():
        with torch.autocast("cuda", dtype=amp):
            sr = g(lr)
            l1 = torch.nn.functional.l1_loss(sr.float(), hr)
            loss = l1 + 0.1 * gan.gen_loss(d(hr), d(sr.float()))
        optg.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        optg.step()

    def d_step():
        with torch.autocast("cuda", dtype=amp):
            sr = g(lr).detach().float()
            loss = gan.disc_loss(d(hr), d(sr))
        optd.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
        optd.step()

    t_fwd = timeit(g_fwd_only)
    t_full = timeit(g_step_full)
    t_nop = timeit(g_step_nopercep)
    t_d = timeit(d_step)
    tot = t_full + t_d
    print(f"patch={patch} batch={batch}: g_fwd={t_fwd:.3f}s g_step={t_full:.3f}s "
          f"(no-percep {t_nop:.3f}s) d_step={t_d:.3f}s | iter={tot:.3f}s "
          f"-> {1/tot:.2f} it/s", flush=True)
    del lr, hr
    torch.cuda.empty_cache()

print("done")
