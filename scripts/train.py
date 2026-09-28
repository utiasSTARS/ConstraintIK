"""Train the constraint-conditioned diffusion model (paper App. C.3; defaults = paper settings).

    python scripts/train.py --robot ur5       # data/train_ur5.pt -> checkpoints/retrained/ur5.pt
    python scripts/train.py --robot franka --device cuda:1

Writes a schema-v1 checkpoint (EMA weights used at inference) and a JSON sidecar <out>.config.json.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from tqdm import tqdm

from constraint_ik.checkpoint import save_checkpoint
from constraint_ik.data import StratifiedSampler, load_training_set
from constraint_ik.diffusion import make_schedule, q_sample
from constraint_ik.model import EMA, CondEndpointDDPM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", required=True, choices=["ur5", "franka"])
    p.add_argument("--data", default=None, help="default: data/train_{robot}.pt")
    p.add_argument("--out", default=None, help="default: checkpoints/retrained/{robot}.pt")
    p.add_argument("--schedule", default="sigmoid", choices=["sigmoid", "cosine", "linear"])
    p.add_argument("--T", type=int, default=100, help="diffusion steps")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cond_dropout", type=float, default=0.10, help="probability of zeroing the whole cond")
    p.add_argument("--ema_decay", type=float, default=0.9995)
    p.add_argument("--ema_warmup", type=int, default=1000, help="EMA copies the weights before this step")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--emb_dim", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=0, help="also save <out stem>_stepNNNNNN.pt (0: off)")
    a = p.parse_args()
    a.data = a.data or f"data/train_{a.robot}.pt"
    a.out = a.out or f"checkpoints/retrained/{a.robot}.pt"
    return a


def main() -> None:
    args = parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    data = load_training_set(args.data, dev)
    if data.robot != args.robot:
        raise ValueError(f"{args.data} holds {data.robot} data")
    model = CondEndpointDDPM(data.x.shape[1], base_channels=args.width, emb_dim=args.emb_dim,
                             depth=args.depth).to(dev)
    alphas = make_schedule(args.schedule, args.T).to(dev)
    ema = EMA(model, args.ema_decay, args.ema_warmup)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batches = iter(StratifiedSampler(data.type_indices, args.batch))

    def save(path: str, step: int) -> None:
        save_checkpoint(path, model=model, ema_state=ema.state_dict(), x_normalizer=data.x_normalizer,
                        pos_normalizer=data.pos_normalizer, alphas=alphas, robot=args.robot, step=step,
                        args=vars(args), type_names=tuple(data.type_names))

    print(f"{args.robot}: {len(data.x)} configurations in {data.type_names}; "
          f"{sum(p.numel() for p in model.parameters())} parameters on {dev}")
    model.train()
    loss_sum, log = torch.zeros((), device=dev), []
    t0 = time.time()
    bar = tqdm(range(args.steps), mininterval=10)
    for step in bar:
        idx = next(batches).to(dev)
        x0, cond = data.x[idx], data.cond[idx]
        cond[torch.rand(args.batch, device=dev) < args.cond_dropout] = 0.0
        t = torch.randint(0, args.T, (args.batch,), device=dev)
        noise = torch.randn_like(x0)
        loss = (model(q_sample(x0, t, noise, alphas), t, cond) - noise).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model, step)
        loss_sum += loss.detach()
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            n = (step % args.log_every) + 1
            log.append((step + 1, loss_sum.item() / n))
            loss_sum.zero_()
            bar.set_postfix(loss=f"{log[-1][1]:.4f}")
        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < args.steps:
            save(args.out.replace(".pt", f"_step{step + 1:06d}.pt"), step + 1)

    seconds = time.time() - t0
    save(args.out, args.steps)
    with open(args.out.replace(".pt", ".config.json"), "w") as f:
        json.dump({**vars(args), "n_configurations": len(data.x), "type_names": data.type_names,
                   "seconds": round(seconds, 1), "steps_per_s": round(args.steps / seconds, 2),
                   "loss": log}, f, indent=2)
    print(f"wrote {args.out} ({seconds:.0f} s, {args.steps / seconds:.1f} steps/s, loss {log[-1][1]:.4f})")


if __name__ == "__main__":
    main()
