import torch


def prepare_batches(trees):
    depths = []
    full_trace = []
    for i in range(len(trees)):
        trace, depth = trees[i].unrolled
        for x in trace:
            full_trace.append((i,) + x)
        depths.append(depth)
    return depths, full_trace


def batch(dataset, trace, depth, collected, hidden_dim):
    x, y, children, targets = [], [], [], []
    for item in trace:
        batch, depth_, idx, target = item
        if depth_ == depth:
            children2 = []
            for child_idx in idx[1:]:
                children2.append(collected[(batch, child_idx)])
            children.append(children2)
            x.append(batch)
            y.append(idx[0])
            targets.append(target)
    # if depth == 0:
    #     return x, y, targets, None, None

    children_h, children_c = [], []
    for i, child in enumerate(children):
        h = [z[0] for z in child]
        c = [z[1] for z in child]
        for _ in range(2 - len(h)):
            h.append(torch.zeros(hidden_dim))
            c.append(torch.zeros(hidden_dim))
        children_h.append(torch.stack(h))
        children_c.append(torch.stack(c))

    children_h = torch.stack(children_h).transpose(0, 1)
    children_c = torch.stack(children_c).transpose(0, 1)
    return x, y, targets, children_h, children_c


def unbatch(dataset, x, y, h, c, collected):
    for x_, y_, h_, c_ in zip(x, y, h, c):
        collected[(x_, y_)] = (h_, c_)
    return collected


def get_suffix(args):
    if args.bottleneck == "dvib":
        suffix = ("_taught" if args.teacher_model is not None else "") + \
            f"_beta={args.beta}"
    elif args.bottleneck == "dropout":
        suffix = ("_taught" if args.teacher_model is not None else "") + \
            f"_dropout={args.dropout}"
    elif args.bottleneck == "size":
        suffix = ("_taught" if args.teacher_model is not None else "") + \
            f"_size={args.hidden_dim}"
    else:
        suffix = ""
    return suffix
