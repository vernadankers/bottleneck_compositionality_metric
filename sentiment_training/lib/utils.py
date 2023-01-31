import torch
import random
import numpy as np


def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def report(logger, dataset, metrics, epoch=-1):
    metric_messages = []
    if epoch != -1:
        message = f"Epoch {epoch}, {dataset}, "
    else:
        message = f"Testing... {dataset}, "

    for metric in metrics:
        metric_messages.append(f"{metric}: {metrics[metric]:.4f}")

    message += "\t".join(metric_messages)
    logger.info(message)
