# hubconf.py

dependencies = ["torch", "py3_wget", "hydra", "omegaconf"]

import torch
from models.sansa.sansa import build_sansa


def sansa(pretrained: bool = True, device: str = "cpu", **kwargs):
    """
    SANSA Model – Torch Hub interface.

    Args:
        pretrained (bool): If True, loads the pretrained SANSA universal checkpoint.
        device (str): Device to map the model to ("cpu" or "cuda").
        **kwargs: Additional arguments passed to build_sansa().

    Returns:
        torch.nn.Module: The SANSA model.
    """

    # 1. Build model
    model = build_sansa(channel_factor=0.8)
    model = model.to(device)

    # 2. Load pretrained weights
    if pretrained:
        checkpoint_url = (
            "https://github.com/ClaudiaCuttano/SANSA/releases/download/v1.0.0/"
            "sansa_universal.pth"
        )

        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url,
            map_location=device,
            progress=True,
        )

        # Your checkpoint stores weights inside state_dict["model"]
        model.load_state_dict(state_dict["model"], strict=False)

    return model
