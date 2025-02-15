import numpy as np
import torch


def sharpen_prob(p, temperature=2):
    """Sharpening probability with a temper