import argparse
import numpy as np
import torch
from pathlib import Path

import dnnlib

import legacy

def factorize(G):
    
    mod_params = [v for k, v in G.named_parameters() if 'affine' in k and 'weight' in k and "torgb" not in k]
    W = torch.cat(mod_params, 0)
    
    eigvec = torch.svd(W).V

    return eigvec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument("--out", type=str, requited=True, help="path to output file")
    parser.add_argument("ckpt", type=str, help="name of the model checkpoint")

    args = parser.parse_args()
    device = 'cuda'
    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    eigvec = factorize(G)
    torch.save(eigvec, args.out)
