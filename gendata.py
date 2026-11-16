"""Generate Spherical MNIST using the method in jonkhler/s2cnn examples/mnist/gendata.py.

Updated for modern torchvision, with the Driscoll-Healy grid implemented
locally instead of importing lie_learn. Requires numpy, torch, torchvision.
"""
import argparse
import gzip
import pickle

import numpy as np
from torchvision import datasets

NORTHPOLE_EPSILON = 1e-3


def rand_rotation_matrix(deflection=1.0):
    theta, phi, z = np.random.uniform(size=3)
    theta *= 2.0 * deflection * np.pi
    phi *= 2.0 * np.pi
    z *= 2.0 * deflection
    r = np.sqrt(z)
    V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    return (np.outer(V, V) - np.eye(3)).dot(R)


def get_projection_grid(b):
    # Identical Driscoll-Healy angular sampling to lie_learn.spaces.S2.meshgrid.
    theta = np.arange(2 * b) * np.pi / (2.0 * b)
    phi = np.arange(2 * b) * np.pi / b
    theta, phi = np.meshgrid(theta, phi, indexing="ij")
    return (np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta))


def rotate_grid(rotation, grid):
    x, y, z = grid
    return tuple(np.einsum("ij,jab->iab", rotation, np.array((x, y, z))))


def project_sphere_on_xy_plane(grid, projection_origin):
    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1
    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y
    xmin = 0.5 * (-1 - sx) - 1
    ymin = 0.5 * (-1 - sy) - 1
    return (qx - xmin) / (2 * abs(xmin)), (qy - ymin) / (2 * abs(ymin))


def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds
    valid = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)
    sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]), dtype=np.float64)
    sample[:, valid] = signal[:, x[valid], y[valid]]
    return sample


def sample_bilinear(signal, rx, ry):
    dim_x, dim_y = signal.shape[1:]
    rx = rx.copy() * dim_x
    ry = ry.copy() * dim_y
    ix0, iy0 = rx.astype(int), ry.astype(int)
    ix1, iy1 = ix0 + 1, iy0 + 1
    bounds = (0, dim_x, 0, dim_y)
    s00 = sample_within_bounds(signal, ix0, iy0, bounds)
    s10 = sample_within_bounds(signal, ix1, iy0, bounds)
    s01 = sample_within_bounds(signal, ix0, iy1, bounds)
    s11 = sample_within_bounds(signal, ix1, iy1, bounds)
    fx1 = (ix1 - rx) * s00 + (rx - ix0) * s10
    fx2 = (ix1 - rx) * s01 + (rx - ix0) * s11
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def project_2d_on_sphere(signal, grid):
    origin = (0, 0, 2 + NORTHPOLE_EPSILON)
    rx, ry = project_sphere_on_xy_plane(grid, origin)
    samples = sample_bilinear(signal, rx, ry)
    samples *= (grid[2] <= 1).astype(np.float64)
    minimum = samples.min(axis=(1, 2), keepdims=True)
    maximum = samples.max(axis=(1, 2), keepdims=True)
    # Safeguard added for a completely blank projection.
    samples = (samples - minimum) / np.maximum(maximum - minimum, 1e-12)
    return (samples * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth", type=int, default=30)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--mnist_data_folder", default="MNIST_data")
    parser.add_argument("--output_file", default="s2_mnist.gz")
    parser.add_argument("--no_rotate_train", action="store_true")
    parser.add_argument("--no_rotate_test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    result = {}
    grid = get_projection_grid(args.bandwidth)
    for name, train, unrotated in (("train", True, args.no_rotate_train),
                                   ("test", False, args.no_rotate_test)):
        mnist = datasets.MNIST(root=args.mnist_data_folder, train=train, download=True)
        images = mnist.data.numpy().astype(np.float64)
        labels = mnist.targets.numpy()
        projections = np.empty((len(images), 2 * args.bandwidth, 2 * args.bandwidth), dtype=np.uint8)
        print(f"Projecting {name} ({len(images)} images)", flush=True)
        for start in range(0, len(images), args.chunk_size):
            end = min(start + args.chunk_size, len(images))
            rotated_grid = grid if unrotated else rotate_grid(rand_rotation_matrix(args.noise), grid)
            projections[start:end] = project_2d_on_sphere(images[start:end], rotated_grid)
            print(f"\r{end}/{len(images)}", end="", flush=True)
        print()
        result[name] = {"images": projections, "labels": labels}
    with gzip.open(args.output_file, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {args.output_file}")


if __name__ == "__main__":
    main()
