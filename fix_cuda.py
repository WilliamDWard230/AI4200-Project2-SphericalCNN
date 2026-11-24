
from pathlib import Path

root = Path(__file__).resolve().parent / "s2cnn"

files = [
    root / "soft/s2_fft.py",
    root / "soft/so3_fft.py",
    root / "s2_mm.py",
    root / "so3_mm.py",
]

for path in files:
    source = path.read_text()

    if path.name.endswith("_fft.py"):
        old = "if x.is_cuda and x.dtype == torch.float32:"
    else:
        old = "if x.is_cuda:"

    new = "if False:  # Use PyTorch implementation on GPU"

    count = source.count(old)
    if count == 0:
        raise RuntimeError(f"No matching branches in {path}")

    path.write_text(source.replace(old, new))
    print(f"{path.relative_to(root)}: {count} branches disabled")

