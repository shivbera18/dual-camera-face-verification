"""Quick setup verification — run after install."""
import sys
import os


def check_packages():
    ok = True
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("insightface", "InsightFace"),
        ("albumentations", "Albumentations"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
    ]
    for mod, name in packages:
        try:
            __import__(mod)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISS] {name}")
            ok = False
    return ok


def check_dirs():
    root = os.path.dirname(os.path.dirname(__file__))
    required = [
        "data/raw/lfw",
        "data/raw/faceforensicspp",
        "data/processed/deepfake_faces/train/real",
        "data/processed/deepfake_faces/train/fake",
        "data/processed/deepfake_faces/val/real",
        "data/processed/deepfake_faces/val/fake",
        "data/processed/deepfake_faces/test/real",
        "data/processed/deepfake_faces/test/fake",
        "data/splits",
        "artifacts/models",
        "artifacts/metrics",
        "artifacts/logs",
        "configs",
    ]
    ok = True
    for d in required:
        path = os.path.join(root, d)
        if os.path.isdir(path):
            print(f"  [OK] {d}")
        else:
            print(f"  [MISS] {d}")
            ok = False
    return ok


def check_lfw():
    root = os.path.dirname(os.path.dirname(__file__))
    lfw_dir = os.path.join(root, "data/raw/lfw/lfw")
    if os.path.isdir(lfw_dir):
        count = len(os.listdir(lfw_dir))
        print(f"  [OK] LFW: {count} person directories")
        return True
    else:
        print("  [MISS] LFW not extracted — run: tar -xzf data/raw/lfw/lfw.tgz -C data/raw/lfw/")
        return False


def check_insightface_models():
    model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
    if os.path.isdir(model_dir):
        files = os.listdir(model_dir)
        print(f"  [OK] InsightFace buffalo_l: {len(files)} model files")
        return True
    else:
        print("  [PENDING] InsightFace buffalo_l not yet downloaded")
        print("    Run: python -c \"from insightface.app import FaceAnalysis; FaceAnalysis('buffalo_l').prepare(ctx_id=-1)\"")
        return False


if __name__ == "__main__":
    print("=== Package check ===")
    p = check_packages()
    print("\n=== Directory check ===")
    d = check_dirs()
    print("\n=== LFW dataset ===")
    l = check_lfw()
    print("\n=== InsightFace models ===")
    m = check_insightface_models()

    print("\n=== FaceForensics++ ===")
    root = os.path.dirname(os.path.dirname(__file__))
    ff_repo = os.path.join(root, "data/raw/faceforensicspp/repo")
    if os.path.isdir(ff_repo):
        print("  [OK] FF++ repo cloned")
    else:
        print("  [MISS] FF++ repo missing — run: scripts/download_faceforensicspp.sh")
    print("  [ACTION] Fill Google form for data access:")
    print("    https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform")

    if all([p, d]):
        print("\n[READY] Core setup complete. Download gated datasets when access arrives.")
    else:
        print("\n[INCOMPLETE] Fix above before training.")
