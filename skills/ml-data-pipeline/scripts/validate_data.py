#!/usr/bin/env python3
"""Validate dataset for common issues."""

from pathlib import Path
import sys

from PIL import Image


def validate_dataset(data_dir: Path):
    """Validate dataset for common issues."""
    print("=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)

    issues = []

    # 1. Check directory structure
    required_dirs = ["train", "val", "test"]
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            issues.append(f"❌ Missing directory: {dir_name}")
        else:
            print(f"✅ Found {dir_name} directory")

    # 2. Check number of samples
    for split in required_dirs:
        split_dir = data_dir / split
        if split_dir.exists():
            num_samples = len(list(split_dir.glob("**/*.jpg")))
            print(f"   - {split}: {num_samples} samples")
            if num_samples == 0:
                issues.append(f"❌ No samples in {split}")

    # 3. Check class balance
    print("\nClass Distribution:")
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
            class_counts = {cls: len(list((split_dir / cls).glob("*.jpg"))) for cls in classes}
            print(f"\n{split}:")
            for cls, count in class_counts.items():
                print(f"  {cls}: {count}")

            # Check imbalance
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                if min_count > 0 and max_count / min_count > 5:
                    issues.append(
                        f"⚠️  High class imbalance in {split} (ratio: {max_count / min_count:.1f}:1)"
                    )

    # 4. Check image properties
    print("\nChecking image properties...")
    sample_images = list((data_dir / "train").glob("**/*.jpg"))[:100]

    sizes = []
    for img_path in sample_images:
        img = Image.open(img_path)
        sizes.append(img.size)

    # Analyze sizes
    if sizes:
        widths, heights = zip(*sizes)
        print(
            f"Image sizes - Width: [{min(widths)}, {max(widths)}], "
            f"Height: [{min(heights)}, {max(heights)}]"
        )

        if len(set(sizes)) > 10:
            issues.append("⚠️  High variance in image sizes - consider resizing")

    # 5. Check for corrupted files
    print("\nChecking for corrupted files...")
    corrupted = []
    for img_path in sample_images:
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception:
            corrupted.append(str(img_path))
            issues.append(f"❌ Corrupted file: {img_path}")

    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    if not issues:
        print("✅ All checks passed!")
    else:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(issue)

    return len(issues) == 0


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/")
    validate_dataset(data_dir)
