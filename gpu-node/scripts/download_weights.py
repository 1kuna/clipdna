import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ViSiL pretrained weights")
    parser.add_argument(
        "--url",
        default=os.getenv("VISIL_WEIGHTS_URL"),
        help="Weights URL (or set VISIL_WEIGHTS_URL)",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("VISIL_WEIGHTS_PATH", "models/visil/visil_weights.pt"),
        help="Output path for weights",
    )
    args = parser.parse_args()

    if not args.url:
        raise SystemExit("Missing --url or VISIL_WEIGHTS_URL")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading weights from {args.url} -> {output_path}")
    urlretrieve(args.url, output_path)
    print("Download complete")


if __name__ == "__main__":
    main()
