import argparse

from core_pipeline.data.loader import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["m5", "vn2"], required=True)
    args = parser.parse_args()
    load_dataset(args.dataset)


if __name__ == "__main__":
    main()
