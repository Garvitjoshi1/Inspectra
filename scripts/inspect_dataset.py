from pathlib import Path
from collections import defaultdict
import argparse
import os

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

LABEL_EXTENSIONS = {
    ".txt",
    ".json",
    ".xml"
}

YAML_EXTENSIONS = {
    ".yaml",
    ".yml"
}

class DatasetScanner:

    def __init__(self, root):

        self.root = Path(root)

        self.folder_count = 0
        self.file_count = 0

        self.image_count = 0
        self.label_count = 0
        self.yaml_count = 0

        self.extension_count = defaultdict(int)

        self.image_files = []
        self.label_files = []
        self.yaml_files = []

    def scan(self):

        for path in self.root.rglob("*"):

            if path.is_dir():
                self.folder_count += 1
                continue

            self.file_count += 1

            ext = path.suffix.lower()

            self.extension_count[ext] += 1

            if ext in IMAGE_EXTENSIONS:
                self.image_count += 1
                self.image_files.append(path)

            elif ext in LABEL_EXTENSIONS:
                self.label_count += 1
                self.label_files.append(path)

            elif ext in YAML_EXTENSIONS:
                self.yaml_count += 1
                self.yaml_files.append(path)

    def print_summary(self):

        print("=" * 70)

        print(f"DATASET : {self.root.name}")

        print("=" * 70)

        print(f"Folders        : {self.folder_count}")
        print(f"Files          : {self.file_count}")

        print()

        print(f"Images         : {self.image_count}")
        print(f"Labels         : {self.label_count}")
        print(f"YAML Files     : {self.yaml_count}")

        print()

        print("Extension Statistics")
        print("-" * 70)

        for ext, count in sorted(self.extension_count.items()):
            print(f"{ext:<10} {count}")

        print()


    def print_tree(self):

        print("=" * 70)
        print("Folder Tree")
        print("=" * 70)

        self._tree(self.root)

    def _tree(self, folder, prefix=""):

        entries = sorted(folder.iterdir())

        for index, entry in enumerate(entries):

            connector = "└── " if index == len(entries)-1 else "├── "

            print(prefix + connector + entry.name)

            if entry.is_dir():

                extension = "    " if index == len(entries)-1 else "│   "

                self._tree(
                    entry,
                    prefix + extension
                )

def discover_datasets(root):

    datasets = []

    root = Path(root)

    for folder in root.iterdir():

        if folder.is_dir():

            datasets.append(folder)

    return datasets

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--datasets",

        default="datasets",

        help="Datasets Root Folder"

    )

    parser.add_argument(

        "--tree",

        action="store_true"

    )

    args = parser.parse_args()

    datasets = discover_datasets(args.datasets)

    print()

    print(f"Found {len(datasets)} datasets")

    print()

    for dataset in datasets:

        scanner = DatasetScanner(dataset)

        scanner.scan()

        scanner.print_summary()

        if args.tree:
            scanner.print_tree()

        print("\n")

if __name__ == "__main__":

    main()