from __future__ import annotations

import argparse
import json

from napari_epicycle.io_registry import migrate_training_data_tree


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy training_data round metadata to canonical Phase A contracts."
    )
    parser.add_argument("working_dir", help="Path to working directory containing training_data/")
    parser.add_argument(
        "--training-data-root",
        default="training_data",
        help="Relative training data root inside working_dir (default: training_data)",
    )
    parser.add_argument(
        "--skip-selected-indices",
        action="store_true",
        help="Do not generate selected_indices.json during migration",
    )
    parser.add_argument(
        "--overwrite-selected-indices",
        action="store_true",
        help="Overwrite existing selected_indices.json files",
    )
    args = parser.parse_args()

    summaries = migrate_training_data_tree(
        working_dir=args.working_dir,
        training_data_root=args.training_data_root,
        write_selected_indices=not args.skip_selected_indices,
        overwrite_selected_indices=args.overwrite_selected_indices,
    )
    print(json.dumps({"migrated_rounds": len(summaries), "rounds": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
