from pathlib import Path
import pandas as pd


def _normalize_join_key(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    # Accepts: item_id, itemid, item
    for c in ("item_id", "itemid", "item"):
        if c in df.columns:
            return df.rename(columns={c: "item_id"})
    raise KeyError(f"[{df_name}] Missing join key. Expected one of: item_id, itemid, item")


def build_ground_truth(
    lab_events_path: Path,
    lab_items_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    print("Loading lab_events...")
    lab_events = pd.read_csv(lab_events_path, compression="infer")
    lab_events = _normalize_join_key(lab_events, "lab_events")

    print("Loading lab_items...")
    # Only useful columns from lab_items header
    lab_items = pd.read_csv(
        lab_items_path,
        compression="infer",
        usecols=["itemid", "label", "fluid", "category"],
    )
    lab_items = _normalize_join_key(lab_items, "lab_items")
    lab_items = lab_items.drop_duplicates(subset=["item_id"])

    print("Joining on item_id...")
    joined = lab_events.merge(
        lab_items,
        on="item_id",
        how="left",  # use "inner" if you only want perfect matches
    )

    joined["ground_truth_itemid"] = joined["item_id"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)

    print(f"Done. Rows joined: {len(joined)}")
    print(f"Saved: {output_path}")
    return joined


if __name__ == "__main__":
    lab_events_path = Path("datasets/mimiciv_demo/hosp/labevents.csv.gz")
    lab_items_path = Path("datasets/mimiciv_demo/hosp/d_labitems.csv.gz")
    output_path = Path("datasets/ground_truth_joined_lab_events_lab_items.csv")

    build_ground_truth(
        lab_events_path=lab_events_path,
        lab_items_path=lab_items_path,
        output_path=output_path,
    )