"""
C2 entry point: raw C-MAPSS -> engineered feature set on disk.

Run:  python -m src.build_dataset
"""
import time

from . import config, data_loader as dl, features as ft


def main(subset=config.DEFAULT_SUBSET):
    t0 = time.time()

    print(f"Building features for {subset}\n" + "=" * 60)

    train = dl.add_training_rul(dl.load_train(subset))
    dl.summarise(train, f"{subset} train")
    print()

    # Sensor selection is fitted on TRAINING data only, then reused for
    # test. Selecting on the combined set would let test data influence
    # which features exist -- a subtle form of leakage.
    sensors, report = ft.select_sensors(train)
    settings = ft.select_operational_settings(train)
    print()

    train_f, feature_names = ft.build_features(train, sensors, settings)
    print(f"Feature count: {len(feature_names)}")
    print(f"Train matrix:  {train_f[feature_names].shape}")

    test_raw, true_rul = dl.load_test(subset)
    test = dl.add_test_rul(test_raw, true_rul)
    test_f, _ = ft.build_features(test, sensors, settings)
    print(f"Test matrix:   {test_f[feature_names].shape}")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["unit_number", "time_in_cycles", "rul"] + [
        c for c in feature_names if c not in ("time_in_cycles",)
    ]
    train_f[keep].to_csv(config.PROCESSED_DATA_DIR / f"train_{subset}_features.csv", index=False)
    test_f[keep].to_csv(config.PROCESSED_DATA_DIR / f"test_{subset}_features.csv", index=False)
    report.to_csv(config.PROCESSED_DATA_DIR / f"sensor_selection_{subset}.csv")

    print(f"\nWritten to {config.PROCESSED_DATA_DIR}/")
    print(f"Done in {time.time()-t0:.1f}s")
    return train_f, test_f, feature_names


if __name__ == "__main__":
    main()
