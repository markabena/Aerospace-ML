import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_loader as dl
from src import config

MOCK = Path(__file__).resolve().parent / "mock_data"

print("="*60)
print("TEST 1: column parsing (trailing-whitespace trap)")
train = dl.load_train("FD001", data_dir=MOCK)
print(f"  Columns parsed: {train.shape[1]} (expected 26)")
assert train.shape[1] == 26, f"Got {train.shape[1]} columns"
assert list(train.columns) == config.ALL_COLUMNS
assert train.isna().sum().sum() == 0, "NaNs present -- phantom columns leaked in"
print("  No phantom NaN columns. PASS\n")

print("TEST 2: training RUL derivation")
train = dl.add_training_rul(train, cap=None)
for unit, lifetime in [(1,150),(2,200),(3,175)]:
    sub = train[train.unit_number==unit]
    first = sub.iloc[0]["rul"]; last = sub.iloc[-1]["rul"]
    assert last == 0, f"Unit {unit} final RUL should be 0, got {last}"
    assert first == lifetime-1, f"Unit {unit} first RUL should be {lifetime-1}, got {first}"
    print(f"  Unit {unit}: RUL {int(first)} -> {int(last)} over {lifetime} cycles. OK")
print("  Uncapped RUL correct. PASS\n")

print("TEST 3: RUL cap")
train_capped = dl.add_training_rul(dl.load_train("FD001", data_dir=MOCK), cap=125)
assert train_capped["rul"].max() == 125
sub = train_capped[train_capped.unit_number==2]
print(f"  Unit 2 (200 cycles) max RUL capped at {int(sub['rul'].max())} (uncapped would be 199)")
assert (sub["rul"].iloc[-1]) == 0
print("  Cap applied, countdown to 0 preserved. PASS\n")

print("TEST 4: test-set RUL from ground-truth file")
test, true_rul = dl.load_test("FD001", data_dir=MOCK)
print(f"  Ground truth RUL per engine: {true_rul.to_dict()}")
test = dl.add_test_rul(test, true_rul, cap=None)
for unit, final_rul, lifetime in [(1,70,80),(2,45,120),(3,130,95)]:
    sub = test[test.unit_number==unit]
    assert sub.iloc[-1]["rul"] == final_rul, f"Unit {unit} final RUL mismatch"
    expected_first = final_rul + lifetime - 1
    assert sub.iloc[0]["rul"] == expected_first
    print(f"  Unit {unit}: first row RUL {int(sub.iloc[0]['rul'])}, final row RUL {int(sub.iloc[-1]['rul'])} (truth: {final_rul}). OK")
print("  Test RUL back-calculation correct. PASS\n")

print("TEST 5: constant sensor detection")
flat = dl.constant_sensors(train)
print(f"  Detected constant sensors: {flat}")
assert set(flat) == {"sensor_1","sensor_10","sensor_5"}, f"Got {flat}"
print("  Correctly found the 3 planted flat sensors. PASS\n")

print("TEST 6: summarise()")
dl.summarise(train, "mock train")
print()
print("="*60)
print("ALL LOADER TESTS PASSED")
