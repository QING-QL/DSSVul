import csv
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


PARTITIONS = ("labeled", "unlabeled", "validation", "test")
FIELDS = ("sample_id", "code", "target", "project", "commit_id", "source_split", "source_file", "source_row")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_manifest(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def manifest_order(saved):
    count = sum(len(saved["indices"][key]) for key in PARTITIONS)
    ids = [None] * count
    for key in PARTITIONS:
        indices = saved["indices"][key]
        values = saved["sample_ids"][key]
        if len(indices) != len(values):
            raise ValueError("Split index/sample-ID lengths differ")
        for index, value in zip(indices, values):
            if type(index) is not int or not 0 <= index < count or ids[index] is not None:
                raise ValueError("Split indices must cover the corpus exactly once")
            ids[index] = value
    if None in ids or len(set(ids)) != count:
        raise ValueError("Split IDs must be unique and complete")
    return ids


def records(path):
    path = Path(path)
    csv.field_size_limit(64 * 1024 * 1024)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)
        return
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            yield from batch.to_pylist()
        return
    with path.open(encoding="utf-8-sig") as stream:
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        stream.seek(0)
        if first != "[":
            for line in stream:
                if line.strip():
                    yield json.loads(line)
            return
        decoder = json.JSONDecoder()
        buffer = stream.read(1024 * 1024).lstrip()[1:]
        eof = False
        expect_value = True
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                block = stream.read(1024 * 1024)
                if not block:
                    raise ValueError("Truncated JSON array")
                buffer += block
                continue
            if buffer[0] == "]":
                if (buffer[1:] + stream.read()).strip():
                    raise ValueError("Trailing JSON data")
                return
            if not expect_value:
                if buffer[0] != ",":
                    raise ValueError("Missing JSON array delimiter")
                buffer = buffer[1:]
                expect_value = True
                continue
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                block = stream.read(1024 * 1024)
                eof = not block
                buffer += block
                continue
            if not isinstance(row, dict):
                raise ValueError("Dataset records must be objects")
            yield row
            buffer = buffer[end:]
            expect_value = False


def binary(value):
    if value in (0, 1, False, True) and not isinstance(value, str):
        return int(value)
    if isinstance(value, str) and value.lower() in ("0", "1", "false", "true"):
        return int(value.lower() in ("1", "true"))
    raise ValueError("Expected a binary vulnerability label")


def normalize(row, dataset, partition=None, text_column=None, label_column=None):
    if text_column or label_column:
        if not text_column or not label_column:
            raise ValueError("Provide both text_column and label_column")
        code, label = row[text_column], binary(row[label_column])
    elif "code" in row and "target" in row:
        code, label = row["code"], binary(row["target"])
    elif dataset == "megavul":
        label = binary(row["is_vul"])
        code = row["func_before"] if label else row["func"]
    elif dataset == "reveal":
        code = row.get("functionSource", row.get("code", row.get("func")))
        label = binary(row["label"])
    elif dataset == "bigvul":
        code, label = row["func_before"], binary(row["vul"])
    else:
        code, label = row["func"], binary(row["target"])
    if not isinstance(code, str) or not code.strip():
        return None
    source_split = partition or row.get("split") or row.get("source_split") or ""
    source_split = {"valid": "validation", "val": "validation"}.get(source_split, source_split)
    return dict(sample_id=hashlib.sha256(code.encode("utf-8")).hexdigest(), code=code, target=label,
                project=str(row.get("project") or row.get("repo_name") or ""),
                commit_id=str(row.get("commit_id") or row.get("commit_hash") or ""), source_split=source_split)


def new_splits(ids, labels, groups, source_splits, dataset, seed, fraction):
    indices = np.arange(len(ids))
    y = np.asarray(labels)
    if dataset == "primevul":
        if set(source_splits) != {"train", "validation", "test"}:
            raise ValueError("PrimeVul requires all three source partitions")
        split_array = np.asarray(source_splits)
        train = indices[split_array == "train"]
        valid = indices[split_array == "validation"]
        test = indices[split_array == "test"]
    elif dataset in ("diversevul", "megavul"):
        group_array = np.asarray(groups)
        _, test = next(StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(indices, y, group_array))
        remaining = np.setdiff1d(indices, test)
        a, b = next(StratifiedGroupKFold(9, shuffle=True, random_state=seed).split(remaining, y[remaining], group_array[remaining]))
        train, valid = remaining[a], remaining[b]
    else:
        train, rest = train_test_split(indices, test_size=0.2, random_state=seed, stratify=y)
        valid, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    labeled, unlabeled = train_test_split(train, train_size=fraction, random_state=seed, stratify=y[train])
    splits = {key: np.sort(value).tolist() for key, value in zip(PARTITIONS, (labeled, unlabeled, valid, test))}
    fingerprint = hashlib.sha256()
    for sid, label in zip(ids, labels):
        fingerprint.update(json.dumps([sid, int(label)], ensure_ascii=True).encode())
    return dict(fingerprint=fingerprint.hexdigest(), indices=splits,
                sample_ids={key: [ids[i] for i in values] for key, values in splits.items()},
                sizes={key: len(values) for key, values in splits.items()},
                class_counts_for_audit_only={key: np.bincount(y[values], minlength=2).tolist() for key, values in splits.items()},
                seed=seed, labeled_fraction=fraction)


def validate_dataset(data, splits, dataset):
    saved = read_manifest(splits)
    ids = manifest_order(saved)
    partitions = [None] * len(ids)
    counts = {key: [0, 0] for key in PARTITIONS}
    groups = {}
    source_partition_differences = 0
    fingerprint = hashlib.sha256()
    count = 0
    for key in PARTITIONS:
        for index in saved["indices"][key]:
            partitions[index] = key
    for index, row in enumerate(records(data)):
        value = normalize(row, dataset)
        if value is None or index >= len(ids) or value["sample_id"] != ids[index]:
            raise ValueError("Canonical data does not match the fixed sample order")
        if row.get("content_sha256", row.get("sample_id", value["sample_id"])) != value["sample_id"]:
            raise ValueError("Code SHA256 mismatch")
        label = value["target"]
        fingerprint.update(json.dumps([ids[index], label], ensure_ascii=True).encode())
        counts[partitions[index]][label] += 1
        partition = "train" if partitions[index] in ("labeled", "unlabeled") else partitions[index]
        if value["source_split"] and value["source_split"] != partition:
            source_partition_differences += 1
        group = value["project"] + "::" + value["commit_id"] if value["project"] and value["commit_id"] else "code::" + ids[index]
        groups.setdefault(group, set()).add(partition)
        count += 1
    if count != len(ids) or fingerprint.hexdigest() != saved["fingerprint"]:
        raise ValueError("Corpus count/label fingerprint mismatch")
    if counts != saved["class_counts_for_audit_only"]:
        raise ValueError("Split class counts mismatch")
    cross_groups = sum(len(values) > 1 for values in groups.values())
    if dataset in ("diversevul", "megavul") and cross_groups:
        raise ValueError("Repository/commit groups cross partitions")
    if dataset == "primevul" and source_partition_differences:
        raise ValueError("PrimeVul source partition mismatch")
    return dict(dataset=dataset, rows=count, counts_NV_V=[sum(x[k] for x in counts.values()) for k in (0, 1)],
                split_counts_NV_V=counts, fingerprint=fingerprint.hexdigest(), data_sha256=sha256(data),
                splits_sha256=sha256(splits), cross_partition_groups=cross_groups,
                source_partition_differences=source_partition_differences)


def prepare_dataset(dataset, inputs, output, splits=None, source_splits=None, seed=42, fraction=0.1,
                    text_column=None, label_column=None):
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    if source_splits and len(source_splits) != len(inputs):
        raise ValueError("Provide one source partition per input")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    database = out / "processing.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE samples (id TEXT PRIMARY KEY, code TEXT, target INT, project TEXT, commit_id TEXT, source_split TEXT, source_file TEXT, source_row INT, conflict INT DEFAULT 0)")
    totals = dict(raw_rows=0, blank_rows=0, duplicate_rows=0, conflicting_groups=0, outside_manifest=0)
    evidence = []
    try:
        with (out / "exclusions.csv").open("w", newline="", encoding="utf-8") as stream:
            ledger = csv.writer(stream)
            ledger.writerow(["source_file", "source_row", "sample_id", "reason"])
            for number, path in enumerate(inputs):
                path = Path(path)
                evidence.append(dict(file=path.name, sha256=sha256(path)))
                for index, raw in enumerate(records(path)):
                    totals["raw_rows"] += 1
                    value = normalize(raw, dataset, source_splits[number] if source_splits else None, text_column, label_column)
                    if value is None:
                        totals["blank_rows"] += 1
                        ledger.writerow([path.name, index, "", "blank_code"])
                        continue
                    sid = value["sample_id"]
                    existing = connection.execute("SELECT target, conflict, source_split FROM samples WHERE id=?", (sid,)).fetchone()
                    if existing:
                        if existing[0] != value["target"]:
                            if not existing[1]:
                                totals["conflicting_groups"] += 1
                            connection.execute("UPDATE samples SET conflict=1 WHERE id=?", (sid,))
                            ledger.writerow([path.name, index, sid, "conflicting_label_group"])
                        else:
                            totals["duplicate_rows"] += 1
                            ledger.writerow([path.name, index, sid, "same_label_duplicate"])
                        if dataset == "primevul" and existing[2] != value["source_split"]:
                            raise ValueError("PrimeVul duplicate code crosses source partitions")
                    else:
                        connection.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,0)",
                                           (sid, value["code"], value["target"], value["project"], value["commit_id"], value["source_split"], path.name, index))
                connection.commit()
            saved = read_manifest(splits) if splits else None
            order = manifest_order(saved) if saved else [r[0] for r in connection.execute("SELECT id FROM samples WHERE conflict=0 ORDER BY rowid")]
            expected = set(order)
            for sid, filename, index, conflict in connection.execute("SELECT id, source_file, source_row, conflict FROM samples"):
                if conflict or sid not in expected:
                    ledger.writerow([filename, index, sid, "excluded_conflicting_group" if conflict else "outside_fixed_manifest"])
                    totals["outside_manifest"] += int(not conflict and sid not in expected)
            labels, groups, memberships = [], [], []
            with (out / "canonical.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.writer(target)
                writer.writerow(FIELDS)
                for sid in order:
                    row = connection.execute("SELECT id,code,target,project,commit_id,source_split,source_file,source_row FROM samples WHERE id=? AND conflict=0", (sid,)).fetchone()
                    if row is None:
                        raise ValueError("Input release is missing a fixed sample or has conflicting labels: " + sid)
                    writer.writerow(row)
                    labels.append(row[2])
                    groups.append(row[3] + "::" + row[4] if row[3] and row[4] else "code::" + sid)
                    memberships.append(row[5])
            if saved is None:
                saved = new_splits(order, labels, groups, memberships, dataset, seed, fraction)
            split_path = out / "splits_seed42.json" if splits else out / f"splits_seed{seed}.json"
            split_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            result = validate_dataset(out / "canonical.csv", split_path, dataset)
            result.update(processing=totals, sources=evidence, fixed_sample_manifest=bool(splits),
                          seed=saved.get("seed", seed), labeled_fraction=saved.get("labeled_fraction", fraction))
            (out / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
    finally:
        connection.close()
        database.unlink()
