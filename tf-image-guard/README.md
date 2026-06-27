# tf-image-guard

> A fault-tolerant CLI toolkit that validates and filters corrupted images for TensorFlow datasets.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why tf-image-guard?

When training TensorFlow models on large image datasets, a single corrupted file can:
- Silently break a `tf.data` pipeline
- Cause mysterious training crashes
- Corrupt batches mid-epoch

**tf-image-guard** pre-screens your dataset through a 3-layer validation pipeline before training begins, ensuring every image is genuinely loadable by TensorFlow.

---

## Features

| Feature | Description |
|---|---|
| **3-Layer Validation** | Filesystem → Pillow → TensorFlow decode |
| **Never Stops** | All errors are caught and logged — execution continues regardless |
| **Parallel Scanning** | Configurable `ThreadPoolExecutor` workers |
| **Quarantine System** | Moves bad files with preserved folder structure + manifest |
| **Dry-Run Mode** | Preview what would be moved without touching files |
| **Rich Console Reports** | Color-coded tables with corruption rates |
| **JSON & CSV Reports** | Machine-readable output for CI/CD pipelines |
| **YAML Configuration** | Fully configurable via config file |

---

## Installation

```bash
# From source (in the project root):
pip install -e .

# With dev dependencies (for testing):
pip install -e ".[dev]"
```

---

## Quick Start

### Scan a directory and see corrupted files
```bash
tf-image-guard scan -i ./my_dataset
```

### Scan and move corrupted files to a quarantine folder
```bash
tf-image-guard clean -i ./my_dataset -o ./quarantine
```

### Preview what would be quarantined (dry run)
```bash
tf-image-guard clean -i ./my_dataset --dry-run
```

### Export a JSON report
```bash
tf-image-guard scan -i ./my_dataset --report-format json --report-output report.json
```

### Use a custom config file
```bash
tf-image-guard scan -i ./my_dataset -c sample_config.yml
```

---

## CLI Reference

### `tf-image-guard scan`
```
Options:
  -i, --input-dir PATH          Directory to scan [required]
  -c, --config PATH             YAML config file
  -l, --log PATH                Write log to file
  -r/-R, --recursive/--no-recursive   Recursively scan subdirectories [default: recursive]
  -w, --workers N               Number of parallel worker threads
  --report-format [console|json|csv]  Report format [default: console]
  --report-output PATH          Write report to file (default: stdout)
  --include-ok                  Include valid files in the report
  -v, --verbose                 Enable debug output
```

### `tf-image-guard clean`
All options from `scan`, plus:
```
  -o, --output-dir PATH         Quarantine directory [default: <input>/_quarantine]
  --dry-run                     Log actions without moving files
```

### `tf-image-guard info`
```
  -c, --config PATH             Show resolved configuration from this file
```

---

## How Validation Works

Each image file passes through 3 sequential layers. Failure at any layer marks the file as corrupted and records the reason:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Filesystem Check                                     │
│  • File exists and is readable                                  │
│  • Size >= min_bytes (default: 100 bytes)                       │
│  • Size <= max_bytes (if configured)                            │
└────────────────────────┬────────────────────────────────────────┘
                         │ PASS
┌────────────────────────▼────────────────────────────────────────┐
│  Layer 2 — Pillow Decode                                        │
│  • PIL.Image.open() succeeds                                    │
│  • img.verify() passes (catches truncated/broken headers)       │
│  • img.load() fully decodes the image                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ PASS
┌────────────────────────▼────────────────────────────────────────┐
│  Layer 3 — TensorFlow Decode                                    │
│  • tf.io.read_file() succeeds                                   │
│  • tf.image.decode_image() succeeds                             │
│  • Tensor .numpy() evaluation succeeds                          │
└────────────────────────┬────────────────────────────────────────┘
                         │ PASS
                      ✓  OK
```

---

## Configuration File

See `sample_config.yml` for all available options. Key settings:

```yaml
# Validation layers
validation:
  filesystem_check: true
  pillow_decode: true
  tensorflow_decode: true

# TensorFlow
tensorflow:
  cpu_only: true   # Prevents GPU allocation during scanning
  channels: 0      # 0=auto, 1=grayscale, 3=RGB, 4=RGBA

# Thresholds
size_limits:
  min_bytes: 100
  max_bytes: null  # null = no upper limit

# Workers
workers: 4
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0`  | All images passed validation |
| `1`  | One or more corrupted images found |
| `2`  | Scan-level error (e.g., invalid input directory) |

---

## Running Tests

```bash
cd "D:\Python Packages\tf-image-guard"
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Project Structure

```
tf-image-guard/
├── pyproject.toml
├── README.md
├── LICENSE
├── sample_config.yml
├── src/
│   └── tf_image_guard/
│       ├── __init__.py
│       ├── cli.py          # Click CLI commands
│       ├── scanner.py      # Parallel directory scanner
│       ├── validators.py   # 3-layer image validation
│       ├── quarantine.py   # File quarantine system
│       ├── reporter.py     # Console/JSON/CSV reports
│       └── utils.py        # Shared utilities
└── tests/
    ├── test_validators.py
    └── test_scanner.py
```

---

## License

MIT — see [LICENSE](LICENSE).
