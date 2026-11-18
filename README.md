# CAPPM: Six-Task Implementation

This project contains CAPPM models, data preparation, training, and evaluation for six tasks derived from two data sources: ANMerge and ADNI. 

## Project layout

```text
cappm/                      # Project root
├── README.md
├── DATA_PROCESSING.md
├── requirements.txt
├── data/                   # Empty placeholder for input data
├── processed_data/         # Created by the preparation commands
├── processing/
│   ├── ANM/
│   └── ADNI/
└── cappm/                  # Model, training, and evaluation code
```

The project contains an empty data/ directory. Actual input files are currently stored in the separate sibling data/ directory, outside this project. Before using the default commands, copy its six task subdirectories into the project's data/ directory. Alternatively, keep the data outside the project and replace `--raw-root ./data` with `--raw-root ../data`.

## Installation

```bash
python -m pip install -r requirements.txt
```

Use a separate Python environment and install a PyTorch build suitable for your hardware. The programs prefer CUDA when available. Add `--device cpu` to use the CPU, or `--num_workers 0` to disable worker processes during data loading.

## Data preparation

Both processing directories contain the original project notebooks, with execution outputs cleared. Their original machine-specific paths and alternative experiment branches remain; running all notebook cells is not a prerequisite for the Python workflow below.

The standalone preparation entry points are:

| Entry point | Supported task names |
|---|---|
| processing/ANM/prepare_tasks.py | ANM, ANM_MMSE, ANM_early |
| processing/ADNI/prepare_tasks.py | ADNI, ADNI_MMSE, ADNI_early |

These programs start from the existing proteomics CSV files and MRI slice pickle files. They do not implement the unavailable upstream conversion from raw MRI scans.

From the project root, run each task separately:

### ANM

```bash
python processing/ANM/prepare_tasks.py --dataset ANM --raw-root ./data --output-root ./processed_data --seed 42 --duplicate-policy first
```

### ANM_MMSE

```bash
python processing/ANM/prepare_tasks.py --dataset ANM_MMSE --raw-root ./data --output-root ./processed_data --seed 42 --duplicate-policy first --merge-lowest-mmse
```

### ANM_early

```bash
python processing/ANM/prepare_tasks.py --dataset ANM_early --raw-root ./data --output-root ./processed_data --seed 42 --duplicate-policy first
```

### ADNI

```bash
python processing/ADNI/prepare_tasks.py --dataset ADNI --raw-root ./data --output-root ./processed_data --seed 42 --duplicate-policy first
```

### ADNI_MMSE

```bash
python processing/ADNI/prepare_tasks.py --dataset ADNI_MMSE --raw-root ./data --output-root ./processed_data --seed 42 --duplicate-policy first --adni-mmse-valid-only
```

### ADNI_early

```bash
python processing/ADNI/prepare_tasks.py --dataset ADNI_early --raw-root ./data --output-root ./processed_data --seed 42 --duplicate-policy first --resize-mri
```

The explicit processing options have the following meanings:

- `--duplicate-policy first`: retain the first MRI record in source order for each duplicated visit before splitting.
- `--merge-lowest-mmse`: merge ANM_MMSE source label 5 into label 4.
- `--adni-mmse-valid-only`: restrict ADNI_MMSE to labels 0-3.
- `--resize-mri`: resize ADNI_early MRI arrays from 224 x 224 to 72 x 72 using bilinear interpolation.

Label and image-processing conventions must match the intended experiment. The MMSE conventions follow the local preparation scripts and still require confirmation against the manuscript's bin definitions.

Outputs are written to processed_data/TASK/overlap/. Subjects are assigned to seven approximately balanced folds: five for training, one for validation, and one for testing. All visits of a subject remain in one split. An existing nonempty output directory is not overwritten unless `--overwrite` is explicitly supplied.

## Training and evaluation

From the project root, enter the model directory:

```bash
cd cappm
```

Each training or evaluation command uses seeds 0-4 by default. Add `--seeds 0` to run only seed 0. Training selects the best checkpoint using validation macro-F1. Test evaluation is performed separately using eval.py.

The following paths assume that preparation outputs were written to the project's processed_data/ directory. Do not pass the unprepared data/ directory as `--data_root`.

### ANM

```bash
python train.py --dataset ANM --data_root ../processed_data --epochs 50 --batch_size 16 --learning_rate 0.0001 --optimizer adam --gamma 0.95 --lr_schedule constant --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --train
python eval.py --dataset ANM --data_root ../processed_data --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --split test
```

### ANM_MMSE

```bash
python train.py --dataset ANM_MMSE --data_root ../processed_data --epochs 50 --batch_size 16 --learning_rate 0.0001 --optimizer adam --gamma 0.95 --lr_schedule constant --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --train
python eval.py --dataset ANM_MMSE --data_root ../processed_data --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --split test
```

### ANM_early

```bash
python train.py --dataset ANM_early --data_root ../processed_data --epochs 50 --batch_size 16 --learning_rate 0.0001 --optimizer adam --gamma 0.95 --lr_schedule constant --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --train
python eval.py --dataset ANM_early --data_root ../processed_data --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --split test
```

### ADNI

```bash
python train.py --dataset ADNI --data_root ../processed_data --epochs 50 --batch_size 16 --learning_rate 0.001 --optimizer adam --gamma 1.1 --lr_schedule constant --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --train
python eval.py --dataset ADNI --data_root ../processed_data --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --split test
```

### ADNI_MMSE

```bash
python train.py --dataset ADNI_MMSE --data_root ../processed_data --epochs 50 --batch_size 16 --learning_rate 0.001 --optimizer adam --gamma 1.1 --lr_schedule constant --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --train
python eval.py --dataset ADNI_MMSE --data_root ../processed_data --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --split test
```

### ADNI_early

```bash
python train.py --dataset ADNI_early --data_root ../processed_data --epochs 50 --batch_size 16 --learning_rate 0.001 --optimizer adam --gamma 1.1 --lr_schedule constant --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --train
python eval.py --dataset ADNI_early --data_root ../processed_data --ckpt_path CAPPM_subject_split --use_conflict --use_healnet --split test
```

