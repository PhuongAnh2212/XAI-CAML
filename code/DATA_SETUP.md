# CAML Data Setup

This repository does not include medical image data. Prepare your own normal and abnormal image folders, then convert them to the CAML layout.

## Required Layout

```text
data/Brain_Tumor2/
  trainA_img/                  # normal training images
  trainB_img/                  # abnormal training images
  testA_img/                   # normal test images
  testB_img/                   # abnormal test images
  trainAB_img-name_label.txt   # image_name label
  testAB_img-name_label.txt    # image_name label
```

Labels use `0` for normal and any non-zero integer for abnormal.

## Prepare From Raw Images

```bash
python scripts/prepare_dataset.py \
  --normal_dir /path/to/normal_images \
  --abnormal_dir /path/to/abnormal_images \
  --output_dir data/Brain_Tumor2 \
  --test_ratio 0.2
```

For a tiny smoke-test dataset:

```bash
python scripts/prepare_dataset.py \
  --normal_dir /path/to/normal_images \
  --abnormal_dir /path/to/abnormal_images \
  --output_dir data/Brain_Tumor2_smoke \
  --test_ratio 0.2 \
  --limit_per_class 10
```

## Train

```bash
python CAML_Train/main_train.py \
  --data_root data/Brain_Tumor2 \
  --output_dir results/train \
  --cuda False
```

Smoke test:

```bash
python CAML_Train/main_train.py \
  --data_root data/Brain_Tumor2_smoke \
  --output_dir results/smoke_train \
  --max_iter 5 \
  --cuda False
```

## Extract Latent CSV

```bash
python CL_Analysis/CL_codes_extract.py \
  --data_root data/Brain_Tumor2 \
  --checkpoint_path trained_models/CAML_brain_trained_model.pt \
  --output_csv CL_Analysis/results/testAB_CL_codes_extraction_results.csv \
  --cuda False
```

## Baseline TDA Mapper

```bash
python CL_Analysis/topological_analysis.py \
  --latent_csv CL_Analysis/results/testAB_CL_codes_extraction_results.csv \
  --output_dir CL_Analysis/results
```

## Hypergraph Pipeline

```bash
python CL_Analysis/hypergraph_builder.py \
  --latent_csv CL_Analysis/results/testAB_CL_codes_extraction_results.csv \
  --method knn \
  --k 10 \
  --output_dir CL_Analysis/results/hypergraph
```

```bash
python Case_Show/hypergraph_shortest_path.py \
  --hypergraph CL_Analysis/results/hypergraph/hypergraph.json \
  --source_image SOURCE.png \
  --target_image TARGET.png \
  --output Case_Show/results/hypergraph_shortest_path.csv
```
