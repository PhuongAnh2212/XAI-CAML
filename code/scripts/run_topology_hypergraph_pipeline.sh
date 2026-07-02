#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

SOURCE_IMAGE="${1:-${SOURCE_IMAGE:-SOURCE.png}}"
TARGET_IMAGE="${2:-${TARGET_IMAGE:-TARGET.png}}"

echo "Topology-derived CAML hypergraph pipeline"
echo "source_image: ${SOURCE_IMAGE}"
echo "target_image: ${TARGET_IMAGE}"

python CL_Analysis/CL_codes_extract.py \
  --data_root data/Brain_Tumor2 \
  --checkpoint_path trained_models/CAML_brain_trained_model.pt \
  --output_csv CL_Analysis/results/testAB_CL_codes_extraction_results.csv \
  --cuda False

python CL_Analysis/topological_analysis.py \
  --latent_csv CL_Analysis/results/testAB_CL_codes_extraction_results.csv \
  --output_dir CL_Analysis/results

python CL_Analysis/topology_hypergraph_builder.py \
  --latent_csv CL_Analysis/results/testAB_CL_codes_extraction_results.csv \
  --output_dir CL_Analysis/results/topology_hypergraph \
  --lambda_reg 0.1 \
  --plot

python Case_Show/hypergraph_shortest_path.py \
  --hypergraph CL_Analysis/results/topology_hypergraph/hypergraph.json \
  --source_image "${SOURCE_IMAGE}" \
  --target_image "${TARGET_IMAGE}" \
  --output Case_Show/results/topology_hypergraph_shortest_path.csv
