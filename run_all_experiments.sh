#!/bin/bash
# Master script to run all experiments in order.
# GPU needed for training and most experiments; falls back to CPU if no GPU.

set -e

DATASETS="Cora Citeseer PubMed"
DEVICE=""
SEED=42
NUM_EDGES=50

if ! nvidia-smi &>/dev/null; then
    echo "No GPU detected, using CPU"
    DEVICE="--device cpu"
fi

echo "============================================"
echo "Step 0: Train GCN models"
echo "============================================"
for ds in $DATASETS; do
    if [ ! -f "checkpoints/${ds}_gcn.pt" ]; then
        echo "Training on $ds..."
        python experiments/train_gcn.py --dataset $ds --epochs 100 $DEVICE --seed $SEED
    else
        echo "Checkpoint exists for $ds, skipping training"
    fi
done

echo ""
echo "============================================"
echo "Step 1: Run existing experiments"
echo "============================================"

echo "--- Explanation fidelity ---"
for ds in $DATASETS; do
    echo "Running explanations on $ds..."
    python experiments/run_explanations.py --dataset $ds --num_edges $NUM_EDGES $DEVICE --seed $SEED || echo "FAILED: explanations on $ds"
done

echo "--- Runtime scaling ---"
python experiments/run_runtime.py $DEVICE --seed $SEED || echo "FAILED: runtime"

echo "--- Coarsening ratio ablation ---"
python experiments/run_ablation.py $DEVICE --seed $SEED || echo "FAILED: ablation"

echo ""
echo "============================================"
echo "Step 2: Priority 1 - Comprehensive metrics"
echo "============================================"
for ds in $DATASETS; do
    echo "Running comprehensive metrics on $ds..."
    python experiments/run_explanations.py --dataset $ds --num_edges $NUM_EDGES $DEVICE --seed $SEED || echo "FAILED: comprehensive metrics on $ds"
done

echo ""
echo "============================================"
echo "Step 3: Priority 3 - Pareto curves"
echo "============================================"
for ds in $DATASETS; do
    echo "Running Pareto curves on $ds..."
    python experiments/run_pareto.py --dataset $ds --num_edges $NUM_EDGES $DEVICE --seed $SEED || echo "FAILED: pareto on $ds"
done

echo ""
echo "============================================"
echo "Step 4: Priority 4 - Oversmoothing depth sweep"
echo "============================================"
for ds in $DATASETS; do
    echo "Running oversmoothing sweep on $ds..."
    python experiments/run_oversmoothing.py --dataset $ds $DEVICE --seed $SEED || echo "FAILED: oversmoothing on $ds"
done

echo ""
echo "============================================"
echo "Step 5: Priority 5 - Oversquashing"
echo "============================================"
python experiments/run_oversquashing.py $DEVICE --seed $SEED || echo "FAILED: oversquashing"

echo ""
echo "============================================"
echo "Step 6: Priority 6 - Hyperparameter ablation"
echo "============================================"
for ds in Cora Citeseer; do
    echo "Running hyperparameter ablation on $ds..."
    python experiments/run_hyperparam_ablation.py --dataset $ds --num_edges 30 $DEVICE --seed $SEED || echo "FAILED: hyperparam on $ds"
done

echo ""
echo "============================================"
echo "Step 7: Priority 7 - Refinement ablation"
echo "============================================"
for ds in $DATASETS; do
    echo "Running refinement ablation on $ds..."
    python experiments/run_refinement_ablation.py --dataset $ds --num_edges 30 $DEVICE --seed $SEED || echo "FAILED: refinement on $ds"
done

echo ""
echo "============================================"
echo "Step 8: Priority 8 - Ground truth"
echo "============================================"
python experiments/run_ground_truth.py $DEVICE --seed $SEED || echo "FAILED: ground truth"

echo ""
echo "============================================"
echo "Step 9: Priority 9 - Profiling"
echo "============================================"
for ds in $DATASETS; do
    echo "Running profiling on $ds..."
    python experiments/run_profiling.py --dataset $ds --num_edges 30 $DEVICE --seed $SEED || echo "FAILED: profiling on $ds"
done

echo ""
echo "============================================"
echo "Step 10: Priority 10 - Multi-backbone"
echo "============================================"
for ds in Cora Citeseer; do
    echo "Running multi-backbone on $ds..."
    python experiments/run_multibackbone.py --dataset $ds --num_edges 30 $DEVICE --seed $SEED || echo "FAILED: multibackbone on $ds"
done

echo ""
echo "============================================"
echo "Converting JSON results to Markdown"
echo "============================================"
python experiments/convert_results_to_md.py || echo "FAILED: conversion"

echo ""
echo "All experiments completed!"
echo "Results in: results/"
echo "Figures in: figures/"
