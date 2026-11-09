#!/bin/bash
# Table 6 runs.  usage (from code/analysis): ./run_sensitivity.sh NAME KY R TARGET
#   ./run_sensitivity.sh y_fault          2.5 10 0
#   ./run_sensitivity.sh y_fault_R5       2.5  5 0
#   ./run_sensitivity.sh y_fault_R20      2.5 20 0
#   ./run_sensitivity.sh y_fault_margin20 2.5 10 4.24e-11
N=$1; export KY=$2 RREACT=$3 TARGET=$4; HERE=$(cd "$(dirname "$0")" && pwd)
export OUTDIR=$HERE/../sensitivity_v12/$N; mkdir -p $OUTDIR; cd $HERE/../audit
python3 search.py lit > $OUTDIR/search.log 2>&1 && python3 validate_fronts.py lit > $OUTDIR/validate.log 2>&1
