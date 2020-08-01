import os,sys

cmd = "python bin/make_covsqrt.py v6.3.0_calibrated act_mr3 --season s13 --patch deep6 --array pa1 --overwrite --mask-version padded_v1 --nsims 3 --debug --delta-ell 200 --covsqrt-kind multipow"
#cmd = "python bin/make_covsqrt.py v6.3.0_calibrated act_mr3 --season s13 --patch deep6 --array pa1 --overwrite --mask-version padded_v1 --nsims 3 --debug --delta-ell 200"
os.system(cmd)

