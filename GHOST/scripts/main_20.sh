#!/bin/sh
export PYTHONPATH= # set to your own GHOST path like /root/GHOST
python tools/main_track.py \
    --det_file yolox_dets.txt \
    --splits mot20_test \
    --config_path config/config_tracker_20.yaml \
    --det_conf 0.45 \
    --act 0.7 \
    --inact 0.8 \
    --inact_patience 50 \
    --combi sum_0.8 \
    --store_feats 0 \
    --new_track_conf 0.6 \
    --len_thresh 0 \
    --remove_unconfirmed 0 \
    --last_n_frames 10
