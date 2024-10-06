#!/bin/bash
METHODS="kgcoop" #lp lp2 coop cocoop prograd clipAdapter tipAdapter-f

TEST_SOURCES="08_ODIR200x3" #02_MESSIDOR 13_FIVES 25_REFUGE 08_ODIR200x3

SHOTS="1" # 1 2 4 8 16

out_path="/export/livia/home/vision/Yhuang/FLAIR/results_time/"


for method in $METHODS

do

    for experiment in $TEST_SOURCES

    do

        for shot in ${SHOTS}

        do

            python main_transferability.py --experiment ${experiment} --method ${method} --load_weights True --shots_train ${shot} --shots_val ${shot} --shots_test 20% --project_features True --norm_features True --folds 1 --out_path ${out_path}

        done

    done

done