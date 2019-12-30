#!/usr/bin/env bash

mkdir -p out/

PROJECT=https://github.com/ParrotPrediction/pyalcs-experiments.git#mountaincar

echo "Starting experiments with MountainCar environment"
nohup mlflow run $PROJECT -P explore_trials=500000 > out/exp_1.out 2>&1 &
nohup mlflow run $PROJECT -P explore_trials=500000 -P decay=true > out/exp_2.out 2>&1 &
nohup mlflow run $PROJECT -P explore_trials=1000000 > out/exp_3.out 2>&1 &
nohup mlflow run $PROJECT -P explore_trials=1000000 -P decay=true > out/exp_4.out 2>&1 &

echo "Starting experiments with Energy MountainCar environment"
nohup mlflow run $PROJECT -P environment=EnergyMountainCar-v0 -P explore_trials=500000 > out/exp_5.out 2>&1 &
nohup mlflow run $PROJECT -P environment=EnergyMountainCar-v0 -P explore_trials=500000 -P decay=true > out/exp_6.out 2>&1 &
nohup mlflow run $PROJECT -P environment=EnergyMountainCar-v0 -P explore_trials=1000000 > out/exp_7.out 2>&1 &
nohup mlflow run $PROJECT -P environment=EnergyMountainCar-v0 -P explore_trials=1000000 -P decay=true > out/exp_8.out 2>&1 &
