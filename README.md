# FedVTP
It is a source code library of Federated Learning-based Vehicles Trajectory Prediction: a method and benchmark(FedVTP). Other details are being uploaded.

## Dataset
The NGSIM and HighD datasets can be downloaded from the official website. To start the experiment, use the tools in the dataset folder for processing. 

## Running the experiments
`python system_trajectory/train.py -data NGSIM -m stgcn -go stgcn -algo FedAvg -nc 2 -ls 3 -jr 1 -lbs 128 -did 3 -ad 0 -gr 1000 -stg 3 -txp 5`
