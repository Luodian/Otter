#!/bin/bash

for i in {1..7}
do
  time python main.py --input_csv_path="./data/Problem_0${i}/input-Problem_0$i.csv" \
   --truth_csv_path="./data/Problem_0${i}/output-Problem_0${i}_correct.csv" \
   --output_csv_path="./data/Problem_0${i}/output.csv"
done