#!/bin/bash

# Declare an associative array where keys are model names and values are config file paths
declare -A model_configs
model_configs["ar2cl"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/ar2cl.yml"
model_configs["ar2pr"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/ar2pr.yml"
model_configs["ar2re"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/ar2re.yml"
model_configs["cl2ar"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/cl2ar.yml"
model_configs["cl2pr"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/cl2pr.yml"
model_configs["cl2re"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/cl2re.yml"
model_configs["pr2ar"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/pr2ar.yml"
model_configs["pr2cl"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/pr2cl.yml"
model_configs["pr2re"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/pr2re.yml"
model_configs["re2ar"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/re2ar.yml"
model_configs["re2cl"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/re2cl.yml"
model_configs["re2pr"]="/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/re2pr.yml"

# Array of calibration modes
calibration_modes=("PSAQ" "trainset" "valset" "gaussian")

# Data types for weights, activation, and attention
data_types_activation=("uint8" "uint4" "uint2")
data_types_attn=("uint8" "uint4" "uint2")
data_types_weight=("int8" "int4" "int2")

# Loop through each model
for model in "${!model_configs[@]}"
do
    # Retrieve the configuration file for the model
    config_file=${model_configs[$model]}

    # Loop through each calibration mode
    for calibration_mode in "${calibration_modes[@]}"
    do
        # Loop through each data type for weights, activation, and attention
        for data_type_activation in "${data_types_activation[@]}"
        do
            for data_type_attn in "${data_types_attn[@]}"
            do
                for data_type_weight in "${data_types_weight[@]}"
                do
                    # Execute the command with specified parameters
                    python test.py --model_name "$model" --config_file "$config_file" --calibration_mode "$calibration_mode" --weights "$data_type_weight" --activation "$data_type_activation" --attn "$data_type_attn"
                done
            done
        done
    done
done
