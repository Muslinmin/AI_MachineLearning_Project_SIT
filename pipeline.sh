#!/bin/bash

# Prompt for model and experiment details at startup
echo "=== Pipeline Configuration ==="
read -p "Model name (e.g. random_forest): " model_name
read -p "Experiment number (e.g. 000001): " exp_number

config_file="model_config/${model_name}/config_${model_name}_exp_${exp_number}.json"
train_metrics_file="metrics/${model_name}/metrics_${model_name}_exp_${exp_number}_train.json"
test_metrics_file="metrics/${model_name}/metrics_${model_name}_exp_${exp_number}_test.json"

echo -e "\nConfig  : $config_file"
echo "Metrics : metrics/${model_name}/metrics_${model_name}_exp_${exp_number}_[train|test].json"

# Display the CLI menu
while true; do
    echo -e "\nPipeline CLI Menu (model: ${model_name} | exp: ${exp_number}):"
    echo "1. Train the Model"
    echo "2. Test on Available Validation Dataset"
    echo "3. Test on New/Random Dataset"
    echo "4. Exit"

    read -p "Enter your choice (1/2/3/4): " choice

    if [[ "$choice" == "1" ]]; then
        echo "Training the model..."
        python3 train_model.py \
            --train MS_1_Scenario_train.csv \
            --model_output random_forest_model.pkl \
            --config "$config_file" \
            --metrics "$train_metrics_file"
        # Generate PDF report for training
        output_pdf="training_report.pdf"
        python3 generate_pdf.py --report "$output_pdf" --metrics "$train_metrics_file"

    elif [[ "$choice" == "2" ]]; then
        echo "Testing on available validation dataset..."
        python3 predict_model.py \
            --test MS_1_Scenario_test.csv \
            --model random_forest_model.pkl \
            --config "$config_file" \
            --metrics "$test_metrics_file"
        # Generate PDF report for validation test
        output_pdf="validation_test_report.pdf"
        python3 generate_pdf.py --report "$output_pdf" --metrics "$test_metrics_file"

    elif [[ "$choice" == "3" ]]; then
        echo "Testing on all CSV files in the 'test CSV' directory..."
        # Loop through all CSV files in the 'test CSV' directory
        for test_file in ./test\ CSV/*.csv; do
            echo "Processing file: $test_file"
            base_name=$(basename "${test_file%.csv}")
            file_metrics="metrics/${model_name}/metrics_${model_name}_exp_${exp_number}_${base_name}_test.json"
            python3 predict_model.py \
                --test "$test_file" \
                --model random_forest_model.pkl \
                --config "$config_file" \
                --metrics "$file_metrics"
            # Generate PDF report for each test file
            output_pdf="${test_file%.csv}_report.pdf"
            python3 generate_pdf.py --report "$output_pdf" --metrics "$file_metrics"
        done

    elif [[ "$choice" == "4" ]]; then
        echo "Exiting the pipeline..."
        exit 0
    else
        echo "Invalid choice. Please enter 1, 2, 3, or 4."
    fi
done
