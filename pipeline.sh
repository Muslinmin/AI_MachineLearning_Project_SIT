#!/bin/bash

# Display the CLI menu
while true; do
    echo -e "\nPipeline CLI Menu:"
    echo "1. Train the Model"
    echo "2. Test on Available Validation Dataset"
    echo "3. Test on New/Random Dataset"
    echo "4. Exit"

    read -p "Enter your choice (1/2/3/4): " choice

    if [[ "$choice" == "1" ]]; then
        echo "Training the model..."
        # Train the model
        python3 train_model.py --train MS_1_Scenario_train.csv --model_output random_forest_model.pkl --config model_config/random_forest/config_random_forest_exp_000001.json
        # Generate PDF report for training
        output_pdf="training_report.pdf"
        python3 generate_pdf.py --report "$output_pdf" --type "training"

    elif [[ "$choice" == "2" ]]; then
        echo "Testing on available validation dataset..."
        # Test on validation dataset
        python3 predict_model.py --test MS_1_Scenario_test.csv --model random_forest_model.pkl
        # Generate PDF report for validation test
        output_pdf="validation_test_report.pdf"
        python3 generate_pdf.py --report "$output_pdf" --type "validation"

    elif [[ "$choice" == "3" ]]; then
        echo "Testing on all CSV files in the 'test CSV' directory..."
        # Loop through all CSV files in the 'test CSV' directory
        for test_file in ./test\ CSV/*.csv; do
            echo "Processing file: $test_file"
            # Test on each file
            python3 predict_model.py --test "$test_file" --model random_forest_model.pkl
            # Generate PDF report for each test file
            output_pdf="${test_file%.csv}_report.pdf"
            python3 generate_pdf.py --report "$output_pdf" --type "random"
        done

    elif [[ "$choice" == "4" ]]; then
        echo "Exiting the pipeline..."
        exit 0
    else
        echo "Invalid choice. Please enter 1, 2, 3, or 4."
    fi
done
