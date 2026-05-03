@echo off
setlocal enabledelayedexpansion

echo === Pipeline Configuration ===
set /p model_name=Model name (e.g. random_forest):
set /p strategy=Search strategy (e.g. random_search, grid_search):
set /p exp_number=Experiment number (e.g. 000001):

set registry_file=model_config\model_registry.json
set config_file=model_config\%model_name%\config_%strategy%_%model_name%_exp_%exp_number%.json
set model_output=models\%model_name%\%strategy%_exp_%exp_number%_%model_name%.pkl
set train_metrics_file=metrics\%model_name%\metrics_%strategy%_%model_name%_exp_%exp_number%_train.json
set test_metrics_file=metrics\%model_name%\metrics_%strategy%_%model_name%_exp_%exp_number%_test.json

set train_output_dir=output\%model_name%\%strategy%_exp_%exp_number%_train
set test_output_dir=output\%model_name%\%strategy%_exp_%exp_number%_test
set predicted_data_dir=output\predicted_data

echo.
echo Registry     : %registry_file%
echo Config       : %config_file%
echo Model        : %model_output%
echo Metrics      : metrics\%model_name%\metrics_%strategy%_%model_name%_exp_%exp_number%_[train^|test].json
echo Train output : %train_output_dir%
echo Test output  : %test_output_dir%
echo Predictions  : %predicted_data_dir%

:menu_loop
echo.
echo Pipeline CLI Menu (model: %model_name% ^| strategy: %strategy% ^| exp: %exp_number%):
echo 1. Train the Model
echo 2. Test on Available Validation Dataset
echo 3. Test on New/Random Dataset
echo 4. Exit
echo.
set /p choice=Enter your choice (1/2/3/4):

if "%choice%"=="1" goto train
if "%choice%"=="2" goto test_validation
if "%choice%"=="3" goto test_new
if "%choice%"=="4" goto exit_pipeline
echo Invalid choice. Please enter 1, 2, 3, or 4.
goto menu_loop

:train
echo Training the model...
if not exist "%train_output_dir%" mkdir "%train_output_dir%"
if not exist "metrics\%model_name%" mkdir "metrics\%model_name%"
if not exist "models\%model_name%" mkdir "models\%model_name%"
python train_model.py ^
    --train raw_csv/MS_1_Scenario_train.csv ^
    --model_output "%model_output%" ^
    --config "%config_file%" ^
    --registry "%registry_file%" ^
    --metrics "%train_metrics_file%" ^
    --output_dir "%train_output_dir%"
python generate_pdf.py ^
    --report "%train_output_dir%/training_report.pdf" ^
    --metrics "%train_metrics_file%"
goto menu_loop

:test_validation
echo Testing on available validation dataset...
if not exist "%test_output_dir%" mkdir "%test_output_dir%"
if not exist "%predicted_data_dir%" mkdir "%predicted_data_dir%"
if not exist "metrics\%model_name%" mkdir "metrics\%model_name%"
python predict_model.py ^
    --test raw_csv/MS_1_Scenario_test.csv ^
    --model "%model_output%" ^
    --config "%config_file%" ^
    --registry "%registry_file%" ^
    --metrics "%test_metrics_file%" ^
    --output_dir "%test_output_dir%"
python generate_pdf.py ^
    --report "%test_output_dir%/MS_1_Scenario_test_report.pdf" ^
    --metrics "%test_metrics_file%"
goto menu_loop

:test_new
echo Testing on all CSV files in the 'test CSV' directory...
if not exist "%test_output_dir%" mkdir "%test_output_dir%"
if not exist "%predicted_data_dir%" mkdir "%predicted_data_dir%"
if not exist "metrics\%model_name%" mkdir "metrics\%model_name%"
for %%f in ("test CSV\*.csv") do (
    echo Processing file: %%f
    set file_metrics=metrics\%model_name%\metrics_%strategy%_%model_name%_exp_%exp_number%_%%~nf_test.json
    python predict_model.py ^
        --test "%%f" ^
        --model "%model_output%" ^
        --config "%config_file%" ^
        --registry "%registry_file%" ^
        --metrics "!file_metrics!" ^
        --output_dir "%test_output_dir%"
    python generate_pdf.py ^
        --report "%test_output_dir%\%%~nf_report.pdf" ^
        --metrics "!file_metrics!"
)
goto menu_loop

:exit_pipeline
echo Exiting the pipeline...
endlocal
exit /b 0
