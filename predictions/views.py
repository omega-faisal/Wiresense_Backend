import os
import pandas as pd
import joblib
import traceback
import json
import torch
from predictions.ml.ThreeToSixteen import reverse_predict
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['POST'])
def predict_three_to_sixteen(request):
    try:
        # Ensure the request contains valid JSON data
        input_data = request.data

        # Convert the request data to JSON string
        json_input = json.dumps(input_data)

        # Call your reverse_predict function
        result = reverse_predict(json_input)

        # Return the result
        return Response(json.loads(result), status=200)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def get_predictions(request):
    try:
        # Load full training data
        full_data = pd.read_csv('predictions/ml/Final_Anomaly_Removed_Data.csv')
        full_data.columns = full_data.columns.str.strip()
        
        # Load inference data (can be same as above or new test input)
        file_path = 'predictions/ml/Final_Anomaly_Removed_Data.csv'
        df = pd.read_csv(file_path)

        expected_features = [
            'EMUL_OIL_L_TEMP_PV_VAL0', 'STAND_OIL_L_TEMP_PV_REAL_VAL0', 'GEAR_OIL_L_TEMP_PV_REAL_VAL0',
            'EMUL_OIL_L_PR_VAL0', 'QUENCH_CW_FLOW_EXIT_VAL0', 'CAST_WHEEL_RPM_VAL0', 'BAR_TEMP_VAL0',
            'QUENCH_CW_FLOW_ENTRY_VAL0', 'GEAR_OIL_L_PR_VAL0', 'STANDS_OIL_L_PR_VAL0',
            'TUNDISH_TEMP_VAL0', 'RM_MOTOR_COOL_WATER__VAL0', 'ROLL_MILL_AMPS_VAL0',
            'RM_COOL_WATER_FLOW_VAL0', 'EMULSION_LEVEL_ANALO_VAL0', '%SI', '%FE', '%TI', '%V', '%AL',
            'Furnace_Temperature'
        ]
        
        for col in expected_features:
            if col not in df.columns:
                df[col] = 1.90  

        X = df[expected_features]

        # Load models
        model_path = 'predictions/ml/'
        model_uts = joblib.load(os.path.join(model_path, 'xgboost_model_output_   UTS.pkl'))
        model_cond = joblib.load(os.path.join(model_path, 'xgboost_model_output_Conductivity.pkl'))
        model_elong = joblib.load(os.path.join(model_path, 'xgboost_model_output_Elongation.pkl'))

        # Make predictions (still normalized)
        df['UTS_norm'] = model_uts.predict(X)
        df['Conductivity_norm'] = model_cond.predict(X)
        df['Elongation_norm'] = model_elong.predict(X)

        # Calculate min-max for denormalization from training data
        min_uts, max_uts = full_data['UTS'].min(), full_data['UTS'].max()
        min_cond, max_cond = full_data['Conductivity'].min(), full_data['Conductivity'].max()
        min_elong, max_elong = full_data['Elongation'].min(), full_data['Elongation'].max()

        # Denormalize
        df['UTS'] = df['UTS_norm'] * (max_uts - min_uts) + min_uts
        df['Conductivity'] = df['Conductivity_norm'] * (max_cond - min_cond) + min_cond
        df['Elongation'] = df['Elongation_norm'] * (max_elong - min_elong) + min_elong

        # Drop normalized columns
        df.drop(columns=['UTS_norm', 'Conductivity_norm', 'Elongation_norm'], inplace=True)

        # Return JSON
        result = df[expected_features + ['UTS', 'Conductivity', 'Elongation']].to_dict(orient='records')
        return Response(result)

    except Exception as e:
     print("❌ Exception in get_predictions():", e)
     traceback.print_exc()  # <-- this gives the actual line where it failed
     return Response({"error": str(e)}, status=500)
