import os
import pandas as pd
import joblib
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def get_predictions(request):
    try:
        # Step 1: Load Excel file
        file_path = 'predictions/ml/Reordered_Real_final_Data_With_anomaly.csv'  # Update this!
        df = pd.read_csv(file_path)

        # Step 2: Ensure all required columns exist
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

        # Step 3: Load Models
        model_path = 'predictions/ml/'
        model_uts = joblib.load(os.path.join(model_path, 'xgboost_model_output_   UTS.pkl'))
        model_cond = joblib.load(os.path.join(model_path, 'xgboost_model_output_Conductivity.pkl'))
        model_elong = joblib.load(os.path.join(model_path, 'xgboost_model_output_Elongation.pkl'))

        # Step 4: Predict
        df['UTS'] = model_uts.predict(X)
        df['Conductivity'] = model_cond.predict(X)
        df['Elongation'] = model_elong.predict(X)

        # Step 5: Return JSON list of rows
        result = df[expected_features + ['UTS', 'Conductivity', 'Elongation']].to_dict(orient='records')
        return Response(result)

    except Exception as e:
        return Response({"error": str(e)}, status=500)
