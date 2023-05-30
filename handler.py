import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from healthinsurance.HealthInsurance import HealthInsurance

# loading model
model = pickle.load( open( 'model/model_linear_regression.pkl', 'rb' ) )

# initialize API
app = Flask( __name__ )

@app.route( '/healthinsurance/predict', methods = ['POST'] )
def health_insurance_predict():
    
    test_json = request.get_json()
    
    # Check if JSON has content
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique data
            test_raw = pd.DataFrame( test_json, index = [0] )
        else: # multiple data
            test_raw = pd.DataFrame( test_json, columns = test_json[0].keys() )
            
        # Instantiate HealthInsurance Class
        pipeline = HealthInsurance()
        
        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )
        
        # feature engeneering
        df2 = pipeline.feature_engineering( df1 )
        
        # data preparation
        df3 = pipeline.data_preparation( df2 )
        
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response
        
    else:
        return Response('{}', status = 200, mimetype = 'application/json')

if __name__ == '__main__':
    # o servidor Flask eh iniciado para escutar a API
    port = int(os.environ.get('PORT', 5000))
    app.run( host='0.0.0.0', port=port, debug=True )
