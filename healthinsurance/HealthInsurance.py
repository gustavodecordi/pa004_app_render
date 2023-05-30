import pickle
import numpy as np
import pandas as pd

class HealthInsurance( object ):
    
    def __init__( self ):
        self.home_path = ''
        self.age_scaler = pickle.load( open( self.home_path + 'parameters/age_scaler.pkl', 'rb' ) )
        self.annual_premium_scaler = pickle.load( open( self.home_path + 'parameters/annual_premium_scaler.pkl', 'rb' ) )
        self.vintage_scaler = pickle.load( open( self.home_path + 'parameters/vintage_scaler.pkl', 'rb' ) )
        self.gender_encoder = pickle.load( open( self.home_path + 'parameters/gender_encoder.pkl', 'rb' ) )
        self.region_encoder = pickle.load( open( self.home_path + 'parameters/region_encoder.pkl', 'rb' ) )
        self.policy_sales_channel_encoder = pickle.load( open( self.home_path + 'parameters/policy_sales_channel_encoder.pkl', 'rb' ) )
        
    def data_cleaning( self, df1 ):
        
        # novas colunas com letras minusculas ('lower_case') 
        cols_new = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 
                    'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage']

        df1.columns = cols_new
        
        return df1
    
    def feature_engineering( self, df2 ):
        # transformar 'vehicle_age' e 'vehicle_damage'
        # 'vehicle_age'
        df2['vehicle_age'] = df2['vehicle_age'].apply( lambda x: 'over_2_years'     if x == '> 2 Years' else
                                                                 'between_1_2_year' if x == '1-2 Year'  else
                                                                 'below_1_year'
                                                     )
        # 'vehicle_damage'
        df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0 )
        
        return df2
        
    def data_preparation( self, df5 ):
        df5['age'] = self.age_scaler.transform( df5[['age']].values )
        df5['annual_premium'] = self.annual_premium_scaler.transform( df5[['annual_premium']].values )
        df5['vintage'] = self.vintage_scaler.transform( df5[['vintage']].values )
        df5['gender'] = self.gender_encoder.transform( df5['gender'] )
        df5.loc[:, 'region_code'] = df5['region_code'].map( self.region_encoder )
        df5 = pd.get_dummies( df5, prefix = 'vehicle_age', columns = ['vehicle_age'], dtype=int )
        df5.loc[:, 'policy_sales_channel'] = df5['policy_sales_channel'].map( self.policy_sales_channel_encoder )
        
        df5 = df5.fillna( 0 )
        
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage', 'previously_insured', 'policy_sales_channel' ]
        
        return df5[ cols_selected ]
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred_prob = model.predict_proba( test_data )
        
        # join pred into the original data
        original_data['score'] = pred_prob[:,1].tolist()
        
        # retorna o conjunto de dados original com as previsoes adicionadas na forma de um objeto json
        return original_data.to_json( orient = 'records', date_format = 'iso' ) 
