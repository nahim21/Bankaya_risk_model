from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import category_encoders as ce
import warnings

# <<<< Function to create the pipeline for the model >>>>

def transform_data(features_TE,features_OHE,continuous_features):
     
     """
     This create a pipleine to transofor the dataset, you need to pass categorical and numerical values. 
        
    Parameters:
    - features_TE (Target Encoder): Variables that you want to encoce as a Target Encode strategy.
    - features_OHE (One Hot Encoder): Variables that you want to encoce as a OHE strategy.
    - continuous_features: Numerical variables. 
    
    Returns:
    A Pipeline object where you can pass fit or transoform.
    """
     
    # Create pipeline for continuous variables 
     continuous_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="mean")),
        ("scaler",RobustScaler(with_centering=True,with_scaling=True))
        ])
    # Create pipeline for continuous variables
     categorical_pipeline_TE = Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("target_encod",ce.TargetEncoder(features_TE))
                ])
    # Create pipeline for categorical variables
     categorical_pipeline_OHE = Pipeline([
        ("one_hot_encod",ce.OneHotEncoder(cols=features_OHE))
        #("imputer",SimpleImputer(strategy="most_frequent")),
        #("one_hot_encod",OneHotEncoder(sparse=False))
            ])

    # Merge all pipeline features to create the process 
     preproccessing_pipeline = ColumnTransformer([
        ("continous",continuous_pipeline,continuous_features),
        ("categorical_TE",categorical_pipeline_TE,features_TE),
        ("categorical_OHE",categorical_pipeline_OHE,features_OHE)
                ],
        remainder = "drop"
              )
    # Pipeline result. 
     pipeline = Pipeline(steps=([
        ("preprocess",preproccessing_pipeline)
            ]))
     return pipeline