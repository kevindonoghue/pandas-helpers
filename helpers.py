import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler


class Dataset:
    """
    After cleaning the data, a class to a store a pandas DataFrame (with metadata like which features are categorical)
    and a method prepare_data to perform train/test splitting, oversampling, and preprocessing.
    
    Instance Attributes:
        data: the underlying pandas DataFrame (not modified in this class)
        target: the name of the column of the independent variable
        target_is_categorical: True if the dependent variable is categorical
        feature_columns: a list of the column names that are not the target
        categorical_features: a list of the column names of categorical features
        numerical_features: a list of the column names of the numerical features
    
    Classes:
        Preprocessor: a class that contains the various encoders (OneHotEncoder, Scaler, etc) and well as methods to fit and transform
                      the output of get_train_val_test.
        
    Methods:
        get_train_val_test: Returns a dict containing a train/test or train/val/test split of the DataFrame, with optional oversampling.
                            The output of get_train_val_test has the same schema as self.data.
        prepare_data: returns a quadruple df, X, y, preprocessor consisting of a train/test or train/val/test split (df), preprocessed
                      version (X, y) of this split, and the Preprocessor object (preprocessor) that the processing.
        
    """
    def __init__(self, data, target, categorical_columns=None):
        """
        Args:
            data: the underlying pandas DataFrame
            target: the column name of the independent variable
            categorical_columns: a list of the columns of categorical variables
        """
        self.data = data
        self.target = target
        self.feature_columns = data.columns.difference([self.target])
        if not categorical_columns:
            self.categorical_features = []
        else:
            self.categorical_features = [x for x in categorical_columns if x in self.feature_columns]
        self.numerical_features = [x for x in self.feature_columns if x not in self.categorical_features]
        self.target_is_categorical = False if not categorical_columns else target in categorical_columns
        
    def get_train_val_test(self, features='all', proportions=(0.8, 0.2), oversample=False, random_state=None):
        """
        Args:
            features: either 'all' or a list of feature columns to restrict to
            proportions: a tuple of length 2 or 3 that sums to 1, giving either train/test proportions or train/val/test proportions
            oversample: True to imblearn.over_sampling.RandomOverSampler
            random_state: a random state passed to sklearn.model_selection.train_test_split
            
        Return:
            df: A dict with keys either ('train', 'test') or ('train', 'val', 'test') depending on the length of the tuple proportions.
        """
        assert len(proportions) in (2, 3) and sum(proportions)==1 and (self.target not in features if type(features)==list else True)
        
        if features == 'all':
            features = list(self.feature_columns)
            
        df = dict()
        if len(proportions) == 2:
            df['train'], df['test'] = train_test_split(self.data[features + [self.target]] ,
                                                       test_size=proportions[1],
                                                       random_state=random_state)
        else:
            df['train'], df_other = train_test_split(self.data[features + [self.target]],
                                                     test_size=1-proportions[0],
                                                     random_state=random_state)
            df['val'], df['test'] = train_test_split(df_other[features + [self.target]],
                                                     test_size=proportions[2]/(1-proportions[0]),
                                                     random_state=random_state)
        if oversample:
            sampler = RandomOverSampler(random_state=random_state)
            indices, _ = sampler.fit_sample(np.array(df['train'].index).reshape(-1, 1),
                                            df['train'][self.target].values)
            indices = pd.Index(indices.reshape(-1))
            df['train'] = df['train'].loc[indices]
            
        return df
        
    class Preprocessor:
        """
        A class that contains the sklearn estimators and methods needed to transform the data from the output of self.get_train_val_test
        a pair of pandas dataframes X, y that can be inputted into sklearn models.
        
        Instance Attributes:
            features: a list of feature names to restrict to
            target: column name of target
            target_is_categorical: True if the dependent variable is categorical
            categorical_features: a list of names of feature columns that are categorical
            numerical_features: a list of names of feature columns that are categorical
            use_pca: True if pca should be applied
            ohe: a sklearn OneHotEncoder to be applied to the categorical features
            label_encoder: a sklearn LabelEncoder to be applied to the dependent variable if it is categorical
            scaler: a sklearn StandardScaler to be applied to all features after one hot encoding the categorical features
            pca: a sklearn PCA to be applied to all features after scaler if use_pca==True
            preprocessed_features: a list of column names for the 'X' output of the preprocess_phase method
            fitted: True only after the fit method has been called
            
            
        Methods:
            fit: Takes as input a dict of a train/test or train/val/test of pandas DataFrames and initalizes all the sklearn estimators.
            preprocess: One hot encodes, normalizes, and applies pca to the features of its input DataFrame. Also encodes the target with a LabelEncoder.
            preprocess_dict: Takes as input a dict of a train/test or train/val/test of pandas DataFrames and applies preprocess_phase to each DataFrame
                             in the input.
        """
        def __init__(self, features, target, target_is_categorical=False, categorical_features=None, numerical_features=None, pca_components=None):
            """
            Args:
                features: a list of feature names to restrict to
                target: column name of target
                target_is_categorical: True if the dependent variable is categorical
                categorical_features: a list of names of feature columns that are categorical
                numerical_features: a list of names of feature columns that are categorical
                pca_components: n_components argument passed into the sklearn.decomposition.PCA object
            """
            self.features = features
            self.target = target
            self.target_is_categorical = target_is_categorical
            self.categorical_features = categorical_features if categorical_features else []
            self.numerical_features = numerical_features if numerical_features else []
            self.use_pca = pca_components != None

            self.ohe = OneHotEncoder()
            self.label_encoder = LabelEncoder()
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=pca_components)

            self.preprocessed_features = None
            self.fitted = False
            
        def fit(self, df):
            """
            Args:
                df: one of the pandas DataFrames outputted from get_test_val_train
            """
            df_target = df[self.target]
            df_cat = df[self.categorical_features]
            df_num = df[self.numerical_features]
            
            if self.target_is_categorical:
                self.label_encoder.fit(df_target)
                
            X = df_num.values

            if self.categorical_features:
                X = self.ohe.fit_transform(df_cat)
                X = np.concatenate([df_num.values, X], axis=1)
            X = self.scaler.fit(X)
            if self.use_pca:
                self.pca.fit(X)
                self.preprocessed_features = [f'pca_feature_{i}' for i in range(self.pca.num_components_)]
            elif self.categorical_features:
                self.preprocessed_features = list(self.numerical_features) + list(self.ohe.get_feature_names())
            else:
                self.preprocessed_features = list(self.numerical_features)
            
            self.fitted = True

        def preprocess(self, df):
            """
            Args:
                df: one of the pandas DataFrames outputted from get_test_val_train
            
            Return:
                If df_phase contains a target column, outputs a pair of pandas DataFrames X, y that can be inputted into sklearn models.
                Otherwise, it outputs the pandas DataFrame X.
            """
            assert self.fitted
            if self.target in df.columns and self.target_is_categorical:
                y = self.label_encoder.transform(df[self.target])
            elif self.target in df.columns:
                y = df[self.target]
            else:
                y = None

            df_cat = df[self.categorical_features]
            df_num = df[self.numerical_features]
            
            X = df_num.values

            if self.categorical_features:
                X_cat = self.ohe.transform(df_cat)
                X = np.concatenate([df_num.values, X_cat])
            X = self.scaler.transform(X)
            if self.use_pca:
                X = self.pca.transform(X)

            if y is not None:
                X = pd.DataFrame(X, index=df.index, columns=self.preprocessed_features)
                y = pd.Series(y, index=df.index)
                return X, y
            else:
                X = pd.DataFrame(X, index=df.index, columns=self.preprocessed_features)
                return X

        def preprocess_dict(self, dfs):
            """
            Args:
                dfs: a dict of pandas DataFrames as in the output of get_test_val_train
            
            Return:
                If all the dataFrames in df contains a target column, outputs a pair of dicts of pandas DataFrames X, y
                whose values can be inputted into sklearn models. Otherwise, it outputs just X.
            """
            assert self.fitted
            X = dict()
            y = dict()
            for phase in df:
                transformed = self.preprocess(df[phase])
                if type(transformed)==tuple:
                    X[phase], y[phase] = transformed
                else:
                    X[phase] = transformed

            if y:
                return X, y
            else:
                return X
            
    def prepare_data(self, features='all', proportions=(0.8, 0.2), oversample=False, pca_components=None, random_state=None):
        """
        Args:
            features: either 'all' or a list of feature names to restict to
            proportions: a tuple of floats that sums to 1 and is of length 2 or 3. Contains train/test or train/val/test proportions.
            oversample: True if to oversample minority classes using imblearn.over_sampling.RandomOverSampler
            pca_components: if not None, then apply pca and pass pca_components to n_components in sklearn.decomposition.PCA
            random_state: random_state passed to sklearn.model_selection.train_test_split
            
        Return:
            df: a dict of pandas DataFrames consisting of the train/test or train/val/test split of the original data, restricted to features
            X: a dict of pandas DataFrames consisting of the train/test or train/val/test split of the transformed features
            y: a dict of pandas DataFrames consisting of the train/test or train/val/test split of the target column
            preprocessor: the Preprocessor object that preprocessed the values of X and y. The method preprocessor.label_encoder.inverse_transform can
                          be called to decode labels of model predictions. And preprocess.preprocess can be used to preprocess new data beyond the
                          train/test or train/val/test split
        """
        df = self.get_train_val_test(features=features,
                                     proportions=proportions,
                                     oversample=oversample,
                                     random_state=random_state)
        preprocessor = self.Preprocessor(features,
                                         self.target,
                                         target_is_categorical=self.target_is_categorical,
                                         categorical_features=self.categorical_features,
                                         numerical_features=self.numerical_features,
                                         pca_components=pca_components)
        preprocessor.fit(df['train'])
        X, y = preprocessor.preprocess_dict(df)
        return df, X, y, preprocessor