import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

def process_data():  
    bioprint_df = pd.read_excel('Original_dataset_1.xlsx') 
    bioprint_df = bioprint_df.drop(['Reference'], axis = 1)
    bioprint_df = bioprint_df.drop(['DOI'], axis = 1)
    #To  return the total number of NaN values in the entire DataFrame:
    bioprint_df.isna().sum()
    # Replacthe different lables of endothelials cells to one lable
    bioprint_df['Cell type'].replace('Endothelial cells ', 'Endothelial cells', inplace=True)
    bioprint_df['Cell type'].replace('Endothelial cells \x02', 'Endothelial cells', inplace=True)
    bioprint_df['Cell type'].replace('embryonic stem cells ', 'Embryonic stem cells ', inplace=True)
    bioprint_df['Cell type'].replace('Endothelial cells _x0002_', 'Endothelial cells', inplace=True)
    # Remove rows where Alginate and Gelatin both are 0
    bioprint_df = bioprint_df[~((bioprint_df['Alginate_Concentration(%w/v)'] == 0) & (bioprint_df['Gelatin_Concentration(%w/v)'] == 0))] 
    #Imputing with KNN
    imputer_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent') #imputing mode value into missing values for temperatures
    bioprint_df.loc[:,['Cartridge_Temperature(°C)','Bed_Temperature(°C)']] = imputer_mode.fit_transform(bioprint_df.loc[:,['Cartridge_Temperature(°C)','Bed_Temperature(°C)']])
    #Drop instances without cell viability values
    bioprint_df = bioprint_df[bioprint_df['Viability(%)'].notna()]
    #Drop nonprinting instances (instances were extrusion pressure is zero)
    bioprint_df = bioprint_df.drop(bioprint_df[bioprint_df['Printing_Pressure(kPa)'] == 0 ].index)
    #Imputing Values
    bioprint_df.isna().sum() #produces a list of each variable’s number of null values
    #Imputation of numerical/continuous values databases
    imputer_knn = KNNImputer(n_neighbors = 10, weights = "uniform") #imputing mode value into missing values
    bioprint_df.iloc[:,1:44] = imputer_knn.fit_transform(bioprint_df.iloc[:,1:44]) #used for cell viability dataset preprocessing 
    #randomly shuffle all the rows of the DataFrame and return them as a shuffled sample:
    # Set the random seed
    np.random.seed(42)
    # Shuffle the DataFrame
    bioprint_df = bioprint_df.sample(frac=1).reset_index(drop=True)
    y = bioprint_df['Viability(%)'].values
    x = bioprint_df.drop('Viability(%)', axis = 1)
    x_original=x.copy(deep=True)
    #One-Hot Encoding
    x = pd.get_dummies(x, columns=['Cell type'])
    x.to_csv (r'processed_x_shuffled.csv',header=True)
    # y.to_csv (r'y.csv',header=True)
    return x,y,x_original
