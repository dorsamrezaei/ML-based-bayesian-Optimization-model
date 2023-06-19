from process_data import *
from bayes_opt import BayesianOptimization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import numpy as np
from keras.models import load_model


#### UTILS
# get params from a row of data
# row is zero-indexed
# assume data_x_row is form of x.loc[[highest_viability]]
def get_params_from_row(list_of_params,data_x_row):
    for p_name in list_of_params:
        # print("p_name: ",p_name)
        for col_name in list(data_x_row.columns):
            # print("c_name: ",col_name)
            if p_name == col_name:
                print(f"{col_name}: ",data_x_row.iloc[0][col_name])

# get a dict of params (as input to 'register parameters' function) from a row of data
def param_dict_from_row(data_x_row):
    res = {}
    for name_p,idx in param_indices_dict.items():
        res[name_p] = data_x_row[name_p]
    return res

### this is a list of parameters that we want to optimize
# the names should be the actual names of the columns
params_to_optimize = ["crosslinker(CaCl2)_Concentarion(mM)", "Physical_Crosslinking_time_(s)"]

# this estimator does not get used, because we load a model
def estimator():
    model = keras.Sequential()
    model.add(keras.layers.Dense(42, activation='tanh'))
   # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(120, activation='tanh', kernel_regularizer=regularizers.l2(0.01))) # used to be 0.05
    #model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='linear'))
    # Compile the model
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Instantiate the model as you please (we are not going to use this)
model2 = KerasRegressor(build_fn=estimator, epochs=10, batch_size=10, verbose=1)
# This is where you load the actual saved model into new variable.
model2.model = load_model('saved_best model_38 layers.h5')
# Now you can use this to predict on new data (without fitting model2, because it uses the older saved model)

# Process the data
x, y, z= process_data()
x_cols = list(x.columns)

# this dict is a mapping of param name to the column index in x
##This code is creating a dictionary called param_indices_dict that maps parameter names to their column indices in a dataset. 
##It does this by iterating over each parameter name in params_to_optimize and setting its initial value in the dictionary to None.
##It then uses a nested loop to iterate over each column in x_cols and each parameter name in params_to_optimize. If the column name matches the parameter name, 
##the corresponding index i is assigned to the dictionary entry with the key of the parameter name. This results in a dictionary where each parameter name
##is mapped to the index of the column in the dataset that contains its values.
param_indices_dict = {} 
for i,col in enumerate(x_cols):
    param_indices_dict[col] = i

##This code defines a function called get_params_from_row_by_idx that prints the values of parameters from a single row of data if that row is represented 
##as a NumPy array. The function iterates over the items in the param_indices_dict dictionary, which maps parameter names to their column indices in the data. 
##For each parameter, the corresponding index is retrieved from the dictionary and used to print the value of that parameter from the specified row of data.
## The output of the function includes the row index and parameter name for each value that is printed.

# print the parameters from the row if the row is a numpy array
def get_params_from_row_by_idx(data_x_row):
    for name_p,idx in param_indices_dict.items():
        print(f"row {idx} with name {name_p}: ",data_x_row[0,idx])

# get the parameters from the row if there are multiple rows in the numpy array (data_x_row)
##This code defines a function called get_params_from_row_by_idx that retrieves the values of parameters from a NumPy array containing multiple rows of data.
## The function takes two arguments: data_x_row, which represents the entire dataset, and i, which specifies the row index for which to retrieve parameter values.
def get_params_from_row_by_idx(data_x_row,i):
    res = {}
    for name_p,idx in param_indices_dict.items():
        res[name_p] = data_x_row[i,idx]
    return res

def get_params_filter_by_celltype(x_norm_p,i):
    res = {}
    for name_p,idx in param_indices_dict.items():
        if "Cell type" not in name_p:
            res[name_p] =  x_norm_p[i,idx]
    return res

x_np = x.to_numpy()
scaler_y = MinMaxScaler()
scaler_x = MinMaxScaler()
x_norm = scaler_x.fit_transform(x)
y_norm = scaler_y.fit_transform(y.reshape(-1, 1))
y_pred = model2.predict(x_norm)
print(f"y_pred: {y_pred[:5]}")
print(f"y_norm: {y_norm[:5]}")

# row with highest viability
##This code uses NumPy's argmax() function to find the index of the maximum value in a NumPy array called y_norm.
highest_viability = np.argmax(y_norm)
print("highest_viability: ",highest_viability)

#show the actual values of crosslinking time and concentration of the dataset
#print(f"Value of column crosslinker(CaCl2)_Concentarion(mM) in row {highest_viability}: {x.loc[highest_viability, 'crosslinker(CaCl2)_Concentarion(mM)']}")
#print(f"Value of column Physical_Crosslinking_time_(s) in row {highest_viability}: {x.loc[highest_viability, 'Physical_Crosslinking_time_(s)']}")

"""## 1. Suggest-Evaluate-Register Paradigm

Internally the `maximize` method is simply a wrapper around the methods `suggest`, `probe`, and `register`. If you need more control over your optimization loops the Suggest-Evaluate-Register paradigm should give you that extra flexibility.

For an example of running the `BayesianOptimization` in a distributed fashion (where the function being optimized is evaluated concurrently in different cores/machines/servers), checkout the `async_optimization.py` script in the examples folder.
"""
# Let's start by defining our function, bounds, and instanciating an optimization object.
def black_box_function(**params_vals): # this asterisks means that
    # we can pass any number of arguments to the function
    input_copy = np.copy(x_norm[highest_viability,:].reshape(1,-1))
    for p_name,val in params_vals.items():
        input_copy[0,param_indices_dict[p_name]] = val
    return model2.predict(input_copy)

#Defining ranges of paramters to prob (Dorsa)

#creating a list of categorical featurs
categorical_params = [col for col in x_cols if "Cell type" in col]
"""
#For categorical parametrs
#Define range for categorical params. As we work with MDA-MB-231 cells I only assign 1 to this cell line and assign 0 to others
# Define a dictionary with all categorical parameters having zero value
categorical_dict = {param: 0 for param in categorical_params}
# Update the "MDA" parameter to have a value of 1
categorical_dict['Cell type_MDA-MB'] = 1
"""
#Define range for Numerical featurs
# Create dictionary of maximum and minimum values for each column in x
max_dict = dict(zip(x_cols, x.max()))
min_dict = dict(zip(x_cols, x.min()))
# Get the row n from the dataset as a list of values for Desired_params, this n can be n=highest_viability = np.argmax(y_norm)
#n = 1
n=highest_viability
#Dfine a dictionary of the desired parameters 
desired_param = dict(zip(x.columns, x.iloc[n]))
# Define range of i
i = 0.001
Params_to_exclude=params_to_optimize+categorical_params 
param_range = {}
for param in x_cols:
    if param not in Params_to_exclude:
        print(desired_param[param])
        param_range[param] = (desired_param[param] - i, desired_param[param] + i)
# Scale parameter ranges using min and max values for each column in x
scaled_range = {}
for param, span in param_range.items():
    scaled_range[param] = ((span[0] - min_dict[param]) / (max_dict[param] - min_dict[param]),
                           (span[1] - min_dict[param]) / (max_dict[param] - min_dict[param]))

#Select rang for the params_to_optimize = ["crosslinker(CaCl2)_Concentarion(mM)", "Physical_Crosslinking_time_(s)"]
#range_param_to_optimize={"crosslinker(CaCl2)_Concentarion(mM)":(0,1), "Physical_Crosslinking_time_(s)":(0,1)}
range_param_to_optimize={"crosslinker(CaCl2)_Concentarion(mM)":(0.02,1), "Physical_Crosslinking_time_(s)":(0.02,1)}#exclude 0 values in scaling
# Initialize the Bayesian optimization with the scaled bounds

#combine dictionaries 
bounds = {}
bounds.update(scaled_range)
bounds.update(range_param_to_optimize)

"""Notice that the evaluation of the blackbox function will NOT be carried out by the optimizer object. We are simulating a situation where this function could be being executed in a different machine, maybe it is written in another language, or it could even be the result of a chemistry experiment. Whatever the case may be, you can take charge of it and as long as you don't invoke the `probe` or `maximize` methods directly, the optimizer object will ignore the blackbox function."""
optimizer = BayesianOptimization(
    f=None,
    pbounds= bounds, #{p:(0,1) for p in params_to_optimize}, # all bounds are 0 to 1 TODO
    verbose=2,
    random_state=1,
    allow_duplicate_points=True,
)
"""One extra ingredient we will need is an `UtilityFunction` instance. In case it is not clear why, take a look at the literature to understand better how this method works."""

from bayes_opt import UtilityFunction

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

# It doesn't make sense to register everyhting in the dataset because we are not optimizing all columns at once
'''
# register all parameter combinations from the dataset
for index, row in x.iterrows():
    if y_norm[index][0] < 1.0:
        param_dict = get_params_from_row_by_idx(x_norm,index)
        print(f"Registering {param_dict} from row {index}")
        print(f"y_norm[index]: {y_norm[index][0]}")
        optimizer.register(
        params=param_dict,
        target=y_norm[index][0],
        )
    else:
        print("Skipped one!")'''
       
# omit row of the n with highest viability because we want it to be unexplored before by BO
omit_row = highest_viability

# Register parameter combinations from the dataset
for index, row in enumerate(x_norm):
    # Omit the specified row
    if index != omit_row:
        continue
    
    # Check if columns "crosslinker(CaCl2)_Concentration(mM)" and "Physical_Crosslinking_time_(s)" have non-zero values
    # Only consider data with non-zero gelatin and alginate concentrations
    if row[x.columns.get_loc("crosslinker(CaCl2)_Concentarion(mM)")] != 0 and row[x.columns.get_loc("Physical_Crosslinking_time_(s)")] != 0 and y_norm[index][0] < 1.0:
        # Check if "Cell type_MDA-MB" column has a non-zero value
        if row[x.columns.get_loc("Cell type_MDA-MB")] != 0:
            # Check if any column has "Cell type" in its name
            #if not any("Cell type" in column for column in x.columns if column != "Cell type_MDA-MB"):
            param_dict = get_params_filter_by_celltype(x_norm, index)
            print(f"Registering {param_dict} from row {index}")
            print(f"y_norm[index]: {y_norm[index][0]}")
            optimizer.register(
            params=param_dict,
                   target=y_norm[index][0],
                )
        else:
                print("Skipped one!")
    else:
            print("Skipped one!")


"""The `suggest` method of our optimizer can be called at any time. What you get back is a suggestion for the next parameter combination the optimizer wants to probe.

Notice that while the optimizer hasn't observed any points, the suggestions will be random. However, they will stop being random and improve in quality the more points are observed.
"""

next_point_to_probe = optimizer.suggest(utility)
print("Next point to probe is:", next_point_to_probe)

"""You are now free to evaluate your function at the suggested point however/whenever you like."""

target = black_box_function(**next_point_to_probe)
print("Found the target value to be:", target)

"""Last thing left to do is to tell the optimizer what target value was observed."""

optimizer.register(
    params=next_point_to_probe,
    target=target,
)

"""### 1.1 The maximize loop

And that's it. By repeating the steps above you recreate the internals of the `maximize` method. This should give you all the flexibility you need to log progress, hault execution, perform concurrent evaluations, etc.
"""

for _ in range(50):
    next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)
    
    print(target, next_point)
# print the maximum
print(optimizer.max)
# this is for scaling back to the original transformation
input_copy = np.copy(x_norm[highest_viability,:].reshape(1,-1))
# take the highest-viability row and put the estimated optimal params in
for p in params_to_optimize:
    input_copy[0,param_indices_dict[p]] = optimizer.max["params"][p]

# print("high-viability row model prediction, with free params optimized: ",model2.predict(input_copy))

# print the optimal values
for p in params_to_optimize:
    optimal_p = scaler_x.inverse_transform(input_copy)[0,param_indices_dict[p]]
    print(f"optimal {p} = {optimal_p}")
print(f"final best predicted viability: {scaler_y.inverse_transform(optimizer.max['target'].reshape(1,-1))[0][0]}")
print(f"row {highest_viability}, highest viability: {y[highest_viability]}")

input_copy = np.copy(x_norm[highest_viability,:].reshape(1,-1))
# get_params_from_row_by_idx(scaler_x.inverse_transform(input_copy))
# for p_name,val in params_vals.items():
#     input_copy[0,param_indices_dict[p_name]] = val
print("high-viability NN model prediction: ",model2.predict(input_copy))

# get_params
print('True values of parameters:')
get_params_from_row(params_to_optimize,x.loc[[highest_viability]])
#print(f"row {highest_viability}: {x.loc[[highest_viability]].to_numpy()[0]}")

