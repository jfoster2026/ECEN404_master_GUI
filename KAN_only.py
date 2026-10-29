import kan, torch
import pandas as pd
import numpy as np

# kan is the pykan module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize KAN
model = kan.KAN(width=[4,8,8,1], grid=30, k=3, seed=1, grid_range= [0,1], device=str(device))

expmnt = [27271, 27272] # holds the experiment numbers 
expmnt_size = []

all_time = np.array([], dtype = "float32")
time_mx_mn = [6.5, 0.0]
all_PCD = np.array([], dtype = "float32")
PCD_mx_mn = [0.0, 0.0]
all_volt = np.array([], dtype = "float32")
volt_mx_mn = [15.0, -10.0]
all_bias = np.array([], dtype = "float32")
bias_mx_mn = [5.25, 0]
all_load = np.array([], dtype = "float32")
load_mx_mn = [6,1.6989]

train_ratio = 0.7 # this means 70% of the data will be used for training and the other 30% for testing during the fit() call

# expmnt was meant to be a list containing all of the experiment number to be used.
for exp_num in expmnt:
    if exp_num == 0: continue
    # collect data from dataset
    file_name = f"data_for_model\\LFXR_Lovejoy_Sept2021_{exp_num}.csv"
    df_train = pd.read_csv(file_name, usecols=[0,1,3])

    df_start_stop = pd.read_csv("start_stop_pts.csv")
    expmnt_row = exp_num - 27271   # the first experiment is on row index 1
    volt_start = int(df_start_stop['volt_start'].iloc[expmnt_row])
    volt_stop = int(df_start_stop['volt_stop'].iloc[expmnt_row])
    rad_start = int(df_start_stop['rad_start'].iloc[expmnt_row])
    rad_stop = int(df_start_stop['rad_stop'].iloc[expmnt_row])

    expmnt_size.append(volt_stop - volt_start)

    bias = float(df_start_stop['bias'].iloc[expmnt_row])
    load = float(df_start_stop['load'].iloc[expmnt_row])

    
    #print(volt_start, volt_stop, rad_start, rad_stop)

    time = df_train['time1'].iloc[volt_start:volt_stop].to_numpy(dtype = np.float32)
    # Make time in microseconds
    time = time * 1e6
    time = time - time.min() # shift time to start at 0


    PCD = df_train['PCD3_B'].iloc[rad_start:rad_stop].to_numpy(dtype = np.float32)

    # output voltage
    volt = df_train['Diode'].iloc[volt_start:volt_stop].to_numpy(dtype = np.float32) #.reshape(-1,1)
    #volt = volt / (bias if bias > 1 else 1)

    all_time = np.append(all_time, time)
    all_PCD = np.append(all_PCD, PCD)
    all_volt = np.append(all_volt, volt)
    all_bias = np.append(all_bias, np.full_like(time, bias))
    all_load = np.append(all_load, np.full_like(time, np.log10(load))) # use logarithmic scale for load to prevent NANs.



all_time = (all_time - time_mx_mn[1]) / (time_mx_mn[0] - time_mx_mn[1])

all_volt = (all_volt - volt_mx_mn[1]) / (volt_mx_mn[0] - volt_mx_mn[1])

PCD_mx_mn[0] = all_PCD.max()
PCD_mx_mn[1] = all_PCD.min()  
all_PCD = (all_PCD - PCD_mx_mn[1]) / (PCD_mx_mn[0] - PCD_mx_mn[1])

all_bias = (all_bias - bias_mx_mn[1]) / (bias_mx_mn[0] - bias_mx_mn[1])
all_load = (all_load - load_mx_mn[1]) / (load_mx_mn[0] - load_mx_mn[1]) 

# Make input matrix
X = np.column_stack((all_time, all_PCD, all_bias, all_load))  # shape (num_samples, 4)


#pull training points from the data
np.random.seed(None)
# need to get certain number of points from each experiment, so we don't just pull the first 70% of the data
slice_idx = 0
test_indices = np.array([], dtype=int)

for idx, i in enumerate(expmnt):
    if i == 0: continue
    slice_idx += expmnt_size[idx]
    exp_indices = np.arange(slice_idx - expmnt_size[idx], slice_idx)
    exp_test_indices = np.random.choice(exp_indices, 
                        round(expmnt_size[idx] * (1 - train_ratio)), replace=False)
    test_indices = np.concatenate((test_indices, exp_test_indices))

X_test = X[test_indices]
n_volt_test = all_volt[test_indices] # Correct Answers

#pull the rest for training
train_indices = np.setdiff1d(np.arange(X.shape[0]), test_indices)
X_train = X[train_indices]
n_volt_train = all_volt[train_indices]

dataset = {
    "train_input": torch.from_numpy(X_train).to(device),
    "train_label": torch.from_numpy(n_volt_train).reshape(-1,1).to(device),
    "test_input":  torch.from_numpy(X_test).to(device),
    "test_label":  torch.from_numpy(n_volt_test).reshape(-1,1).to(device),
}

print(f"Points for training: {X_train.shape}\nPoints for testing {X_test.shape}")


results = model.fit(dataset,
        opt="Adam", # Adam (faster) or LBFGS (slower)
        steps=300,
        lr=0.03, # was 0.016
        update_grid=False,  # set to false for single experiment training.
        start_grid_update_step= 100,
        stop_grid_update_step= 120,
        lamb=0.1,        # master scale — tune this first
        lamb_l1=0.0,      # high — you want to prune weak connections
        lamb_entropy=0.0, # high — encourages interpretable simple functions
        lamb_coef=0.0,    # usually redundant if l1 is active
        lamb_coefdiff=0.00) # smooth splines — important for noisy data)
