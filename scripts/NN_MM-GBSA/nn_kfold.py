#!/usr/bin/env python

import sys, os, time
import numpy as np
import torch as T
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import r2_score

# Global variables.
device = "cpu" # ["cpu", "gpu"]
model_type = "reg" # ["reg", "class"]
model_obj = "gbsa" # ["gbsa", "rose"]
data_scale = "nonlog" # ["log", "nonlog", "ddg"]
if model_obj in ["gbsa"]:
  ninpt = 18
elif model_obj in ["rose"]:
  ninpt = 20
noupt = 1
kfold = 5
max_epochs = round(4*0.5*1000)
lrn_rate = 0.001 # Learning rate.
wt_decay = 0.005 # The weight_decay argument in Adam optimizer.
bat_size = 999999 # A big number, use all data in a single batch.
ep_log_interval = max_epochs/10

# Specify device to use.
if device in ["cpu"]:
  device = T.device("cpu")
elif device in ["gpu"]:
  device = T.device("cuda:0")
else:
  print("Unknown device, aborted.")
  exit()

# Define the threshold value for different types of model.
if model_type in ["class"]:
  thld = 0.5
  loss_obj = T.nn.BCELoss() # Binary Cross Entropy loss function for classification model construction.
else:
  if data_scale in ["nonlog"]:
    thld = 1.0
  else:
    thld = 0.0
  loss_obj = T.nn.MSELoss() # Mean Squared Error loss function for regression model construction.

# Get mutant-map information, for output purpose.
sufx = ".csv"
fn_mutMap = "./Data/" + "mutant-map" + sufx
dat_mutMap = np.loadtxt(fn_mutMap, delimiter=",", dtype=np.str)
mutMapDct = {int(x[0]):x[1] for x in dat_mutMap} # Contains information about included mutant name and its unique index.

# Get information about average values and standard deviation from the original dataset, and these information are the same across k-fold cross validation.
prfx = "./Data/" + model_obj + "_" + model_type
fn_avg = prfx + '_avg' + sufx 
avg_all = np.loadtxt(fn_avg, delimiter=",", dtype=np.float32)
fn_std = prfx + '_std' + sufx 
std_all = np.loadtxt(fn_std, delimiter=",", dtype=np.float32)
fn_med = prfx + '_med' + sufx 
med_all = np.loadtxt(fn_med, delimiter=",", dtype=np.float32)
#print("avg_all: %s" % avg_all)
#print("std_all: %s" % std_all)
#print("med_all: %s" % med_all)

# ---------------------------------------------------------

class Dataset(T.utils.data.Dataset):

  def __init__(self, src_file, num_rows=None):
    all_data = np.loadtxt(src_file, max_rows=num_rows, delimiter=",", skiprows=0, dtype=np.float)
    self.x_data = T.tensor(all_data[:,1:ninpt+1], dtype=T.float).to(device) # Change the indexes to include different columns
    self.y_data = T.tensor(all_data[:,-1], dtype=T.float).to(device)
    self.y_data = self.y_data.reshape(-1,1)
    self.label = T.tensor(all_data[:,0], dtype=T.float).to(device)
    self.label = self.label.reshape(-1,1)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    if T.is_tensor(idx):
      idx = idx.tolist()
    lbl = self.label[idx,:] # Name of each mutant.
    fea = self.x_data[idx,:] # Input features.
    tgt = self.y_data[idx,:] # Target experimental value (K_D,app ratio) for each mutant. 
    sample = { 'feature':fea, 'target':tgt, 'label':lbl}

    return sample

# ---------------------------------------------------------
# General function to evaluate the NN model.
# ---------------------------------------------------------
def accuracy(model, ds):

  ndat = len(ds)
  tgts, prds = [], []
  tgts_mcc, prds_mcc = [], []
  tgts_multi, prds_multi = [], []
  tgts_mcc_multi, prds_mcc_multi = [], []
  rst_multi = []
  dct_rst = {}
  mutMap = {}
  loss_val = 0.0

  for idat in range(ndat):
    lbl = ds[idat]['label']
    fea = ds[idat]['feature']
    tgt = ds[idat]['target']
    lbl = int(T.round(lbl))

    with T.no_grad():
      prd = model(fea)

    # Recover the original target value, which is scaled previous to have zero mean and unity variance.
    if model_type in ["reg"]:
      tgt *= std_all[-1]
      prd *= std_all[-1]
      tgt += avg_all[-1]
      prd += avg_all[-1]

    # Recover the value first to get the true averaged loss value.
    loss_val += loss_obj(prd, tgt).item()
    mutMap[lbl] = tgt.item()
    tgts.append(tgt.item()) 
    prds.append(prd.item()) 

    # Prepare 0/1 values for %VC and MCC calculation.
    if tgt < thld:
      tgts_mcc.append(0)
    else:
      tgts_mcc.append(1)

    if prd < thld:
      prds_mcc.append(0)
    else:
      prds_mcc.append(1)

    # Collect and assign mutants with the same name.
    if lbl in dct_rst:
      dct_rst[lbl][0] += 1
      dct_rst[lbl][1].append(prd.item())
    else:
      dct_rst[lbl] = [1,[prd.item()]]

  tgts = np.array(tgts)
  prds = np.array(prds) 
  rst = np.array(list(zip(tgts, prds)))
  tgts_mcc = np.array(tgts_mcc)
  prds_mcc = np.array(prds_mcc)

  acc = np.count_nonzero(tgts_mcc == prds_mcc) * 1.0/ndat # %VC
  mcc = matthews_corrcoef(tgts_mcc, prds_mcc) # Matthews Correlation Coefficient.
  mse = loss_val * 1.0/ndat
  r2  = r2_score(tgts, prds) # Coefficient of Determination.
  r   = np.corrcoef(tgts, prds)[0,1] # Pearson Correlation Coefficient.

  for lbl in dct_rst:
    rst_multi.append([mutMap[lbl], np.median(dct_rst[lbl][1]), lbl])
  ndat_multi = len(rst_multi)

  # Calculate the NN model performance using multiple predictions for each mutant.
  for x in rst_multi:
    tgts_multi.append(x[0])
    if x[0] < thld:
      tgts_mcc_multi.append(0)
    else:
      tgts_mcc_multi.append(1)
    prds_multi.append(x[1])
    if x[1] < thld:
      prds_mcc_multi.append(0)
    else:
      prds_mcc_multi.append(1)

  tgts_multi = np.array(tgts_multi)
  prds_multi = np.array(prds_multi)
  tgts_mcc_multi = np.array(tgts_mcc_multi)
  prds_mcc_multi = np.array(prds_mcc_multi)
  mse_multi = ((tgts_multi - prds_multi)**2).mean()
  acc_multi = np.count_nonzero(tgts_mcc_multi == prds_mcc_multi) * 1.0/ndat_multi
  mcc_multi = matthews_corrcoef(tgts_mcc_multi, prds_mcc_multi)

  rst_multi_arr = np.array(rst_multi)
  r2_multi = r2_score(rst_multi_arr[:,0], rst_multi_arr[:,1])
  r_multi  = np.corrcoef(rst_multi_arr[:,0], rst_multi_arr[:,1])[0,1]

  return rst, acc, mcc, mse, r2, r, rst_multi, acc_multi, mcc_multi, mse_multi, r2_multi, r_multi

# ---------------------------------------------------------

def blind_test(model, ds):

  ndat = len(ds)
  rst_multi = []
  dct_rst = {}
  lbls = []
  tgts = []
  prds = []

  for idat in range(ndat):
    lbl = ds[idat]['label']
    fea = ds[idat]['feature']
    tgt = ds[idat]['target']
    lbl = int(T.round(lbl))

    with T.no_grad():
      prd = model(fea)

    # Data recovery
    if model_type in ["reg"]:
      tgt *= std_all[-1]
      prd *= std_all[-1]
      tgt += avg_all[-1]
      prd += avg_all[-1]

    lbls.append(lbl)
    tgts.append(tgt.item()) 
    prds.append(prd.item()) 

    if lbl in dct_rst:
      dct_rst[lbl][0] += 1
      dct_rst[lbl][1].append(prd.item())
    else:
      dct_rst[lbl] = [1,[prd.item()]]

  lbls = np.array(lbls)
  prds = np.array(prds) 
  rst = np.array(list(zip(lbls, prds)))

  for lbl in dct_rst:
    rst_multi.append([lbl, np.median(dct_rst[lbl][1])])
  
  return rst, rst_multi

# ----------------------------------------------------------
# NN model construction.
# ----------------------------------------------------------
class Net(T.nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    s = ninpt
    n = 8
    self.nhid = 4

    # Define more hidden layers than needed, adjust self.nhid to use different number of hidden layers.
    self.hid1 = T.nn.Linear(s*1, s*n)
    self.hid2 = T.nn.Linear(s*n, s*n)
    self.hid3 = T.nn.Linear(s*n, s*n)
    self.hid4 = T.nn.Linear(s*n, s*n)
    self.hid5 = T.nn.Linear(s*n, s*n)
    self.hid6 = T.nn.Linear(s*n, s*n)
    self.hid7 = T.nn.Linear(s*n, s*n)
    self.hid8 = T.nn.Linear(s*n, s*n)
    self.hid9 = T.nn.Linear(s*n, s*n)
    self.oupt = T.nn.Linear(s*n, 1)

    self.hids = [self.hid1, self.hid2, self.hid3, self.hid4, self.hid5, self.hid6, self.hid7, self.hid8, self.hid9]
    self.layers = self.hids[:self.nhid] + [self.oupt]

    for layer in self.hids:
      T.nn.init.xavier_uniform_(layer.weight) 
      T.nn.init.zeros_(layer.bias)
    
    # Define proportion or neurons to dropout
    self.dropout_io = T.nn.Dropout(0.5) # Input and output layer retain more neurons than hidden layers do.
    self.dropout = T.nn.Dropout(0.75)
    self.relu = T.nn.ReLU()

  def forward(self, x):
    z = x

    for i in range(self.nhid):
      layer = self.hids[i]
      z = self.relu(layer(z))
      if i in [1, self.nhid-1]:
        z = self.dropout_io(z)
      else:
        z = self.dropout(z)

    if model_type in ["class"]:
      z = T.sigmoid(self.oupt(z))
    else:
      z = self.oupt(z) 

    return z

# ----------------------------------------------------------

def main():

  startTime = time.time()

  # Job modes: 
  # "1": Training; 
  # "2": Use existing NN model to predict on unknown mutants;
  mode = sys.argv[1]
    
  # 0. Get started
  net = Net().to(device)
  net = net.train()
  optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate, weight_decay=wt_decay, amsgrad=False)
  print("\nPreparing training")
  print("GBSA binding affinity fitting using PyTorch")
  print("Loss function: " + str(loss_obj))
  print("Optimizer: " + str(optimizer))
  print("Max epochs: " + str(max_epochs))

  if mode == '1':

    acc_bat = [[] for ifold in range(kfold)]
    acc_multi_bat = [[] for ifold in range(kfold)]
    mcc_bat = [[] for ifold in range(kfold)]
    mcc_multi_bat = [[] for ifold in range(kfold)]
    mse_bat = [[] for ifold in range(kfold)]
    mse_multi_bat = [[] for ifold in range(kfold)]
    rst_bat = [[] for ifold in range(kfold)]
    rst_multi_bat = [[] for ifold in range(kfold)]
    r2_bat  = [[] for ifold in range(kfold)]
    r2_multi_bat = [[] for ifold in range(kfold)]
    r_bat  = [[] for ifold in range(kfold)]
    r_multi_bat = [[] for ifold in range(kfold)]

    for ifold in range(kfold):

      # Specify seed if wants to reproduce data
      #T.manual_seed(1)
      #np.random.seed(1)

      net = Net().to(device)
      net = net.train()  # set training mode
      optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate, weight_decay=wt_decay, amsgrad=False)

      # 1. Create Dataset and DataLoader objects
      print("\n[ifold = %s/%s]" % (ifold+1, kfold))
      dstps = ["trn",  "tst", "all"] # Data set types.
      ndstp = len(dstps)
      fnMap = {} # File names of different datasets: "trn", "tst".
      dsMap = {} # Datasets for different types: "trn", "tst".
      #prfx = "./Data/" + model_obj + "_" + model_type
      #sufx = ".csv"
    
      for dstp in dstps:
        if dstp in ["all"]:
          fn = prfx + "_" + dstp + sufx # One k-fold cross-validation has only one set of all data.
        else:
          fn = prfx + "_" + dstp + "_" + str(ifold) + sufx
        ds = Dataset(fn)
        fnMap[dstp] = fn
        dsMap[dstp] = ds

      # 2. Train network
      # Train the model with training data when evaluating the model performance.
      trn_ldr = T.utils.data.DataLoader(dsMap["trn"], batch_size=bat_size, shuffle=True)
	  # Train the model with all data when predicting on unknown mutants.
      #trn_ldr = T.utils.data.DataLoader(dsMap["all"], batch_size=bat_size, shuffle=True)

      print("\nStart training.")
      for epoch in range(0, max_epochs):
        epoch_loss = 0.0
      
        for (batch_idx, batch) in enumerate(trn_ldr):
          fea = batch['feature']
          tgt = batch['target']
          prd = net(fea)
      
          loss_val = loss_obj(prd, tgt)
          epoch_loss += loss_val.item()
      
          optimizer.zero_grad()
          loss_val.backward()
          optimizer.step()
      
        if (epoch+1) % ep_log_interval == 0:
          epoch_pct = 100*(epoch+1)/max_epochs
          print("epoch%% = %3d%%   loss = %0.4f" % (epoch_pct, epoch_loss))

      print("Training done.")
    
      # 3. Save model
      print("\nSaving trained model state_dict\n")
      path = "./Model/" + model_obj + "_sd_model_" + str(ifold) + ".pth"
      T.save(net.state_dict(), path)
      
      # 4. Evaluate model
      net = net.eval()
      
      for itp in range(ndstp): 
        dstp = dstps[itp]
        ds = dsMap[dstp]
        rst, acc, mcc, mse, r2, r, rst_multi, acc_multi, mcc_multi, mse_multi, r2_multi, r_multi = accuracy(net, ds)
        if dstp in ["trn", "tst"]:
        #  print("[%s]: %%VC = %.2f%%, MCC = %.2f, MSE = %.2f, r2 = %.2f, r = %.2f" % \
        #(dstp, acc*100.0, mcc, mse, r2, r))
          print("[%s]: %%VC_multi = %.2f%%, MCC_multi = %.2f, MSE_multi = %.2f, r2_multi = %.2f, r_multi = %.2f" % \
        (dstp, acc_multi*100.0, mcc_multi, mse_multi, r2_multi, r_multi))
        fn_rst = 'cout.rslt_' + dstp + '_' + str(ifold) + '.csv'
        fn_rst_multi = 'cout.rslt_multi_' + dstp + '_' + str(ifold) + '.csv'
        np.savetxt(fn_rst, rst, delimiter=',')
        np.savetxt(fn_rst_multi, rst_multi, delimiter=',')
        if dstp in ["tst"]:
          rst_bat[ifold].append(rst[:])
          rst_multi_bat[ifold].append(rst_multi[:])
        acc_bat[ifold].append(acc*100.0)
        acc_multi_bat[ifold].append(acc_multi*100.0)
        mcc_bat[ifold].append(mcc)
        mcc_multi_bat[ifold].append(mcc_multi)
        mse_bat[ifold].append(mse)
        mse_multi_bat[ifold].append(mse_multi)
        r2_bat[ifold].append(r2)
        r_bat[ifold].append(r)
        r2_multi_bat[ifold].append(r2_multi)
        r_multi_bat[ifold].append(r_multi)
      
      print("\nEnd GBSA learning process\n-----")
    
    # Average over kfold results.
    acc_avg = np.mean(acc_bat, axis=0)
    acc_std = np.std(acc_bat, axis=0)
    acc_multi_avg = np.mean(acc_multi_bat, axis=0)
    acc_multi_std = np.std(acc_multi_bat, axis=0)
    mcc_avg = np.mean(mcc_bat, axis=0)
    mcc_std = np.std(mcc_bat, axis=0)
    mcc_multi_avg = np.mean(mcc_multi_bat, axis=0)
    mcc_multi_std = np.std(mcc_multi_bat, axis=0)
    mse_avg = np.mean(mse_bat, axis=0)
    mse_std = np.std(mse_bat, axis=0)
    mse_multi_avg = np.mean(mse_multi_bat, axis=0)
    mse_multi_std = np.std(mse_multi_bat, axis=0)
    r2_avg  = np.mean(r2_bat, axis=0)
    r2_std  = np.std(r2_bat, axis=0)
    r2_multi_avg = np.mean(r2_multi_bat, axis=0)
    r2_multi_std = np.std(r2_multi_bat, axis=0)
    r_avg  = np.mean(r_bat, axis=0)
    r_std  = np.std(r_bat, axis=0)
    r_multi_avg = np.mean(r_multi_bat, axis=0)
    r_multi_std = np.std(r_multi_bat, axis=0)
    
    print("\nFinal results (average over k-fold):\n")
    for itp in range(ndstp):
      dstp = dstps[itp]
      if dstp in ["trn","tst"]:
        #print("[%s]: %%VC = %.2f(%.2f)%%, MCC = %.2f(%.2f), MSE = %.2f(%.2f), r2 = %.2f(%.2f), r = %.2f(%.2f)" % \
        #  (dstp, acc_avg[itp], acc_std[itp], mcc_avg[itp], mcc_std[itp], mse_avg[itp], mse_std[itp], r2_avg[itp], r2_std[itp], r_avg[itp], r_std[itp]))
        print("[%s]: %%VC_multi = %.2f(%.2f)%%, MCC_multi = %.2f(%.2f), MSE_multi = %.2f(%.2f), r2_multi = %.2f(%.2f), r_multi = %.2f(%.2f)" % \
          (dstp, acc_multi_avg[itp], acc_multi_std[itp], mcc_multi_avg[itp], mcc_multi_std[itp], mse_multi_avg[itp], mse_multi_std[itp], r2_multi_avg[itp], r2_multi_std[itp], r_multi_avg[itp], r_multi_std[itp]))

    # Average using all kfold data
    rst_multi_kfold = []
    for x in rst_multi_bat: 
      rst_multi_kfold.extend(np.vstack(x))
    rst = np.array(rst_multi_kfold)
    
    ndat = len(rst)
    tgts = rst[:,0]
    prds = rst[:,1]
    tgts_mcc = []
    prds_mcc = []
    for i in range(ndat):
      tgt = rst[i][0]
      prd = rst[i][1]
      if tgt < thld:
        tgts_mcc.append(0)
      else:
        tgts_mcc.append(1)
    
      if prd < thld:
        prds_mcc.append(0)
      else:
        prds_mcc.append(1)
    
    tgts_mcc = np.array(tgts_mcc)
    prds_mcc = np.array(prds_mcc)
    
    acc = np.count_nonzero(tgts_mcc == prds_mcc) * 1.0/ndat
    mcc = matthews_corrcoef(tgts_mcc, prds_mcc) # Matthews correlation coefficient
    mse = ((tgts - prds)**2).mean()
    r2  = r2_score(tgts, prds)
    r   = np.corrcoef(tgts, prds)[0,1]

    print("\nFinal results (recalculate using multi results) with %s data points:\n" % (ndat))
    print("[%s]: %%VC_multi = %.2f%%, MCC_multi = %.2f, MSE_multi = %.2f, r2_multi = %.2f r_multi = %.2f" % \
      ("tst", acc*100.0, mcc, mse, r2, r))
    
    rst_final = [[str(x[0]),str(x[1]),str(int(x[2])),mutMapDct[int(x[2])]] for x in rst]
    fn_rst = "cout.rslt_multi_tst_recalc.csv"
    np.savetxt(fn_rst, rst_final, fmt="%s", delimiter=',')

  else: # Evaluate unknown mutants with existing model

    if mode == '2': # Evaluate new mutants
      dn_tst = "./Mater/cout.gbsa_blind.csv"
      fn_tst = "./Data/" + model_obj + "_bt.csv" # Blind test
      nshuffle = 1
      navg = 240
      usecols = list(range(ninpt+1))
      usecols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
      tmpDat = np.genfromtxt(dn_tst, delimiter=",", skip_header=1, usecols=usecols, dtype=None, encoding=None)
      mutMap = {}
      mutDct = {}
      tstDat = []
      # Assign mutant data
      for idat in range(len(tmpDat)):
        dat = list(tmpDat[idat])
        mut = dat[0]
        if mut[0].isalpha():
          mut = mut[1:]
        if mut not in mutMap:
          mutMap[mut] = len(mutMap)
        val_mut = 1.0 # Not used, just to conform to the data structure
        dat = [mutMap[mut]] + dat[1:] + [val_mut]
        if mut in mutDct:
          mutDct[mut].append(dat[:])
        else:
          mutDct[mut] = [dat[:]]

      mutKey = list(mutDct.keys())
      for i in range(len(mutKey)):
        mut = mutKey[i]
        mutDat = mutDct[mut]
        for j in range(nshuffle):
          np.random.shuffle(mutDat)
          for k in range(len(mutDat)//navg):
            l = navg*k
            weights = np.random.uniform(1.0, 1.0, navg)
            tmpDat = np.average(mutDat[l:l+navg], axis=0, weights=weights)
            tstDat.append(tmpDat[:])

      # Perform data scaling for unknown mutants.
      out_tst = np.array(tstDat)
      out_tst[:,1:-1] = (out_tst[:,1:-1] - avg_all[1:-1])/std_all[1:-1]
      out_tst[:,-1] = (out_tst[:,-1] - avg_all[-1])/std_all[-1]

      np.savetxt(fn_tst, out_tst[:,:], delimiter=",")
      ds_tst = Dataset(fn_tst)

      rst_kfold = {}
      # Evaluate unknown mutants.
      for ifold in range(kfold):
        path = "./Model/" + model_obj + "_sd_model_" + str(ifold) + ".pth"
        net = Net()
        net.load_state_dict(T.load(path))
        net = net.eval()
        mutMapRev = {v:k for k,v in mutMap.items()}
        rst, rst_multi = blind_test(net, ds_tst)
        for r in rst_multi:
          r[0] = mutMapRev[r[0]]
          if r[0] in rst_kfold:
            rst_kfold[r[0]].append(r[1])
          else:
            rst_kfold[r[0]] = [r[1]]

      print("\nResults for unknown mutants:\n")
      for k,v in rst_kfold.items():
        print("%16s: %.4f" %(k,np.mean(v)))

  # Calculate elapsed time.
  print("\n-----\nElapsed time: %.4f s" % (time.time() - startTime))

if __name__== "__main__":
  main()
