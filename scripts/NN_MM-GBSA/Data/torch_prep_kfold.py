#!/usr/bin/env python

# >>>>>>
# [DESCRIPTION]:
#
# [AUTHOR]: Chen Chen, Penn State Univ, 2021
# <<<<<<

import sys, os, copy, math, re, shutil, glob, itertools
import subprocess as sub, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

myself = 'torch_prep_kfold.py'
if len(sys.argv) == 1 or sys.argv[1] == '-h':
  print('[Usage]: %s scramble_frac scramble_col navg fn_dat1 size_dat1 [fn_dat2 size_dat2] ...' % (myself))
  print('[Example]: ./%s 0.0 0 3 ../Mater/cout.rose.csv' % (myself))
  print('[Example]: ./%s 0.0 0 240 ../Mater/rawdat_1.csv ../Mater/rawdat_2.csv ../Mater/rawdat_3.csv ../Mater/rawdat_4.csv' % (myself))
  exit()

# Parse args
scramble_frac = float(sys.argv[1]) # Fraction of data been scrambled
scramble_icol = int(sys.argv[2]) # Index of column to scrambl
navg = int(sys.argv[3]) # Number of entries to average over, e.g. 240, 120, 80, etc.
npre = 4 # Parameters defined in front of the data sets
nset = 1
kfold = 5
nshuffle = 1
ratio_data = 1.00 # Use this fraction of data for each mutant
model_type = "reg" # ["class", "reg"]
model_obj = "gbsa" # ["gbsa", "rose"]
data_scale = "nonlog" # ["log", "nonlog"]
prfx = model_obj + "_" + model_type
sufx = ".csv" 

nbat = int((len(sys.argv)-npre)/nset)
fnPool = []
for ibat in range(nbat):
  fn = sys.argv[nset*ibat+npre+0] # CSV file name.
  fnPool.append(fn)

if model_obj in ["gbsa"]:
  usecols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
elif model_obj in ["rose"]:
  usecols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

ninpt = len(usecols)

if model_type in ["class"]:
  thld = 0.5
else:
  if data_scale in ["nonlog"]:
    thld = 1.0
  else:
    thld = 0.0

# Prepare reference dict.
fn_ref = "../Mater/exp_data_all_RBD.csv"
dat_ref = np.genfromtxt(fn_ref, delimiter=",", skip_header=1, dtype=None, encoding=None)

if data_scale in ["nonlog"]: # {mutant:kd_ratio}
  ref = {x[0]:10**(float(x[1])) for x in dat_ref}
elif data_scale in ["log"]: # {mutant:log10(kd_ratio)}
  ref = {x[0]:float(x[1]) for x in dat_ref}

# Initialize output data
dat_all = []
mdsDct  = {} # Average MD results
dat_trn = [[] for _ in range(kfold)]
dat_val = [[] for _ in range(kfold)]
dat_tst = [[] for _ in range(kfold)]
mut_trn = [[] for _ in range(kfold)]
mut_val = [[] for _ in range(kfold)]
mut_tst = [[] for _ in range(kfold)]

# Loop over mutants
print("\nProcessing data ...\n")

posDct = {}
negDct = {}
mutMap = {} # {mut:idx}
for ibat in range(nbat): # Different files
  fn = fnPool[ibat]
  tmpDat = np.genfromtxt(fn, delimiter=",", skip_header=1, usecols=usecols, dtype=None, encoding=None)

  # Assign values for mutants, and separate positive and negative examples.
  for idat in range(len(tmpDat)):
    dat = list(tmpDat[idat])
    mut = dat[0]
    if mut[0].isalpha():
      mut = mut[1:]
    if mut not in mutMap:
      mutMap[mut] = len(mutMap)
    if mut in ref:
      val_mut = ref[mut]
      if val_mut >= thld:
        if model_type in ["class"]: 
          val_mut = 1 # Improving
        dat = [mutMap[mut]] + dat[1:] + [val_mut]
        if mut in posDct:
          posDct[mut].append(dat[:])
        else:
          posDct[mut] = [dat[:]]
      else:
        if model_type in ["class"]: 
          val_mut = 0 # Worsening
        dat = [mutMap[mut]] + dat[1:] + [val_mut]
        if mut in negDct:
          negDct[mut].append(dat[:])
        else:
          negDct[mut] = [dat[:]]
    else:
      print("Mutant %s not found, skipped." % (mut))
      continue

mutMapRev = [[str(v),k] for k,v in mutMap.items()] # {idx:mut}
#print("Mutant map:\n%s" % (mutMapRev))
fn_map = "mutant-map.csv"
np.savetxt(fn_map, mutMapRev, fmt="%s", delimiter=",")

# Choose part of the data as the input, checking if 48*4ns is necessary
npck = 0 # Change to 0 if all data will be used, currently interface and noninterface sites have different amount of data, 50 vs 30.
for mut in posDct:
  ndat = len(posDct[mut])
  np.random.shuffle(posDct[mut])
  posDct[mut] = posDct[mut][0:int(ndat*ratio_data)]
  if npck > 0:
    posDct[mut] = posDct[mut][0:npck]
for mut in negDct:
  ndat = len(negDct[mut])
  np.random.shuffle(negDct[mut])
  negDct[mut] = negDct[mut][0:int(ndat*ratio_data)]
  if npck > 0:
    negDct[mut] = negDct[mut][0:npck]

# Split mutants into kfold.
posKey = list(posDct.keys())
negKey = list(negDct.keys())
print("%s Positive mutants: %s" % (len(posKey), posKey))
print("%s Negative mutants: %s" % (len(negKey), negKey))
np.random.shuffle(posKey)
np.random.shuffle(negKey)
posKeySplit = np.array_split(posKey,kfold)
negKeySplit = np.array_split(negKey,kfold)

for ifold in range(kfold):
  pk = posKeySplit[ifold]
  nk = negKeySplit[ifold]
  tmp_pos = []
  tmp_neg = []

  # Handle positive keys in ith fold
  for i in range(len(pk)):
    mut = pk[i]
    mutDat = posDct[mut]
    mutid = mutMap[mut]
    for j in range(nshuffle):
      np.random.shuffle(mutDat)
      for k in range(len(mutDat)//navg):
        l = navg*k
        weights = np.random.uniform(1.0, 1.0, navg)
        tmpDat = np.average(mutDat[l:l+navg], axis=0, weights=weights)
        tmp_pos.append(tmpDat[:])
        if mut in mdsDct: 
          mdsDct[mut].append(tmpDat[1])
        else:
          mdsDct[mut] = [tmpDat[1]]

  # Handle negative keys in ith fold
  for i in range(len(nk)):
    mut = nk[i]
    mutDat = negDct[mut]
    mutid = mutMap[mut]
    for j in range(nshuffle):
      np.random.shuffle(mutDat)
      for k in range(len(mutDat)//navg):
        l = navg*k
        weights = np.random.uniform(1.0, 1.0, navg)
        tmpDat = np.average(mutDat[l:l+navg], axis=0, weights=weights)
        tmp_neg.append(tmpDat[:])
        if mut in mdsDct: 
          mdsDct[mut].append(tmpDat[1])
        else:
          mdsDct[mut] = [tmpDat[1]]

  tmp_pos = np.vstack(tmp_pos)
  tmp_neg = np.vstack(tmp_neg)
  tmp_all = np.vstack([tmp_pos, tmp_neg])
  np.random.shuffle(tmp_pos)
  np.random.shuffle(tmp_neg)
  dat_all.append(tmp_all[:])

  for jfold in range(kfold):
    if jfold == ifold:
      mut_tst[jfold].extend(pk)
      mut_tst[jfold].extend(nk)
      dat_tst[jfold].append(tmp_pos[:])
      dat_tst[jfold].append(tmp_neg[:])
    else:
      mut_trn[jfold].extend(pk)
      mut_trn[jfold].extend(nk)
      dat_trn[jfold].append(tmp_pos[:])
      dat_trn[jfold].append(tmp_neg[:])

# Collect max/min/avg/std information from dataset.
out_all = np.vstack(dat_all)
max_all = np.max((out_all), axis=0)
min_all = np.min((out_all), axis=0)
std_all = np.std((out_all), axis=0)
avg_all = np.mean((out_all), axis=0)
med_all = np.median((out_all), axis=0)
fn_avg = prfx + '_avg' + sufx
np.savetxt(fn_avg, avg_all, delimiter=",")
fn_std = prfx + '_std' + sufx
np.savetxt(fn_std, std_all, delimiter=",")
fn_med = prfx + '_med' + sufx
np.savetxt(fn_med, med_all, delimiter=",")

# Process data and savetxt
for ifold in range(kfold):
  print("\n-----\n\nInfo for ifold = %s files:\n" % (ifold))

  # Define vars for output
  fn_trn = prfx + '_trn_' + str(ifold) + sufx
  fn_tst = prfx + '_tst_' + str(ifold) + sufx
  sub.call('rm -rf %s %s' % (fn_trn, fn_tst), shell=True)

  # Merge trn/tst/all sets
  out_trn = np.vstack(dat_trn[ifold])
  out_tst = np.vstack(dat_tst[ifold])
  
  if model_type in ["class"]: 
    # Standardize to mean = 0 and std = 1. (x - mean)/std
    out_trn[:,1:-1] = (out_trn[:,1:-1] - avg_all[1:-1])/std_all[1:-1]
    out_tst[:,1:-1] = (out_tst[:,1:-1] - avg_all[1:-1])/std_all[1:-1]
    if ifold == 0:
      out_all[:,1:-1] = (out_all[:,1:-1] - avg_all[1:-1])/std_all[1:-1]
  else:
    # Standardize to mean = 0 and std = 1. (x - mean)/std
    out_trn[:,1:-1] = (out_trn[:,1:-1] - avg_all[1:-1])/std_all[1:-1]
    out_tst[:,1:-1] = (out_tst[:,1:-1] - avg_all[1:-1])/std_all[1:-1]
    out_trn[:,-1] = (out_trn[:,-1] - avg_all[-1])/std_all[-1]
    out_tst[:,-1] = (out_tst[:,-1] - avg_all[-1])/std_all[-1]
    if ifold == 0:
      out_all[:,1:-1] = (out_all[:,1:-1] - avg_all[1:-1])/std_all[1:-1]
      out_all[:,-1] = (out_all[:,-1] - avg_all[-1])/std_all[-1]
  
  # Print standard deviation
  print("\nSTD of data set is:")
  print(np.std(out_all, axis=0))

  # Scrambling data
  if scramble_frac > 0.0:
    print("Scrambled %s of data on column %s" % (scramble_frac,scramble_icol) )
    # Scramble training set
    np.random.shuffle(out_trn[:int(len(out_trn)*scramble_frac),scramble_icol])
    # Scramble test set
    np.random.shuffle(out_tst[:int(len(out_tst)*scramble_frac),scramble_icol])

  # Shuffle trn and tst set
  np.random.shuffle(out_trn)
  np.random.shuffle(out_tst)
  split_out = [len(out_trn), len(out_tst)]
  ratio_out = [float(x/sum(split_out)) for x in split_out]
  # Ratio for each class
  ntot = len(out_trn)
  npos = sum((out_trn[:,-1]*std_all[-1]+avg_all[-1])>thld)
  nneg = ntot - npos
  ratio_pos = float(npos/ntot)
  ratio_neg = 1.0 - ratio_pos
  print("# of pos/neg samples in trn set: %s, %s" % (int(npos), int(nneg)))
  print("Ratio of pos/neg samples in trn set: %s, %s" % (ratio_pos, ratio_neg))
  print("Ratio of trn/tst sets is: %s" % (ratio_out))
  print("Size of trn/tst sets:  %s, %s" % (len(out_trn), len(out_tst)))
  print("%s mutants in trn set: %s" % (len(mut_trn[ifold]), mut_trn[ifold]))
  print("%s mutants in tst set: %s" % (len(mut_tst[ifold]), mut_tst[ifold]))
  print("# of Common mutants in trn and tst set: %s" % (len(set(mut_trn[ifold]).intersection(set(mut_tst[ifold])))))
  
  # Write data
  np.savetxt(fn_trn, out_trn[:,:], delimiter=",")
  np.savetxt(fn_tst, out_tst[:,:], delimiter=",")
  
  if ifold == 0:
    fn_all = prfx + '_all' + sufx
    np.savetxt(fn_all, out_all, delimiter=",", fmt="%s")
  
