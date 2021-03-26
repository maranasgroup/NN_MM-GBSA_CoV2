import numpy as np
import os
import pandas as pd
import json
import sys

f = open('inputs/6lzg_wt_seq.csv')
lines = f.readlines()
f.close()
wt = {}
for line in lines:
    if line.strip():
        res = line.strip()[:-1]
        aa = line.strip()[-1]
        wt[res] = aa
        
wt_dirs = []
for key in wt.keys():
    wt_dirs.append('{1}{0}{1}'.format(key.strip(),wt[key]))

WORK_DIR = './work'

DIR_TO_ANALYZE = WORK_DIR

dirs = os.listdir(DIR_TO_ANALYZE)
print('Found directories to analyze:')
print(dirs)

def process_one_dir(dirname='N439K',dirpath='./work/N439K',top_n=None):
    print(dirname,dirpath)
    try:
        files = os.listdir(dirpath)
        print(dirname)
        #for f in files:
            #print(f)
            #if 'score' in f and dirname+'.sc' in f:
                #scorefile = dirpath+'/'+f
                #print(scorefile)
        scorefile = dirpath+'/score_{0}.sc'.format(dirname)
        print(scorefile)
        df = pd.read_fwf(scorefile,header=1)
        #print('Read {0}/score_{1}.sc SUCCESS.'.format(dirpath,dirname))
        
        # This has the input structure as well, drop it
        df = df.drop([0]) #specific for this case
        
        # Now collect data
        avgs = {}
        stds = {}
        raw = {}
        for key in df.keys():
            values = df[key]
            if 'description'==key:
                continue
            elif 'SCORE: total_score'==key:
                # The first column is like this:- SCORE: -2133.663
                # split it to get the score
                temp = []
                for v in values:
                    temp.append(float(v.split()[-1]))
            else:
                temp = list(map(float,values))
            
            if top_n:
                temp = sorted(temp)[:top_n]
                
            avg = round(np.average(temp),2)
            std = round(np.std(temp),2)
            avgs[key] = avg
            stds[key] = std
            raw[key] = temp
        
        return avgs,stds,raw
    
    except:
        print('Score file not found: ', dirpath)
        return {},{},{}


def process_all(dirs,top_n=None):
    wt_scores = {}
    mut_scores = {}
    raw = {}
    for each in dirs:
        raws = {}
        try:
            res = each[1:-1]
            aa = each[-1]
        except ValueError:
            print('Fail',each)
            continue
        if each in wt_dirs:
            avgs,stds, raws = process_one_dir(each,'./'+DIR_TO_ANALYZE+'/'+each,top_n)
            wt_scores[res] = {'avg':avgs,'std':stds}
        else:
            avgs, stds, raws = process_one_dir(each,'./'+DIR_TO_ANALYZE+'/'+each,top_n)
            if res in mut_scores:
                mut_scores[res][aa] = {'avg':avgs, 'std': stds}
            else:
                mut_scores[res] = {aa : {'avg':avgs, 'std': stds}}
        raw[each] = raws
        
    return wt_scores, mut_scores, raw

wt, mut, raw = process_all(dirs)
f = open('wt_scores.json','w')
f.write(json.dumps(wt,indent=True))
f.close()

f = open('mut_scores.json','w')
f.write(json.dumps(mut,indent=True))
f.close()

print(raw.keys())

f = open('raw_scores.json','w')
f.write(json.dumps(raw,indent=True))
f.close()

