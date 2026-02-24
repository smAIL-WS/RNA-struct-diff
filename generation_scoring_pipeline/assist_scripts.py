#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:56:57 2024

@author: back1622
"""

import os
import numpy as np
def create_motif_directs(file,prefix,outputdir,result_dir):
    with open(file) as f:
        c_id = None
        
        for line in f:
            if line[0]==">":
                c_id = line[1:].strip()
                os.mkdir(result_dir+"/"+prefix+c_id)
            elif "." in line:
                with open(outputdir+"/"+prefix+c_id+".txt","w") as out:
                    out.write(line.split(" ")[0].strip())
                

        
files = []
subdirs = []
for subdir, dirs, f in os.walk("/home/back1622/compare_structure/snake/results/"):
    files.append(f)
    subdirs.append(subdir)
    

list_dict_scores={}

for directory,c_file in zip(subdirs[1:],files[1:]):
    seq_name = directory.split("/")[-1]
    list_dict_scores[seq_name]={}
    for file in c_file:
        temp = np.loadtxt(directory+"/"+file)
        a = file.split("_")[0]
        
        list_dict_scores[seq_name][a] = temp

def calculate_function_recursive(data, func):
    if isinstance(data, dict):
        # Recursively process each dictionary entry
        return {key: calculate_function_recursive(value, func) for key, value in data.items()}
    elif isinstance(data, (list, np.ndarray)):
        try:
            # Convert lists to numpy arrays for uniformity
            array_data = np.array(data)
            # Apply the function
            return func(array_data)
        except Exception as e:
            print(f"Error applying function: {e}")
            return None
    else:
        # Handle unexpected data types
        raise TypeError("Unexpected data type encountered. Expected dict, list, or numpy array.")

medians = calculate_function_recursive(list_dict_scores,np.median)

combinations = []
for i in range(15):
    combinations.append("rnd_seq"+str(i))
    
    
filtered_medians = {x:medians[x] for x in combinations}


targets = ["classical","random","linear","conv"]

res = []
for i in targets:
    tmp=[]
    for j in filtered_medians.keys():
        tmp.append(filtered_medians[j][i])
    res.append(tmp)
res=np.array(res)


def identity(x):
    return x


