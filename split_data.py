#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import pandas as pd
import sklearn.model_selection

import time
import math
import numpy as np

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))
    return basename

def get_data(filename):
    """
    Assumes column 0 is the instance index stored in the
    csv file.  If no such column exists, remove the
    index_col=0 parameter.
    """
    data = pd.read_csv(filename, index_col=0)
    return data

def split_data(data, train_name, test_name, ratio, seed,verify_name):
    data_train_unsplit, data_test = sklearn.model_selection.train_test_split(data, test_size=ratio, random_state=seed)
    data_train, data_train_test = sklearn.model_selection.train_test_split(data_train_unsplit, test_size=ratio, random_state=seed)
    # data_train.to_csv(train_name,index=False,na_rep='NA')
    # data_train_test.to_csv(verify_name,index=False,na_rep='NA')
    # data_test.to_csv(test_name,index=False,na_rep='NA')
    # data_train.to_csv(train_name,na_rep='NA')
    
    # data_train_test.to_csv(verify_name,na_rep='NA')
    # data_test.to_csv(test_name,na_rep='NA')
    data_train.to_csv(train_name)
    
    data_train_test.to_csv(verify_name)
    data_test.to_csv(test_name)

    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Split Data into Training/Testing Sets')
    parser.add_argument('action', default='all',
                        choices=[ "split", "all" ], 
                        nargs='?', help="desired action")
    parser.add_argument('--data-file',     '-d', default="",    type=str,   help="csv file of data to split")
    parser.add_argument('--test-ratio',    '-r', default=0.2,   type=float, help="fraction of data to use as test data")
    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file to save training data (default is constructed from input file name)")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file to save test data (default is constructed from input file name)")
    parser.add_argument('--verification-file',     '-v', default="",    type=str,   help="name of file to save verification data (default is constructed from input file name)")
    
    parser.add_argument('--random-seed',             '-R', default=314159265,type=int,help="random number seed (-1 to use OS entropy)")

    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    filename =my_args.data_file
    print(filename)
    filename =fix_Data_rewrite(filename)
    print(filename)
    my_args.data_file =filename
    print(filename,my_args.data_file)

    # fix_Data(filename)
    
    return my_args

def fix_Data(filename):
    # data = get_data(filename)
    data = pd.read_csv(filename)
    data2=data
    # print(data)
    location=0
    for i in data:
        changes=False
        print("working on ",i)
        lines=[]
        halfway = len(data[i])/2

        # print (i)
        # print(data[i])
        if i=='name':
            name=i
        elif i=='cr':
            # location=0
            for j in range(len(data[i])):
                # print(j) 
                # print(data[name][location]," has a CR of ",j)
                if (j==None):
                   print(data[name][j]," is missing its CR")
                elif (j=='1/8'):
                    # print("Altering ",j, "from 1/8 to 0")
                    changes=True
                    lines_changed='0'
                    lines.append(lines_changed)

                    # data2.set_index([i,j])=str(0)

                elif (j=='1/4'):
                    # print("Altering ",j, "from 1/4 to 1")
                    changes=True

                    lines_changed='1'
                    lines.append(lines_changed)

                    # data2.set_index([i,j])=str(1)

                elif (j=='1/2'):
                    changes=True

                    # print("Altering ",j, "from 1/2 to 2")
                    # data2.set_index([i,j])=str(2)
                    lines_changed='3'
                    lines.append(lines_changed)
                # elif (float(j)==float('nan')):
                    # print(data[i]," is showing as NaN ",j)
                else:
                    changes=True

                    buffer=j
                    # print("buffeirng ",buffer)
                    lines_changed= int(buffer)
                    lines_changed+=3
                    # data2.set_index([i,j])=str(j_modified)

                    lines.append(str(lines_changed))
                # location+=1
        
        elif i!='url':
            # if i=='speed' or i=='align' or i=='legendary':
                    print ("feature column",i)
                    for j in data[i]:
                        check=isinstance(j, float)
                        if (check):
                            print("line ",j, " in ", i, " has registered as a float.")

                            check2=np. isnan(j)
                            if (check2):
                                print('showing up as a float of not nan, but as ',j)
                                changes=True
                                lines.append(int(lines_changed))
                                changes=True
                            else:
                                if i=='speed' or i=='align' or i=='legendary':
                                
                                # print('showing up as a float')
                                    lines_changed=None
                                    changes=True
                                    lines.append(lines_changed)
                                    changes=True
                                else:
                                    lines_changed=''
                                    changes=True
                                    lines.append(lines_changed)
                                    changes=True



                            # feature_column[count]=lines
                        else:
                            lines.append(lines)
            # location_2=0
            # last_datatype='Unknown'
            # data.fillna('')
            # numeric_cols = data.select_dtypes(include=np.float16).columns
            # # Convert to numeric, coercing errors
            # for col in numeric_cols:
            #     data[col] = pd.to_numeric(data[col], errors='coerce')
            #     # Fill NaN values with 0 (optional)
            # data.fillna(0, inplace=True)
            # # Convert numeric columns to integers
            # for col in numeric_cols:
            #     data[col] = data[col].astype(int)
        if changes:
            data[i]=lines

    data2.to_csv("fixed"+filename)

def fix_Data_rewrite(filename):
    data = pd.read_csv(filename)
    print("nulls in data=\n",data.isna().sum(),"end nulls")

    data
    null=data["legendary"][0]
    print(null)
    # print(data[i].dtype)
    # print(data[i].unique())
    # print("null is ",null)
    # time.sleep(10)
    for i in data.columns:
        # print("column:",i)
        if i in ['Legendary','legendary']:
            
            data["legendary_missing"]=data[i].isnull().astype(int)

            # print(null)
            # print(data[i].dtype)
            # print(data[i].unique())
            # print('hit legendary',data[i])
            coldata=data[i]
            # before=data[i].isna().sum()
            # if before!=0:
            #     print('')
                # print("before",before)
            # data[i].replace(0,null)
            # coldata2=coldata.replace(0,np.nan)
            coldata2=coldata.apply(lambda x:1 if x=='Legendary'else 0)

            # coldata2=coldata.replace(0,null)
            # print("coldata",coldata,"\ncol2",coldata2)
            data[i]=coldata2
            # print('saved data[i]',data[i])
            # data[i].replace(1,"Legendary")
            if before!=0:
                print("after:",data[i].isna().sum())
        elif i in ['cr']:
            lines=[]
            for j in data[i]:
                if (j=='1/8'):
                    # print("Altering ",j, "from 1/8 to 0")
                    changes=True
                    lines_changed='0'
                    lines.append(lines_changed)

                    # data2.set_index([i,j])=str(0)

                elif (j=='1/4'):
                    # print("Altering ",j, "from 1/4 to 1")
                    changes=True

                    lines_changed='1'
                    lines.append(lines_changed)

                    # data2.set_index([i,j])=str(1)

                elif (j=='1/2'):
                    changes=True

                    # print("Altering ",j, "from 1/2 to 2")
                    # data2.set_index([i,j])=str(2)
                    lines_changed='2'
                    lines.append(lines_changed)
                # elif (float(j)==float('nan')):
                    # print(data[i]," is showing as NaN ",j)
                else:
                    changes=True

                    buffer=j
                    # print("buffeirng ",buffer)
                    lines_changed= int(buffer)
                    lines_changed+=3
                    # data2.set_index([i,j])=str(j_modified)

                    lines.append(str(lines_changed))
            data[i]=lines
                

        elif i not in ['hahaha']:
            print (i)
            d_data=data[i].dtype
            # print("Col",i," has a datatype of ",d_data)
            if d_data=='object':
                data_modified=pd.to_numeric(data[i],errors='coerce')
                if data_modified.isnull().sum()>0:
                    # print("Col",i," has a datatype of ",d_data)
                    before=data[i].isna().sum()
                    coldata=data[i]

                    if before!=0:
                        print("before",before)
                    # coldata2=coldata.replace('',np.nan)
                    coldata2=coldata.fillna('')


                    # data[i]=data[i].replace('',np.nan)
                    # data[i]=data[i].replace(1,"Legendary")
                    if before!=0:
                        print("after",data[i].isna().sum())
                    data[i]=coldata2           

                else:
                    data[i]=data_modified
 
            else:
                check=isinstance(data[i][0], int)
                check2=isinstance(data[i][0],float)
                if check or check2:
                    coldata=data[i]
                    data[i+'_missing']=coldata.isnull().astype(int)
                    coldata2=coldata.fillna(-1)
                    data[i]=coldata2
                    # coldata=data[i]
                    # coldata3=coldata.astype("Int64")
                    # coldata2=coldata.astype("Int64")

                    # coldata2=coldata3.fillna('')
                    # data[i]=coldata2
                    # print("column",i,"set to be int 64")
        end=data[i].isna().sum()
        if end!=0 and i not in ['name','url']:
            print("nulls in column=",i,"is ",end,"end nulls")
    print("nulls in data=\n",data.isna().sum(),"end nulls")
    file2="mid"+filename
    # data.to_csv(file2,na_rep='NA')
    data.to_csv(file2)

    return file2

        


    
def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=logging.WARN)

    if my_args.random_seed == -1:
        seed = None
    else:
        seed = my_args.random_seed

    filename = my_args.data_file
    train_file = my_args.train_file
    test_file = my_args.test_file
    verification_name=my_args.verification_file
    if os.path.exists(filename) and os.path.isfile(filename):
        
        basename = get_basename(filename)
        if train_file and os.path.exists(train_file):
            raise Exception("training data file: {} already exists.".format(train_file))
        if test_file and os.path.exists(test_file):
            raise Exception("testing data file: {} already exists.".format(test_file))

        if not train_file:
            train_file = "{}-train.csv".format(basename)
        if not test_file:
            test_file = "{}-test.csv".format(basename)
        if not verification_name:
            verification_name = "{}-verification.csv".format(basename)
            
        data = get_data(filename)
        split_data(data, train_file, test_file, my_args.test_ratio, seed,verification_name)
    else:
        print("{} doesn't exist, or is not a normal file.".format(filename))
    
    return

if __name__ == "__main__":
    main(sys.argv)





            # for k in data[i]:
            #     test_String=(str(k)+" for "+str(i)+" with a type of "+str(type(k))+"on "+str(data[name][location_2]))
            #     if (('NaN' in test_String) or ('Nan' in test_String) or ('nan' in test_String ) or ('NAN' in test_String)):
            #          print(test_String)
            #     if (((isinstance(k, str))==True)):
            #         last_datatype='str'
            #     elif (((isinstance(k, int))==True)):
            #         last_datatype='int'
            #         if ((math.isnan(k))):
            #             data.fillna('')
            #                 # print("changing",data[name][location_2],"from NaN to None")
            #                 # data2.set_index([i,k])=int(-1)
            #                 # time.sleep(1)
            #     # if ((isinstance(k, str))==False):
            #     else:
            #         if ((isinstance(k, float))):
            #             if ((i=='str') or (i=='dex') or (i=='con') or (i=='wis') or (i=='int') or (i=='cha')):
            #                 if (math.isnan(k)):
            #                     print("changing",data[name][location_2],"from NaN to None (fromfloat)")
            #                     # data2.set_index([i,k])=str(-1)         
            #                     # data2[i][k]=str(-1)
            #                 else:
            #                     print("changing",data[name][location_2],"from flaot to int")
            #                     data2.set_index([i,k])=str(k)
            #                     # data2[i][k]=str(k)
                            
            #             elif ((math.isnan(k)) and (last_datatype=='str')):
            #                 print("changing",data[name][location_2],"from NaN to None")
            #                 data2.set_index([i,k])=str('')
                            
            #                 # data2[i][k]=str('')
            #                 # time.sleep(1)
            #             elif ((math.isnan(k)) and (last_datatype=='int')):
            #                 print("changing",data[name][location_2],"from NaN to None")
            #                 data2.set_index([i,k])=str(-1)
                            
                            # data2[i][k]=str(-1)
                            # time.sleep(1)
                # location_2+=1
                
                # if (location_2==halfway):
                #     print('halfway')
        # input("waiting to move")

        
    # data = pd.read_csv("fixed"+filename, index_col=0)
    # datanew= pd.read_csv(filename)
    # for i in data:
    #     if (i==''):
    #         datanew[i]=''
    #     if (i=='url'):
    #         datanew[i]=''

    # datanew.to_csv("fixed_2_"+filename)

        # time.sleep(2)
