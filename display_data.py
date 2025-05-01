#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path
import numpy

import pandas as pd
import time

import math
import matplotlib.pyplot as plt

def get_data(filename):
    """
    Assumes column 0 is the instance index stored in the
    csv file.  If no such column exists, remove the
    index_col=0 parameter.
    """
    data = pd.read_csv(filename, index_col=0)
    return data

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))
    return basename

def get_feature_and_label_names(my_args, data):
    # print("grabbing features and labels")
    label_column = my_args.label
    feature_columns = my_args.features

    if label_column in data.columns:
        label = label_column
    else:
        label = ""

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in data.columns:
                # print (feature_column)
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label and feature_column!='url'and feature_column!='name':
                features.append(feature_column)

    print("features are: ",features,"\nLabel is ",label)
    return features, label

def display_feature_histograms(my_args, data, figure_number):
    """
    Display a histogram for every feature and the label, if identified.
    """
    feature_columns, label_column = get_feature_and_label_names(my_args, data)

    total_count = len(feature_columns)
    if label_column:
        total_count += 1
    size = int(math.ceil(math.sqrt(total_count)))
    # size = 500


    # size = int(math.ceil(math.sqrt(total_count)))
    # size = int(math.ceil(total_count/2))

    
    fig = plt.figure(figure_number, figsize=(50, 50))
    fig.suptitle( "Feature Histograms" )
    n_max = 1
    all_ax = []
    for i in range(1, len(feature_columns)+1):
        feature_column = feature_columns[i-1]
        if feature_column in data.columns:
            print("feature column:", feature_column, "\ndata :\n",data[feature_column])
            count=0
            for lines in data[feature_column]:
                if feature_column=='speed' or feature_column=='legendary':
                    check=isinstance(lines, float)
                    print("line ",lines, "has this for a float check",check)
                    if (check):
                        lines=''
                        
                        # feature_column[count]=lines
                        data[feature_column][count]=lines

                else:   
                    print (lines)
                count+=1
                 
            # for i in feature_column:
            #     # if (isinstance(feature_column, float)):
            #         if (isinstance(feature_column[0], str)):
            #             print("got a float")
            #             data[feature_column].fillna('', inplace=True)
            #         elif (isinstance(feature_column[0], int)):
            #             print("got a float")
            #             data[feature_column].fillna(-1, inplace=True)
                # feature_column.__floor__
            print(feature_column)
            ax = fig.add_subplot(size, size, i)
            # print('ax',ax)
            ax.set_yscale("log")
            # print('ax.yscale set',ax,'data is ',data[feature_column])
            # for j in range(1,len(data[feature_column])):
            #     if ((isinstance(data[feature_column][j], str))==False):
            #         if ((isinstance(data[feature_column][j], float))):
            #             if ((math.isnan(data[feature_column][j]))):
            #                 print("changing",data[feature_column][j],"from NaN to None")
            #                 data[feature_column][j]=str()
            #                 time.sleep(1)
            
            if data[feature_column].dtype == bool or data[feature_column].dtype == numpy.bool_:
                data[feature_column] = data[feature_column].astype(int)
            n, bins, patches = ax.hist(data[feature_column], bins=20)
            # print("n ",n,"\nbins ",bins,"\npatches ",patches)
            if max(n) > n_max:
                n_max = max(n)
                # print(n_max)
            ax.set_xlabel(feature_column)
            ax.locator_params(axis='x', tight=False, nbins=5)
            all_ax.append(ax)
        else:
            logging.warn("feature_column: '{}' not in data.columns: {}".format(feature_column, data.columns))


    if label_column:
        ax = fig.add_subplot(size, size, total_count)
        ax.set_yscale("log")
        
        if data[feature_column].dtype == bool or data[feature_column].dtype == numpy.bool_:
            data[feature_column] = data[feature_column].astype(int)
        n, bins, patches = ax.hist(data[label_column], bins=50)
        if max(n) > n_max:
            n_max = max(n)
        ax.set_xlabel(label_column)
        ax.locator_params(axis='x', tight=False, nbins=5)
        all_ax.append(ax)

    for ax in all_ax:
        ax.set_ylim(bottom=1.0, top=n_max)

    # fig.tight_layout()
    basename = get_basename(my_args.data_file)
    if (len(feature_columns)>10):
        figure_name = "{}-histogram.{}".format(basename, "pdf")
    else:
        figure_name = "{}-histogram-{}.{}".format(basename, "-".join(feature_columns), "pdf")
    
    fig.savefig(figure_name)
    plt.close(fig)
    return

def display_label_vs_features(my_args, data, figure_number):
    """
    Display a plot of label vs feature for every feature and the label, if identified.
    """
    feature_columns, label_column = get_feature_and_label_names(my_args, data)

    total_count = len(feature_columns)
    if label_column:
        total_count += 1
    # size = int(math.ceil(total_count))
    size = int(math.ceil(math.sqrt(total_count)))

    # size = 500

    
    # size = int(math.ceil(math.sqrt(total_count)))

    all_ax = []
    fig = plt.figure(figure_number, figsize=(50, 50))
    fig.suptitle( "Label vs. Features" )
    for i in range(1, len(feature_columns)+1):
        feature_column = feature_columns[i-1]
        if feature_column in data.columns:
            # print("\n\n\nfeature column:", feature_column, "\ndata :\n",data[feature_column],"\n\n\n")
            inlines=[]
            count=0
            changes=False
            for lines in data[feature_column]:
                # align
                # if feature_column=='speed' or feature_column=='align' or feature_column=='legendary':
                    # print ("feature column",feature_column)
                check=isinstance(lines, float)
                # print("line ",lines, "has this for a float check",check)
                if (check):
                    print(f"line {lines} in {feature_column} showing up as a float")
                    lines_changed=''
                    changes=True
                    inlines.append(lines_changed)
                    # feature_column[count]=lines
                else:
                    inlines.append(lines)
                # else:   
                    # print (lines)
                count+=1
            if (changes):
                # print (inlines)
                data[feature_column]=inlines
                data[feature_column] = data[feature_column].astype(str)
            
            ax = fig.add_subplot(size, size, i)
            # print("ax",ax)
            ax.scatter(feature_column, label_column, data=data, s=1)
            ax.set_xlabel(feature_column)
            # print("testing")
            ax.set_ylabel(label_column)
            ax.locator_params(axis='both', tight=False, nbins=5)
            all_ax.append(ax)
        else:
            logging.warn("feature_column: '{}' not in data.columns: {}".format(feature_column, data.columns))
    data.to_csv("midpoint.csv")
            
    if label_column:
        ax = fig.add_subplot(size, size, total_count)
        ax.scatter(label_column, label_column, data=data, s=1)
        ax.set_xlabel(label_column)
        ax.set_ylabel(label_column)
        ax.locator_params(axis='both', tight=False, nbins=5)
        all_ax.append(ax)

    # fig.tight_layout()
    basename = get_basename(my_args.data_file)
    if (len(feature_columns)>10):
        figure_name = "{}-scatter.{}".format(basename, "pdf")
    else:
        figure_name = "{}-scatter-{}.{}".format(basename, "-".join(feature_columns), "pdf")
    fig.savefig(figure_name)
    plt.close(fig)

    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Create Data Plots')
    parser.add_argument('action', default='all',
                        choices=[ "label-vs-features", "feature-histograms",
                                  "all" ], 
                        nargs='?', help="desired action")
    parser.add_argument('--data-file',               '-d', default="",    type=str,   help="csv file of data to display")
    parser.add_argument('--features',  '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',               '-l', default="label",    type=str,   help="column name for label")

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
    
    return my_args
def fix_Data_rewrite(filename):
    data2 = pd.read_csv(filename)
    data=data2
    print("nulls in data=\n",data.isna().sum(),"end nulls")

    # null=data["legendary"][0]
    # print(null)
    # print(data[i].dtype)
    # print(data[i].unique())
    # print("null is ",null)
    # time.sleep(10)
    for i in data.columns:
        # print("column:",i)
        if i in ['initiative/advantageMode','legendary','isNpc','isNamedCreature']:
            
            data[i+"_missing"]=data[i].isnull().astype(int)

            # print(null)
            print(data[i].dtype)
            print(data[i].unique())
            # print('hit legendary',data[i])
            coldata=data[i]

            before=data[i].isna().sum()
            # if before!=0:
            #     print('')
                # print("before",before)
            # data[i].replace(0,null)
            # coldata2=coldata.replace(0,np.nan)
            data[i]=coldata
            print(i)
            if (i in ['initiative/advantageMode','legendary']):
                coldata2=coldata.apply(lambda x:1 if x==data[i].unique()[1] else 0)
                
                # data[i].replace(1,data[i].unique()[1])
            else:
                coldata2=coldata.apply(lambda x:1 if (x=='true' or x=='True' or x==data[i].unique()[2])  else 0)

            data[i]=coldata2
            # coldata2=coldata.replace(0,null)
            # print("coldata",coldata,"\ncol2",coldata2)
            print('saved data[i]',data[i])
            print(data[i].unique())

            # data[i].replace(1,data[i].unique()[1])
            if before!=0:
                print("after:",data[i].isna().sum())
        if i in ['cr']:
            lines=[]
            for j in data[i]:
                if (j=='1/8' or j=='8-Jan'):
                    # print("Altering ",j, "from 1/8 to 0")
                    changes=True
                    lines_changed='0'
                    lines.append(lines_changed)

                    # data2.set_index([i,j])=str(0)

                elif (j=='1/4' or j=='4-Jan'):
                    # print("Altering ",j, "from 1/4 to 1")
                    changes=True

                    lines_changed='1'
                    lines.append(lines_changed)

                    # data2.set_index([i,j])=str(1)

                elif (j=='1/2' or j=='2-Jan'):
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
                

        elif i not in ['type/tags','speed/burrow/condition',"speed/climb/condition","speed/fly/condition","speed/swim/condition","traitTags"
                       ,"spellcasting","damageTagsSpell","spellcastingTags","conditionInflictSpell","savingThrowForcedSpell","actionTags","bonus","damageTagsLegendary",
                       "savingThrowForcedLegendary","conditionInflictLegendary","resist","immune","conditionImmune","items","group","summonedByClass"]:
            print (i)
            d_data=data[i].dtype
            
            if d_data == 'object':
                print(f"Fixing string column: {i}")
                data[i + '_missing'] = data[i].isna().astype(int)
                data[i] = data[i].fillna("").astype(str)
            check=isinstance(data[i][0], int)
            check2=isinstance(data[i][0],float)
            if (d_data==float or d_data==numpy.float64):
                print(f'{i} is showing up as D_data type:( {d_data} )(should be float)')
                
                try:
                    # Try to convert to numeric
                    coldata = pd.to_numeric(data[i], errors='coerce')
                    missing_before = coldata.isna().sum()
                    if missing_before > 0:
                        print(f"{i} had {missing_before} NaNs before fixing")

                    data[i + '_missing'] = coldata.isna().astype(int)
                    data[i] = coldata.fillna(-1).astype(int)
                except Exception as e:
                    print(f"Skipping column {i} due to error: {e}")
                # coldata=data[i]
                # data[i+'_missing']=coldata.isnull().astype(int)
                # coldata2=coldata.fillna(-1).astype(float)
                # data[i]=coldata2
                # data[i]=data[i].astype(int)
            elif check or check2 or d_data==float or d_data==int:
                coldata=data[i]
                data[i+'_missing']=coldata.isnull().astype(int)
                coldata2=coldata.fillna(-1).astype(int)
                data[i]=coldata2
                # coldata=data[i]
                # coldata3=coldata.astype("Int64")
                # coldata2=coldata.astype("Int64")
                # coldata2=coldata3.fillna('')
                # data[i]=coldata2
                # print("column",i,"set to be int 64")
            else:
                pass
        end=data[i].isna().sum()
        if end!=0 and i not in ['name','url']:
            print("nulls in column=",i,"is ",end,"end nulls")
    print("nulls in data=\n",data.isna().sum(),"end nulls")
    file2="mid"+filename
    # data.to_csv(file2,na_rep='NA')
    data.to_csv(file2)

    return file2

# def fix_Data_rewrite(filename):
#     data = pd.read_csv(filename)
#     print("nulls in data=\n",data.isna().sum(),"end nulls")

#     data
#     # null=data["legendary"][0]
#     # print(null)
#     # print(data[i].dtype)
#     # print(data[i].unique())
#     # print("null is ",null)
#     # time.sleep(10)
#     for i in data.columns:
#         # print("column:",i)
#         if i in ['Legendary','legendary']:
            
#             data["legendary_missing"]=data[i].isnull().astype(int)

#             # print(null)
#             print(data[i].dtype)
#             print(data[i].unique())
#             # print('hit legendary',data[i])
#             coldata=data[i]

#             # before=data[i].isna().sum()
#             # if before!=0:
#             #     print('')
#                 # print("before",before)
#             # data[i].replace(0,null)
#             # coldata2=coldata.replace(0,np.nan)
#             coldata2=coldata.apply(lambda x:1 if x=='Legendary'else 0)

#             # coldata2=coldata.replace(0,null)
#             # print("coldata",coldata,"\ncol2",coldata2)
#             data[i]=coldata2
#             # print('saved data[i]',data[i])
#             # data[i].replace(1,"Legendary")
#             if before!=0:
#                 print("after:",data[i].isna().sum())
#         elif i in ['cr']:
#             lines=[]
#             for j in data[i]:
#                 if (j=='1/8'):
#                     # print("Altering ",j, "from 1/8 to 0")
#                     changes=True
#                     lines_changed='0'
#                     lines.append(lines_changed)

#                     # data2.set_index([i,j])=str(0)

#                 elif (j=='1/4'):
#                     # print("Altering ",j, "from 1/4 to 1")
#                     changes=True

#                     lines_changed='1'
#                     lines.append(lines_changed)

#                     # data2.set_index([i,j])=str(1)

#                 elif (j=='1/2'):
#                     changes=True

#                     # print("Altering ",j, "from 1/2 to 2")
#                     # data2.set_index([i,j])=str(2)
#                     lines_changed='2'
#                     lines.append(lines_changed)
#                 # elif (float(j)==float('nan')):
#                     # print(data[i]," is showing as NaN ",j)
#                 else:
#                     changes=True

#                     buffer=j
#                     # print("buffeirng ",buffer)
#                     lines_changed= int(buffer)
#                     lines_changed+=3
#                     # data2.set_index([i,j])=str(j_modified)

#                     lines.append(str(lines_changed))
#             data[i]=lines
                

#         elif i not in ['hahaha']:
#             print (i)
#             d_data=data[i].dtype
#             # print("Col",i," has a datatype of ",d_data)
#             if d_data=='object':
#                 data_modified=pd.to_numeric(data[i],errors='coerce')
#                 if data_modified.isnull().sum()>0:
#                     # print("Col",i," has a datatype of ",d_data)
#                     before=data[i].isna().sum()
#                     coldata=data[i]

#                     if before!=0:
#                         print("before",before)
#                     # coldata2=coldata.replace('',np.nan)
#                     coldata2=coldata.fillna('')


#                     # data[i]=data[i].replace('',np.nan)
#                     # data[i]=data[i].replace(1,"Legendary")
#                     if before!=0:
#                         print("after",data[i].isna().sum())
#                     data[i]=coldata2           

#                 else:
#                     data[i]=data_modified
 
#             else:
#                 check=isinstance(data[i][0], int)
#                 check2=isinstance(data[i][0],float)
#                 if check or check2:
#                     coldata=data[i]
#                     data[i+'_missing']=coldata.isnull().astype(int)
#                     coldata2=coldata.fillna(-1)
#                     data[i]=coldata2
#                     # coldata=data[i]
#                     # coldata3=coldata.astype("Int64")
#                     # coldata2=coldata.astype("Int64")

#                     # coldata2=coldata3.fillna('')
#                     # data[i]=coldata2
#                     # print("column",i,"set to be int 64")
#         end=data[i].isna().sum()
#         if end!=0 and i not in ['name','url']:
#             print("nulls in column=",i,"is ",end,"end nulls")
#     print("nulls in data=\n",data.isna().sum(),"end nulls")
#     file2="mid"+filename
#     # data.to_csv(file2,na_rep='NA')
#     data.to_csv(file2)

#     return file2


def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=logging.WARN)

    filename = my_args.data_file
    if os.path.exists(filename) and os.path.isfile(filename):
        data = get_data(filename)

        if my_args.action in ("all", "label-vs-features"):
            display_label_vs_features(my_args, data, 1)
        if my_args.action in ("all", "feature-histograms"):
            display_feature_histograms(my_args, data, 2)

    else:
        print(filename + " doesn't exist, or is not a file.")
    
    return

if __name__ == "__main__":
    main(sys.argv)
    
