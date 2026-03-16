import numpy as np
import pandas as pd
import sklearn as sk
import os


def  get_treated_dataframe(verbose=False):
    """
    Read and treat the adult census data in a similar fashion as what was done in [Besse et al., The American Statistician, 2021]
    -> DirectoryName is the directory in which the files "adult.data.csv" and "adult.test.csv" are located
    -> Return the treated dataframe with no train/test split and sensitive variable extraction
    """

    full_path = os.path.realpath(__file__)
    DirectoryName, filename = os.path.split(full_path)

    #read and merge the original data
    original_data_train = pd.read_csv(
        DirectoryName+"/adult.data.csv",
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "OrigEthn", "Gender", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

    original_data_test = pd.read_csv(
        DirectoryName+"/adult.test.csv",
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "OrigEthn", "Gender", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


    original_data = pd.concat([original_data_test,original_data_train])
    original_data.reset_index(inplace = True, drop = True)

    if verbose==True:
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+                      Original data                          +')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(original_data.tail())

    #data preparation 1/2
    data=original_data.copy()
    data['Child'] = np.where(data['Relationship']=='Own-child', 'ChildYes', 'ChildNo')
    data=data.drop(columns=['fnlwgt','Relationship','Country','Education'])

    #data preparation 2/2
    data=data.replace('<=50K.','<=50K')
    data=data.replace('>50K.','>50K')

    data['OrigEthn'] = np.where(data['OrigEthn']=='White', 'CaucYes', 'CaucNo')

    data_ohe=data.copy()

    data_ohe['Target'] = np.where(data_ohe['Target']=='>50K', 1., 0.)
    #print(' -> In column Target: label >50K gets 1.')

    data_ohe['OrigEthn'] = np.where(data_ohe['OrigEthn']=='CaucYes', 1., 0.)
    #print(' -> In column '+str('OrigEthn')+': label '+str('CaucYes')+' gets 1.')

    data_ohe['Gender'] = np.where(data_ohe['Gender']=='Male', 1., 0.)
    #print(' -> In column '+str('Gender')+': label '+str('Male')+' gets 1.')

    for col in ['Workclass', 'Martial Status', 'Occupation', 'Child']:
        if len(set(list(data_ohe[col])))==2:
            LabelThatGets1=data_ohe[col][0]
            data_ohe[col] = np.where(data_ohe[col]==LabelThatGets1, 1., 0.)
            #print(' -> In column '+str(col)+': label '+str(LabelThatGets1)+' gets 1.')
        else:
            #print(' -> In column '+str(col)+': one-hot encoding conversion with labels '+str(set(list(data_ohe[col]))))
            data_ohe=pd.get_dummies(data_ohe,prefix=[col],columns=[col])

    data_ohe = data_ohe.loc[data_ohe.Age!='age']
    for elem in ['Age', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']:
        data_ohe[elem] = data_ohe[elem].apply(float)

    if verbose==True:
        print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+                      Returned data                          +')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(data_ohe.tail())

    return data_ohe




def  as_in_Besse_AmStat21(SensitiveVarName='Gender',verbose=False,test_ratio=0.33):
    """
    Read and treat the adult census data as in [Besse et al., The American Statistician, 2021]
    ->  SensitiveVarName is the string representing the sensitive variable in "adult.data.csv" and "adult.test.csv"
    -> Return [X_train, X_test, y_train, y_test, S_train, S_test]
    """

    #read and merge original data
    full_path = os.path.realpath(__file__)
    DirectoryName, filename = os.path.split(full_path)

    original_data_train = pd.read_csv(
        os.path.join(DirectoryName, "adult.data.csv"),
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "OrigEthn", "Gender", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    original_data_test = pd.read_csv(
        os.path.join(DirectoryName, "adult.test.csv"),
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "OrigEthn", "Gender", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    original_data = pd.concat([original_data_test,original_data_train])
    original_data.reset_index(inplace = True, drop = True)




    if verbose==True:
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+                      Original data                          +')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(original_data.tail())

    #data preparation 1/3
    data=original_data.copy()

    data['Child'] = np.where(data['Relationship']=='Own-child', 'ChildYes', 'ChildNo')
    data['OrigEthn'] = np.where(data['OrigEthn']=='White', 'CaucYes', 'CaucNo')
    data=data.drop(columns=['fnlwgt','Relationship','Country','Education'])
    data=data.replace('<=50K.','<=50K')
    data=data.replace('>50K.','>50K')

    #print(original_data.tail())

    #data preparation 2/3
    data_ohe=data.copy()
    data_ohe['Target'] = np.where(data_ohe['Target']=='>50K', 1., 0.)
    #print(' -> In column Target: label >50K gets 1.')
    data_ohe['OrigEthn'] = np.where(data_ohe['OrigEthn']=='CaucYes', 1., 0.)
    #print(' -> In column '+str('OrigEthn')+': label '+str('CaucYes')+' gets 1.')

    data_ohe['Gender'] = np.where(data_ohe['Gender']=='Male', 1., 0.)
    #print(' -> In column '+str('Gender')+': label '+str('Male')+' gets 1.')

    for col in ['Workclass', 'Martial Status', 'Occupation', 'Child']:
        if len(set(list(data_ohe[col])))==2:
            LabelThatGets1=data_ohe[col][0]
            data_ohe[col] = np.where(data_ohe[col]==LabelThatGets1, 1., 0.)
            #print(' -> In column '+str(col)+': label '+str(LabelThatGets1)+' gets 1.')
        else:
            #print(' -> In column '+str(col)+': one-hot encoding conversion with labels '+str(set(list(data_ohe[col]))))
            data_ohe=pd.get_dummies(data_ohe,prefix=[col],columns=[col])

    if verbose==True:
        print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+                          Treated data                       +')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(original_data.tail())


    #data preparation 3/3
    #... extract the X and y np.arrays
    y=data_ohe['Target'].values.reshape(-1,1)

    data_ohe_wo_target=data_ohe.drop(columns=['Target'])

    X_col_names=list(data_ohe_wo_target.columns)
    X=data_ohe_wo_target.values

    #... split the learning and test samples
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

    #... print the np.array shapes
    #print('n_train=',X_train.shape[0])
    #print('n_test=',X_test.shape[0])
    #print('p=',X_test.shape[1])

    #... center-reduce the arrays X_train and X_test to make sure all variables have the same scale
    X_train_NoScaling=X_train.copy()
    X_train=sk.preprocessing.scale(X_train)
    X_test_NoScaling=X_test.copy()
    X_test=sk.preprocessing.scale(X_test)

    S_train=X_train_NoScaling[:,X_col_names.index('Gender')].ravel()
    S_test=X_test_NoScaling[:,X_col_names.index('Gender')].ravel()

    print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+             Shape of the returned np.arrays                 +')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print("S_train.shape:",S_train.shape)
    print("X_train.shape:",X_train.shape)
    print("y_train.shape:",y_train.shape)
    print("S_test.shape:",S_test.shape)
    print("X_test.shape:",X_test.shape)
    print("y_test.shape:",y_test.shape)

    print("\nX_col_names:",X_col_names)


    return [X_train, X_train_NoScaling, X_test, X_test_NoScaling, y_train, y_test, S_train, S_test,X_col_names]




def  as_in_Zafar_package(SensitiveVarName='Gender'):

  full_path = os.path.realpath(__file__)
  DirectoryName, filename = os.path.split(full_path)

  [X_train, X_test, y_train, y_test, S_train, S_test,columnNames]= ReadAndTreatAdultCensusData(DirectoryName,SensitiveVarName=SensitiveVarName)

  X_tr=X_train
  y_tr=y_train.flatten()
  y_tr[y_tr==0]=-1
  x_control_tr={}
  x_control_tr[SensitiveVarName]=S_train.flatten()

  X_ts=X_test
  y_ts=y_test.flatten()
  y_ts[y_ts==0]=-1
  x_control_ts={}
  x_control_ts[SensitiveVarName]=S_test.flatten()



  return X_tr, y_tr, x_control_tr, X_ts, y_ts, x_control_ts , columnNames
