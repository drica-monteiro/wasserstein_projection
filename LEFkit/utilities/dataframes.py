import numpy as np


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# dataframe management utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def GetNumericAndCategoricalVariables(input_df):
    """
    Check which variables are Numeric or Categorical. Return two lists with the distinguished
    variables.

    Input:
    - input_df: the studied pandas dataframe
    Ouptuts:
    - cols_numeric: estimated list of the numeric variables in input_df
    - cols_categorical: estimated list of the categorical variables in input_df
    """

    #first treatment
    col_data_raw=list(input_df.columns)
    cols_numeric = list(input_df._get_numeric_data().columns)
    cols_categorical = list(set(col_data_raw) - set(cols_numeric))

    #move binary numeric variables into the list of categorical variables (as this is common in dataframes)
    variables_to_move=[]

    for variable in cols_numeric:
        list_categories=list(set(input_df[variable]))
        if len(list_categories)<3:
            variables_to_move.append(variable)

    for variable in variables_to_move:
        cols_numeric.remove(variable)
        cols_categorical.append(variable)


    return cols_numeric,cols_categorical



def Transform_df_categories(input_df,cols_categorical):
    """
    Transform all categorial variables into integer values and get the convertion to the original name.

    Input:
    - input_df: the studied pandas dataframe
    - cols_categorical: list containing the name of the categorical variables
    Ouptuts:
    - output_df: the transformed dataframe
    - Categories_name_to_id: dictionary containing the conversion from the categories of input_df to those of output_df
    - Categories_id_to_name: dictionary containing the conversion from the categories of output_df to those of input_df
    """

    output_df=input_df.copy(deep=True)

    Categories_name_to_id={}
    Categories_id_to_name={}

    #replace the categories by integer labels
    for variable in cols_categorical:
        Categories_name_to_id[variable]={}
        Categories_id_to_name[variable]={}

        list_categories=list(set(output_df[variable]))
        list_categories.sort()
        id_categories=list(range(len(list_categories)))
        #print('For variable',variable,':',list_categories,' -> ',id_categories)

        output_df[variable].replace(list_categories,id_categories, inplace=True)

        for i in range(len(list_categories)):
            Categories_name_to_id[variable][list_categories[i]]=id_categories[i]
            Categories_id_to_name[variable][id_categories[i]]=list_categories[i]

    return output_df,Categories_name_to_id,Categories_id_to_name


def Get_df_CategoricalVarIndices(input_df,cols_categorical):
    """
    Get the different indices of each categorical variable.

    Input:
    - input_df: the studied pandas dataframe
    - cols_categorical: list containing the name of the categorical variables
    Ouptuts:
    - CategoricalVarIndices: indices of each categorical variable.
    """

    CategoricalVarIndices={}

    #replace the categories by integer labels
    for variable in cols_categorical:

        CategoricalVarIndices[variable]=list(set(input_df[variable]))
        CategoricalVarIndices[variable].sort()

    return CategoricalVarIndices
