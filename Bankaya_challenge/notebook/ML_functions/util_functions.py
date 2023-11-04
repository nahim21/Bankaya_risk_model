import math
import pandas as pd
import seaborn as sns
import missingno as msno
import warnings
from scipy.stats import norm
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


# Function to detect the missing values
def inspection(dataframe):
    # === Define numerical and categorical variables ===
    cols_numerical = dataframe.select_dtypes(include=np.number).columns.tolist()
    cols_categorical = dataframe.columns.difference(cols_numerical).tolist()

    print("==== Number of labeles: Cardinality ====\n")
    for var in cols_categorical:
        print(var, "contains", len(dataframe[var].unique()), "labels")

    print(" \n==== Types of the variables we are working with ====\n")
    print(dataframe.dtypes, "\n")

    # print("\n====Total missing values per variable ====\n")
    # print(dataframe.isnull().sum())

    total_samples = dataframe.isnull().any(axis=1).sum()
    print(f"\nTotal Samples with missing values: {total_samples}")

    print("\n====Total percentage of missing values per variable ====\n")
    print((dataframe.isnull().sum() / len(dataframe) * 100).sort_values(ascending=False))

    # Identify duplicate rows
    duplicate_rows = dataframe.duplicated().sum()
    # Remove duplicate rows if any
    if duplicate_rows > 0:
        dataframe.drop_duplicates(inplace=True)
        message = f'\n ===== Removed {duplicate_rows} duplicate rows.====== \n '
        print(message)
    else:
        message = '\n ===== No duplicate rows found. ====== \n'
        print(message)

    msno.bar(dataframe)



# Function compare two mean values for independet groups in order to get a statistical difference.
def two_means_t_test(group1, group2):

    stats_group1 = group1.describe()
    stats_group2 = group2.describe()

    # Perform independent t-test
    t_statistic, p_value = stats.ttest_ind(group1, group2)

    # Print the t-statistic and p-value
    print("t-statistic:", t_statistic)
    print("p-value:", p_value)

    if p_value > 0.05:
        print("Accept null hyptohesis (no statistically significant difference)")
    else:
        print("Reject null hyptohesis and Accept alternative(there is a statistically significant difference)")


#Function to plot the outliers with a histogram and boxplot
def plot_diagnostic_outlier(dataframe,col_to_eval):
    """
    This function return both plots dist plot and boxplot, in order to evaluate the outliers
    """
    # Crete template
    fig,axes = plt.subplots(1,2,figsize=(16,5))
    fig.suptitle(f"Diagnostics for the {col_to_eval} variable ")
    # Include the distribuition plot
    plt.subplot(1,2,1)
    sns.distplot(dataframe[col_to_eval])
    plt.subplot(1,2,2)
    # Include the boxplot
    sns.boxplot(dataframe[col_to_eval])
    plt.show()


#Function to run a diagnostic with distribuitions and correlation variables
def full_diagnostic(dataframe,cols_to_exclude=None,class_col=None):
    """
     This function returns a full diagnostic in distribuitions and correlation between the variables.
    """
    cols_numerical = dataframe.select_dtypes(include=np.number).columns.tolist()# --finding all the numerical columns from the dataframe
    df = dataframe[cols_numerical].astype(float) # --Creating a dataframe only with the numerical columns
    if (cols_to_exclude == None):
        sns.pairplot(df,kind="scatter",diag_kind = "kde",palette="Rainbow")

    else:
        df = df[df.columns.difference(cols_to_exclude)] #--Columns to exclude
        #df = df[df.columns.difference([class_col])]
        sns.pairplot(df,kind="scatter",diag_kind = "kde",palette="Rainbow")


#Function to detect the limits upper and lower to prepare the dataframe to remove outliers
def select_diagnostic_outliers(dataframe,col_to_eval,kind="normal"):

    """
    This function remove the outliers from the data set when you pass the columns. There are two kinds of treatment for removing
    outliers, Z-score ttreatment when the dist follows a normal and when the dist is skewedthe technique is IQR(Inter Quartil Range)
    """
    if kind == "normal":
        # ----- Create boundaries (upper and lower limits)
        print("\n==== Upper and lower boundaries =====\n")
        upper_limit = dataframe[col_to_eval].mean() + 3 * dataframe[col_to_eval].std()
        lower_limit = dataframe[col_to_eval].mean() - 3 * dataframe[col_to_eval].std()
        print(f"Highets allowed: {upper_limit}")
        print(f"Lowest allowed: {lower_limit}")
        print("\n==== Finding outliers to remove =====\n")
        # -------- Finding the outliers
        df = dataframe[(dataframe[col_to_eval] > upper_limit) | (dataframe[col_to_eval] < lower_limit)]
        print(f"outliers to removed: {len(df)}")


    elif kind == "iqr":
        q1,q3 = np.percentile(dataframe[col_to_eval],[25,75])
        iqr = q3-q1
        # finding upper and lower limits
        print("\n==== Upper and lower boundaries =====\n")
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + (1.5 * iqr)
        print(f"Highets allowed: {upper_limit}")
        print(f"Lowest allowed: {lower_limit}")
        # -------- Finding the outliers
        print("\n==== Finding outliers to remove =====\n")
        df = dataframe[(dataframe[col_to_eval] > upper_limit) | (dataframe[col_to_eval] < lower_limit)]
        print(f"outliers to removed: {len(df)}")

#Function to remove the outliers from the dataset
def trimming_outliers(dataframe,col_to_eval,upper_limit,lower_limit):
    """
    This function remove the outliers from the dataframe. As a result returns a new dataframe without the outliers
    """
    new_dataframe = dataframe[(dataframe[col_to_eval]<=upper_limit) & (dataframe[col_to_eval]>=lower_limit)]
    return new_dataframe


# This function help to find the user id base on an identifier:
def lookup_user_id(identifier, install_df):
    match = install_df["identifier1"] == identifier
    user_id = install_df.loc[match, "user_id"]
    if len(user_id) > 0:
        return user_id.values[0]
    else:
        return None



#Function to run a matrix correlation
def matrix_correlation(dataframe,cols_to_exclude = None,class_col=None):
    """
    This function is for plotting the matrix correlation in a dataset:
    cols_to_exclude: columns that your dont have to be include in the dataset, the value is passed in a list form
    class_col = if you want to include the target value to be plotted you can added it, if not is anone value fixed.
    """
    if cols_to_exclude == None:
        correlation = dataframe.corr()
        plt.figure(figsize=(16,12))
        plt.title("Correlation Heatmap")
        ax = sns.heatmap(correlation,square=True,annot=True,fmt=".2f",linecolor="White",cmap="RdYlGn")
        ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
        ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
        plt.show()
    else:
        df = dataframe[dataframe.columns.difference(cols_to_exclude)] #--Columns to exclude
        correlation = df.corr()
        plt.figure(figsize=(16,12))
        plt.title("Correlation Heatmap")
        ax = sns.heatmap(correlation,square=True,annot=True,fmt=".2f",linecolor="White",cmap="RdYlGn")
        ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
        ax.set_yticklabels(ax.get_yticklabels(),rotation=30)
        plt.show()

# Function to convert days to months
def days_to_months(n):
    if n > 31:
        return f"{n // 30} months {n % 30} days"
    else:
        return f"{n} days"

# Function to return the columns which should be removed from the dataset.
def remove_highly_correlated_features(df, threshold,cols_to_exclude=None):
    """
    Remove highly correlated features from a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame with numerical features.
        threshold (float): The correlation coefficient threshold for feature removal.
        cols_to_exclude (list): List of columns to exclude from the correlation check.

    Returns:
        list: A list of features to remove.
    """

    # Remove the excluded columns from the DataFrame for correlation calculation
    if cols_to_exclude is not None:
        df_filtered = df.drop(columns=cols_to_exclude, errors='ignore')
    else:
        df_filtered = df

    # Calculate the correlation matrix
    corr_matrix = df_filtered.corr().abs()

    # Create a boolean mask for the upper triangle of the correlation matrix
    upper_triangle_mask = pd.DataFrame(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_),
                                       index=corr_matrix.index, columns=corr_matrix.columns)

    # Find index and columns pairs for features with correlation above the threshold
    to_drop = [column for column in upper_triangle_mask.columns if any(corr_matrix.loc[column][upper_triangle_mask[column]] > threshold)]
    print(f" === High correlated features to be removed from the Dataset====")

    return to_drop