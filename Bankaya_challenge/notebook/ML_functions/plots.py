import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns

# This function plot multiple attributes and cross them in order to get a better understanding of the
#data behavior

def cross_values_plot(df, numeric_col, cat_col1, cat_col2, temp_col_name):
    """
    Plots the distribution of a numeric column grouped by two categorical columns.
    
    Parameters:
    - df (DataFrame): The input DataFrame.
    - numeric_col (str): The name of the numeric column to analyze.
    - cat_col1 (str): The name of the first categorical column to group by.
    - cat_col2 (str): The name of the second categorical column to group by.
    - temp_col_name (str): Temporary column name to hold aggregated data. 
                            Should be different from existing column names.
    
    Returns:
    None. Displays a Plotly bar plot.
    """
    
    # Calculate the total count for cat_col1
    total_by_cat1 = df.groupby(cat_col1).size().reset_index(name=numeric_col)
    
    # Calculate the distribution of numeric_col grouped by cat_col1 and cat_col2
    distribution_by_both_cats = df.groupby([cat_col1, cat_col2]).size().reset_index(name=temp_col_name)
    
    # Merge the two DataFrames on cat_col1
    merged_df = pd.merge(distribution_by_both_cats, total_by_cat1, on=cat_col1)
    
    # Calculate the percentage for each group
    merged_df['percentage'] = (merged_df[temp_col_name] / merged_df[numeric_col]) * 100
    
    # Create the plot
    fig = px.bar(merged_df, 
                 x=cat_col1, 
                 y=numeric_col, 
                 color=cat_col2,
                 title=f'{numeric_col} by {cat_col1} Split by {cat_col2}', 
                 text='percentage',
                 labels={cat_col1: cat_col1, numeric_col: temp_col_name, cat_col2: cat_col2})
    
    # Update the text on the bars
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    
    # Show the plot
    fig.show()


def weekdays_plot (df,col_num,col1,col2,col_date,calculation="sum",col3=None): # --- remove var

    """
    This function plots multiple categorical columns with an objetctive variable.
    df:dataframe
    col_num: numeric variable or objective variable
    cols1: categorical variable to cross the information
    cols2: categorical variable to cross the information
    col_date: It is important that your data set contains a date column in roder to create the weeks
    calcultation: By default its size, which its how you want that the group makes the computation(size, sum, mean,) depending
    of the metric.
    """

    # Create a copy of the dataset in order that the dataset is not affected
    df2 = df.copy()

    # Convert date column to datetime
    df2[col_date] = pd.to_datetime(df2[col_date])

    # Create the weekdays column in the datset
    df2['day_of_week'] = df2[col_date].dt.day_name()

    # Choose the appropriate calculation method
    if calculation == "sum":
        calculation_method = pd.DataFrame.sum
    elif calculation == "mean":
        calculation_method = pd.DataFrame.mean
    elif calculation == "count":
        calculation_method = pd.DataFrame.count
    else:
        raise ValueError("Invalid calculation method. Choose from: count, sum, mean.")



    if col2 == None:

        # Most active metrics
        most_active_weekdays = df2.groupby("day_of_week")[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)
        most_active_dates = df2.groupby(col_date)[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)
        most_active_col1 = df2.groupby(col1)[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)




        print("\n====== Most Active Weekdays:========\n")
        print(most_active_weekdays)

        print("====== Most Active Dates:======\n")
        print(most_active_dates)

        print(f"\n====== Most Active {col1}:======\n")
        print(most_active_col1)


        #General
        fig = px.bar(most_active_weekdays, x='day_of_week', y=col_num, labels={'weekday_name': 'Weekday', col1: col1})
        fig.update_layout(title=f'{col_num} by Day of Week')
        fig.show()

        # Distribution by weekday and other metrics
        most_active_with_col1 = df2.groupby(["day_of_week", col1])[col_num].apply(calculation_method).reset_index()

        #Calculate percentage over the total of each weekday col1
        total_weekly_col_num_col1 = most_active_with_col1.groupby('day_of_week')[col_num].transform('sum')
        most_active_with_col1['percentage'] = most_active_with_col1[col_num] / total_weekly_col_num_col1 * 100

        # Distribution by weekday and col1
        fig = px.bar(most_active_with_col1, x="day_of_week", y=col_num, color=col1,
                     title=f'{col_num} by Weekday and {col1}',
                     labels={'day_of_week': 'Weekday', col_num: col_num, col1: col1},text="percentage")

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.show()


    elif col3 == None:

        # Distribution by weekday
        most_active_weekdays = df2.groupby("day_of_week")[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)
        most_active_dates = df2.groupby(col_date)[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)
        most_active_col1 = df2.groupby(col1)[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)
        most_active_col2 = df2.groupby(col2)[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)


        print("\n====== Most Active Weekdays:========\n")
        print(most_active_weekdays)

        print("====== Most Active Dates:======\n")
        print(most_active_dates)

        print(f"\n====== Most Active {col1}:======\n")
        print(most_active_col1)

        print(f"\n====== Most Active {col2}:======\n")
        print(most_active_col2)

        #General
        fig = px.bar(most_active_weekdays, x='day_of_week', y=col_num, labels={'weekday_name': 'Weekday', col2: col2})
        fig.update_layout(title=f'{col_num} by Day of Week')
        fig.show()

        # Distribution by weekday and col2
        most_active_with_col2 = df2.groupby(["day_of_week", col2])[col_num].apply(calculation_method).reset_index()

        #Calculate percentage over the total of each weekday col2
        total_weekly_col_num = most_active_with_col2.groupby('day_of_week')[col_num].transform('sum')
        most_active_with_col2['percentage'] = most_active_with_col2[col_num] / total_weekly_col_num * 100



        fig = px.bar(most_active_with_col2, x="day_of_week", y=col_num, color=col2,
                     title=f'{col_num} by Weekday and {col2}',
                     labels={'day_of_week': 'Weekday', col_num: col_num, col2: col2},text="percentage")

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.show()

        # Distribution by weekday and other metrics
        most_active_with_col1 = df2.groupby(["day_of_week", col1])[col_num].apply(calculation_method).reset_index()

        #Calculate percentage over the total of each weekday
        total_weekly_col_num_col1 = most_active_with_col1.groupby('day_of_week')[col_num].transform('sum')
        most_active_with_col1['percentage'] = most_active_with_col1[col_num] / total_weekly_col_num_col1 * 100



        fig = px.bar(most_active_with_col1, x="day_of_week", y=col_num, color=col1,
                     title=f'{col_num} by Weekday and {col1}',
                     labels={'day_of_week': 'Weekday', col_num: col_num, col1: col1},text="percentage")

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.show()

    else:


        most_active_weekdays = df2.groupby("day_of_week")[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)
        most_active_dates = df2.groupby(col_date)[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)
        most_active_col1 = df2.groupby(col1)[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)
        most_active_col2 = df2.groupby(col2)[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)
        most_active_col3 = df2.groupby(col3)[col_num].apply(calculation_method).reset_index().sort_values(col_num,ascending=False).head(5)



        print("\n====== Most Active Weekdays:========\n")
        print(most_active_weekdays)

        print("====== Most Active Dates:======\n")
        print(most_active_dates)

        print(f"\n====== Most Active {col1}:======\n")
        print(most_active_col1)

        print(f"\n====== Most Active {col2}:======\n")
        print(most_active_col2)

        print(f"\n====== Most Active {col3}:======\n")
        print(most_active_col3)


        # Distribution by weekday and other metrics
        most_active_with_col1 = df2.groupby(["day_of_week", col1])[col_num].apply(calculation_method).reset_index()
        most_active_with_col2 = df2.groupby(["day_of_week", col2])[col_num].apply(calculation_method).reset_index()
        most_active_with_col3 = df2.groupby(["day_of_week", col3])[col_num].apply(calculation_method).reset_index()

        #Calculate percentage over the total of each weekday col1
        total_weekly_col_num_col1 = most_active_with_col1.groupby('day_of_week')[col_num].transform('sum')
        most_active_with_col1['percentage'] = most_active_with_col1[col_num] / total_weekly_col_num_col1 * 100

        #Calculate percentage over the total of each weekday col2
        total_weekly_col_num = most_active_with_col2.groupby('day_of_week')[col_num].transform('sum')
        most_active_with_col2['percentage'] = most_active_with_col2[col_num] / total_weekly_col_num * 100

        #Calculate percentage over the total of each weekday col3
        total_weekly_col_num_col3 = most_active_with_col3.groupby('day_of_week')[col_num].transform('sum')
        most_active_with_col3['percentage'] = most_active_with_col3[col_num] / total_weekly_col_num_col3 * 100

        #General
        fig = px.bar(most_active_weekdays, x='day_of_week', y=col_num, labels={'weekday_name': 'Weekday', col2: col2})
        fig.update_layout(title=f'{col_num} by Day of Week')
        fig.show()

        # Distribution by weekday and col1
        fig = px.bar(most_active_with_col1, x="day_of_week", y=col_num, color=col1,
                     title=f'{col_num} by Weekday and {col1}',
                     labels={'day_of_week': 'Weekday', col_num: col_num, col1: col1},text="percentage")

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.show()


        # Distribution by weekday and col2
        fig = px.bar(most_active_with_col2, x="day_of_week", y=col_num, color=col2,
                     title=f'{col_num} by Weekday and {col2}',
                     labels={'day_of_week': 'Weekday', col_num: col_num, col2: col2},text="percentage")

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.show()


        # Distribution by weekday and col3

        fig = px.bar(most_active_with_col3, x="day_of_week", y=col_num, color=col3,
                     title=f'{col_num} by Weekday and {col3}',
                     labels={'day_of_week': 'Weekday', col_num: col_num, col3: col3},text="percentage")

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.show()

def plot_analysis(df,col,col2,col_date,col3=None,col4=None):
    """
    This function plot until for categorical variables and one numerical.
    Because this function plots the numerical variable over the time it is neccesary that the dataset
    has a date variable.
    col: numerical attrinute
    col2, col3, col4: categorical variables.
    col_date: Its the colum with the date to plot.
    """

    # Distribution of clicks over time
    clicks_over_time = df.groupby(col_date).size().reset_index(name=col)
    clicks_over_time[col_date] = pd.to_datetime(clicks_over_time[col_date]) # --  Convert date column to datetime

    if (col3 == None or col4 == None):

        fig1 = go.Figure() # ---> Create template
        fig1.add_trace(go.Scatter(x=clicks_over_time[col_date], y=clicks_over_time[col], mode='lines',
                                  name=f'{col.title()} Over Time'))
        fig1.update_layout(title=f'{col.title()} Over Time')

        # Distribution of clicks across networks, platforms, and countries

        clicks_by_network = df[col2].value_counts().reset_index()
        clicks_by_network.columns = [col2, col]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=clicks_by_network[col2], y=clicks_by_network[col],
                              name=f'{col} by {col2}'))
        fig2.update_layout(title=f'{col} by {col2}')

        # Create subplot figure
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=[f'{col} Over Time', f'{col} by {col2}'])
        fig.add_trace(fig1.data[0], row=1, col=1)
        fig.add_trace(fig2.data[0], row=1, col=2)

        fig.update_layout(showlegend=False)
        fig.show()

    else:


        fig1 = go.Figure() # ---> Create template
        fig1.add_trace(go.Scatter(x=clicks_over_time[col_date], y=clicks_over_time[col], mode='lines',
                                  name=f'{col.title()} Over Time'))

        fig1.update_layout(title=f'{col.title()} Over Time')

        # Distribution of clicks across networks, platforms, and countries

        clicks_by_network = df[col2].value_counts().reset_index()
        clicks_by_network.columns = [col2, col]

        clicks_by_platform = df[col3].value_counts().reset_index()
        clicks_by_platform.columns = [col3, col]

        clicks_by_country = df[col4].value_counts().reset_index()
        clicks_by_country.columns = [col4, col]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=clicks_by_network[col2], y=clicks_by_network[col],
                              name=f'{col} by {col2}'))
        fig2.update_layout(title=f'{col} by {col2}')

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=clicks_by_platform[col3], y=clicks_by_platform[col],
                              name=f'{col} by {col3}'))
        fig3.update_layout(title=f'{col} by {col3}')

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=clicks_by_country[col4], y=clicks_by_country[col],
                              name=f'{col} by {col4}'))
        fig4.update_layout(title=f'{col} by {col4}')

        # Create subplot figure
        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=[f'{col} Over Time', f'{col} by {col2}',
                                                               f'{col} by {col3}', f'{col} by {col4}'])
        fig.add_trace(fig1.data[0], row=1, col=1)
        fig.add_trace(fig2.data[0], row=1, col=2)
        fig.add_trace(fig3.data[0], row=2, col=1)
        fig.add_trace(fig4.data[0], row=2, col=2)

        fig.update_layout(showlegend=False)
        fig.show()

# This function plot multiple attributes and cross them in order to get a better understanding of the
#data behavior

def plot_cross_data(df, metric_column, primary_category, secondary_category, aggregated_metric, aggregation_method='mean'):
    """
    Plot aggregated data with percentages for better understanding.
    Parameters:
    df: DataFrame containing the data.
    metric_column: Column containing the metric to be aggregated.
    primary_category: Main categorical variable for grouping data.
    secondary_category: Secondary categorical variable for splitting the primary category.
    aggregated_metric: Renamed metric after aggregation.
    aggregation_method: Method to aggregate the metric ('count', 'sum', 'mean'). Defaults to 'count'.
    """

    # Validate the aggregation method
    if aggregation_method not in ["sum", "mean", "count"]:
        raise ValueError("Invalid aggregation method. Choose from: count, sum, mean.")

    # Aggregate data based on primary and secondary categories
    aggregated_data = df.groupby([primary_category, secondary_category]).agg({metric_column: aggregation_method}).reset_index()
    aggregated_data.rename(columns={metric_column: aggregated_metric}, inplace=True)

    # Calculate the total aggregated metric for each primary category
    total_aggregated_per_primary = aggregated_data.groupby(primary_category)[aggregated_metric].sum().reset_index(name='total_aggregated_metric')

    # Merge and calculate the percentage
    aggregated_data = pd.merge(aggregated_data, total_aggregated_per_primary, on=primary_category)
    aggregated_data['percentage'] = (aggregated_data[aggregated_metric] / aggregated_data['total_aggregated_metric']) * 100

    # Plotting
    fig = px.bar(aggregated_data, x=primary_category, y='total_aggregated_metric', color=secondary_category,
                 title=f'{aggregation_method.capitalize()} of {metric_column} by {primary_category} Split by {secondary_category}',
                 text='percentage',
                 labels={primary_category: primary_category, 'total_aggregated_metric': aggregated_metric, secondary_category: secondary_category})

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.show()

# Function to create a times series plot
def plot_time_series(df, col_to_eval=None, col_date=None, frequency='D',calculation="sum"):

    """
    :param df: the dataset
    :param col_to_eval: numerical column to evaluate
    :param col_date: the date in order to plot as a time series
    :param frequency: daily ('D') or monthly ('M').
    :return: Time series plot
    """
    # Choose the appropriate calculation method
    if calculation == "sum":
        calculation_method = 'sum'
    elif calculation == "mean":
        calculation_method = 'mean'
    elif calculation == "count":
        calculation_method = 'count'
    else:
        raise ValueError("Invalid calculation method. Choose from: count, sum, mean.")

    # Drop NaN rows
    #df = df.dropna(subset=[col_to_eval, col_date])

    # Convert to datetime format if not already
    df[col_date] = pd.to_datetime(df[col_date])
    # Sort Data: Ensure that your dataframe is sorted by date before plotting.
    df = df.sort_values(by=[col_date])
    # Aggregate Data: If there's more than one entry per day, consider aggregating the data to a daily level.
    df = df.set_index(col_date).resample(frequency).agg({col_to_eval: calculation_method}).reset_index()


    if "DataFrame" in str(type(df)):
        df[col_date] = pd.to_datetime(df[col_date])

        # Check frequency to determine plot type
        if frequency == 'M':
            fig = px.bar(df, x=col_date, y=col_to_eval)
        else:
            fig = px.line(df, x=col_date, y=col_to_eval)

        fig.update_layout(title=f"Time Series of {col_to_eval}",
                          xaxis_title=col_date,
                          yaxis_title=col_to_eval)
        fig.show()

    elif "Series" in str(type(df)):
        fig = go.Figure()

        # Check frequency to determine plot type
        if frequency == 'M':
            fig.add_trace(go.Bar(x=df.index, y=df.values, name="Time Series"))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df.values, mode="lines", name="Time Series"))

        fig.update_layout(title=f"Time Series",
                          xaxis_title="Date",
                          yaxis_title="Values")
        fig.show()
    else:
        print("Invalid input: Please provide a DataFrame or a Series.")

#Function to run a matrix correlation
def matrix_correlation(dataframe,cols_to_exclude = None,class_col=None):
    """
    This function is for plotting the matrix correlation in a dataset:
    cols_to_exclude: columns that your dont have to be include in the dataset, the value is passed in a list form
    class_col = if you want to include the target value to be plotted you can added it, if not is a none value fixed.
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

# This function is to create a histogram for the variables.
def plot_histogram(df, var_name):

    # Rice Rule to calculate optimal bin width
    bins = int(2 * len(df[var_name]) ** (1/3))
    # Create a histogram for the specified variable
    fig = px.histogram(df, x=var_name, nbins=bins, title=f'Distribution of {var_name}', histnorm='percent')

    # Update x-axis and y-axis titles
    fig.update_xaxes(title_text=var_name)
    fig.update_yaxes(title_text='Percentage')

    # Add lines to bars for better distinction
    fig.update_traces(marker=dict(line=dict(width=1, color='black')))

    fig.show()

# Function to plot a pie
def plot_pie(df, var_name):

    fig = px.pie(df, names=var_name, title=f'Pie Chart of {var_name}')
    fig.show()

# Function to plot a box plot Analysis
def plot_box(df, x_col, y_col, title):
    """
    This function is for plotting box plots for different categories in a dataset.

    Parameters:
    - df: DataFrame containing the data
    - x_col: The column to be plotted on the x-axis
    - y_col: The column to be plotted on the y-axis
    - title: The title of the plot
    - cols_to_exclude: List of columns to be excluded from the plot
    - class_col: The target value to differentiate the plot (optional)

    Returns:
    None. A box plot will be displayed.
    """
    fig = px.box(df, x=x_col, y=y_col, title=title)
    fig.show()











