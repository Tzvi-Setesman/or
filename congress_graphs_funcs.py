import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from scipy.stats import pearsonr
from scipy import stats
from pandas import Timestamp
from statsmodels.formula.api import ols
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import date


def load_us_metadata():
    # Define the paths to the metadata files
    metadata_path = "Netivot/notebook/twitter_meta/metadata.xlsx"
    # Read the metadata files using pandas
    metadata = pd.read_excel(metadata_path) 
    
    return metadata

def format_data_for_graphs(df = pd.DataFrame):
    
    # Set start of war date
    october_seventh = datetime.strptime("2023-10-07 06:00:00", "%Y-%m-%d %H:%M:%S")

    # List of retired members - filter out if still in db
    ret_members = ["replouiegohmert", "repkevinbrady", "jimlangevin","reppeterdefazio", 
                "repmondaire", "repfredupton", "repseanmaloney","repmeijer",
                "repspeier", "repmobrooks"]

    # Select and rename columns (based on tagged column - needs change when goes automatic)
    df = df.drop(columns=['country', 'conflict_sentiment']).rename(columns={'conflict_sentiment_v': 'conflict_sentiment'})

    # Add columns to df
    df = df.assign(conflict_sentiment=lambda x: pd.to_numeric(x['conflict_sentiment']),
                            date=lambda x: pd.to_datetime(x['upload_date']).dt.date,
                            day_of_week=lambda x: pd.to_datetime(x['upload_date']).dt.dayofweek,
                            # weeks since start of war
                            n_week_from_7_10=lambda x: np.round((pd.to_datetime(x['upload_date']) - october_seventh) / np.timedelta64(1, 'W')),
                            # 
                            username=lambda x: x['username'].str.lower())

    # Translate numeric values to text
    conflict_numeric_dict = { 0 : "not_related", 1: "pro_palestinian" , 2: "complex_attitude" , 3:  "pro_israeli" ,9: "ambiguous"}

    df['conflict_sentiment_chr'] = df['conflict_sentiment'].apply(lambda x: conflict_numeric_dict.get(x))

    # Translate numeric values to rank
    conflict_rank_dict = {1: -1 , 2: 0 , 3: 1 }

    df['conflict_sentiment_rank'] = df['conflict_sentiment'].apply(lambda x: conflict_rank_dict.get(x))

    # Creates new column for conflict relation
    df['is_conflict_related'] = np.select(
        [df['conflict_sentiment'] != 0, df['conflict_sentiment'] ==0],
        ["conflict_related", "not_related"])

    # Various username replacements - if we collect several twitter accounts of the same person, or changed his account
    username_replacements = {
        "ossoff": "senossoff",
        "claudiatenney": "reptenney",
        'repaoc': 'aoc'
        } # Add all other replacements here following the same pattern

    for original, replacement in username_replacements.items():
        df['username'] = np.where(df['username'] == original, replacement, df['username'])

    # Remove data from retired members and lowercasing columns
    df = df[~df['username'].isin(ret_members)]
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]


    # Calculate distance from the latest time point
    last_data_point = df['date'].max()
    df['diff_from_latest_datapoint'] = (df['date'] - last_data_point)
    # Print the latest data point
    print(last_data_point)

    return df 


def daily_tweets_graph(df = pd.DataFrame):
    
    # Total number of tweets per day
    tweets_daily = df.groupby(['date', 'is_conflict_related']).size().reset_index(name='Count')

    # Tidying data for prettier legend
    tweets_daily['is_conflict_related'] = tweets_daily['is_conflict_related'].str.replace("conflict_related", "Related to Conflict").str.replace("not_related", "Not Related")

    # Create a bar graph with both binary values shown side by side
    fig = px.bar(tweets_daily, x='date', y='Count', color='is_conflict_related', barmode='group', title='Amount of Tweets by Relation to Conflict Per Day',
        labels = {"date": "Date", "Count": "Amount of Tweets"})

    fig.update_layout(title={'font_size': 20 ,"x": 0.5}, legend_title = "Relation to Conflict" )

    return fig

def daily_users_conflict_graph(df = pd.DataFrame):
    # Number of users tweeted per day
    conflict_related_data = df[df['is_conflict_related'] == 'conflict_related']
    unique_users_per_day = conflict_related_data.drop_duplicates(subset=['username', 'date'])

    # Count the number of unique users per day
    grouped_users_per_day = unique_users_per_day.groupby('date').size().reset_index(name='Count')


    fig = px.bar(grouped_users_per_day, x= 'date', y = 'Count', title = "Number of Members tweeting conflict related tweets",
                labels = {"date": "Date", "Count": "Amount of Users"} ,color_discrete_sequence = ["#1f77b4"])
    fig.update_layout(title={'font_size': 20 ,"x": 0.5} )

    return fig

def conflict_relation_graph(df = pd.DataFrame):
    conflict_or_not = df.groupby('is_conflict_related').size().reset_index(name='Count').sort_values('is_conflict_related', ascending= False)

    # Tidying data for prettier legend
    conflict_or_not['is_conflict_related'] = conflict_or_not['is_conflict_related'].str.replace("conflict_related", "Related to Conflict").str.replace("not_related", "Not Related")
    conflict_or_not.rename(columns= {"is_conflict_related" : "Conflict Relation"}, inplace= True)

    fig = px.bar(conflict_or_not, x = "Count",y= "Conflict Relation", title = "Overall Amount of Conflict Related Tweets",
    color = 'Conflict Relation', color_discrete_map= {'Not Related' : 'green', 'Related to Conflict': '#d62728'})

    fig.update_layout(title={'text': 'Overall Amount of Conflict Related Tweets','font_size': 20 ,"x": 0.5})

    return fig

def total_stance_graph(df = pd.DataFrame):
    # Filter out anything that isn't a conflict related stance  - (pro-israeli, pro-palestinian, mixed)
    only_stance_data = df[~df['conflict_sentiment_rank'].isna()]

    # Group by stance and sort data
    stance_grouped = only_stance_data.groupby('conflict_sentiment_chr').size().reset_index(name='Count').sort_values('Count', ascending = False)

    # Tidying data for prettier legend
    stance_grouped['conflict_sentiment_chr'] = stance_grouped['conflict_sentiment_chr'].str.replace("pro_israeli", "Pro Israeli").str.replace("pro_palestinian", "Pro Palestinian").str.replace("complex_attitude", "Complex Stance")
    stance_grouped.rename(columns= {"conflict_sentiment_chr" : "Stance on Conflict"}, inplace= True)


    fig = px.bar(stance_grouped, x= 'Count', y = 'Stance on Conflict',title = 'Overall Count per Stance' ,
    color = 'Stance on Conflict' ,color_discrete_map = {'Pro Palestinian': '#d62728','Complex Stance': 'goldenrod','Pro Israeli': 'forestgreen' })

    fig.update_layout(title={'text': 'Overall Amount of Conflict Related Tweets','font_size': 20 ,"x": 0.5})

    return fig

def stance_daily_tweets_amount(df = pd.DataFrame,  barmode = 'stack'):

        # Filter out anything that isn't a conflict related stance  - (pro-israeli, pro-palestinian, mixed)
    only_stance_data = df[~df['conflict_sentiment_rank'].isna()]

    by_stance_daily_amount = only_stance_data.groupby(['conflict_sentiment_chr', 'date']).size().reset_index(name='Count')

    # Tidying data for prettier legend
    by_stance_daily_amount['conflict_sentiment_chr'] = by_stance_daily_amount['conflict_sentiment_chr'].str.replace("pro_israeli", "Pro Israeli").str.replace("pro_palestinian", "Pro Palestinian").str.replace("complex_attitude", "Complex Stance")
    by_stance_daily_amount.rename(columns= {"conflict_sentiment_chr" : "Stance on Conflict"}, inplace= True)

    by_stance_daily_amount = by_stance_daily_amount.sort_values('Count', ascending=False)

    fig = px.bar(by_stance_daily_amount, x='date', y='Count', color='Stance on Conflict', barmode=barmode,
                category_orders={'date': sorted(by_stance_daily_amount['date'].unique())},
                color_discrete_map={'Pro Palestinian': '#d62728','Complex Stance': 'goldenrod','Pro Israeli': 'forestgreen' })
    fig.update_layout(title={'text': 'Total Tweets per Stance per day','font_size': 20 ,"x": 0.5},
                    legend_orientation = "h", legend_x =  0, legend_y = -0.15, legend_title = "")
    
    return fig

def stance_freq_by_party(df = pd.DataFrame, metadata = pd.DataFrame, barmode = 'stack'): 
    conflict_data = df[df['is_conflict_related'] == "conflict_related"]
    conflict_data_with_metadata = conflict_data.merge(metadata, how='left', on='username')  # Update 'some_common_column' accordingly

    # Count and pivot wider
    count_data = conflict_data_with_metadata.groupby(['conflict_sentiment_chr', 'date', 'party']).size().reset_index(name='count')
    pivot_data = count_data.pivot_table(index=['date', 'party'], columns='conflict_sentiment_chr', values='count', fill_value=0)

    # Calculate frequencies
    pivot_data['total'] = pivot_data.sum(axis=1)
    for col in pivot_data.columns[:-1]:  # Excluding the total column
        pivot_data[f'freq_{col}'] = pivot_data[col] / pivot_data['total']
        
    # Reshape for plotting
    pivot_data_dropped = pivot_data.drop(columns=['total','ambiguous','pro_israeli','pro_palestinian','complex_attitude'])
                                            
    freq_sentiment = pd.melt(pivot_data_dropped.reset_index(), id_vars=['date', 'party'], 
                            value_vars=[f'{col}' for col in pivot_data_dropped.columns[1:]],
                            var_name='sentiment', value_name='freq')


    # Making the graph
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Democratic Party', 'Republican Party'))

    # Creates colormap and list to iterate over
    parties = ['Democratic', 'Republican']
    colors = {'freq_pro_palestinian': '#d62728', 'freq_complex_attitude': 'goldenrod', 'freq_pro_israeli': 'forestgreen'}


    for idx, party in enumerate(parties, start=1):
        party_data = freq_sentiment[freq_sentiment['party'] == party]

        # Orders the data in a specific way so that the graph looks nice 
        cat_dtype = pd.CategoricalDtype(categories=['freq_pro_israeli', 'freq_complex_attitude', 'freq_pro_palestinian'], ordered=True)
        party_data['sentiment'] = party_data['sentiment'].astype(cat_dtype)
        party_data = party_data.sort_values('sentiment')

        # Creates graph
        fig_party = px.bar(party_data, x='date', y='freq', color='sentiment', barmode='stack',
                    category_orders={'date': sorted(freq_sentiment['date'].unique())},
                    color_discrete_map=colors)
        
        # 'Loads' subplot into figure
        for trace in fig_party['data']:
            if party == "Democratic":
                trace.update(showlegend=False)
            fig.add_trace(trace, row=1, col=idx)

    # Fix layout
    fig.update_layout(title={'text': 'Stance Comparison by Party','font_size': 20 ,"x": 0.5},
                    legend_orientation = "h", legend_x =  0.25, legend_y = -0.15, 
                    barmode = barmode)
    return fig

def party_interest_over_time(df = pd.DataFrame,  metadata = pd.DataFrame): 

    temporal_interest = pd.merge(df, metadata[['username', 'party', 'legislation_body']], on='username', how='left')

    # Count the data
    temporal_interest = temporal_interest.groupby(['date', 'party', 'is_conflict_related', 'day_of_week', 'legislation_body']).size().reset_index(name='n')

    # Pivot wider
    temporal_interest = temporal_interest.pivot_table(index=['date', 'party', 'day_of_week', 'legislation_body'], columns='is_conflict_related', values='n', fill_value=0).reset_index()
    temporal_interest.columns.name = None  # Remove the MultiIndex

    temporal_interest['total_n'] = temporal_interest['conflict_related'] + temporal_interest['not_related']
    temporal_interest['interest'] = temporal_interest['conflict_related'] / temporal_interest['total_n']

    temporal_interest['is_weekend'] = np.where(temporal_interest['day_of_week'].isin([6, 7]), 1, 0)
    temporal_interest['diff_from_0710'] = (pd.to_datetime(temporal_interest['date']) - pd.to_datetime("2023-10-07")).dt.days

    temporal_interest['datetime'] = pd.to_datetime(temporal_interest['date'])


    fig = px.scatter(temporal_interest, x='datetime', y='interest', color='party',color_discrete_sequence=["#457b9d", "#e63946"],
                    trendline='ols', # Adding trendline
                    title='Interest in Conflict Over Time',
                    labels={'datetime': 'Date', 'interest': 'Interest in % of Tweets'})

    fig.update_layout(title={'font_size': 20 ,"x": 0.5})

    return fig

def party_stance_vs_interest_eng(df = pd.DataFrame, metadata = pd.DataFrame, annotation = True):
    user_summary_stat = df.groupby('username').agg(
        mean_like_amount=('like_amount', 'mean'),
        mean_retweet=('retweet_amount', 'mean'),
        mean_comments=('comments_amount', 'mean'),
        n_tweet=('post_id', 'count')  # Assuming there's a tweet_id column
    )

    # User interest in Israel (% of tweets on conflict)
    user_interest = df.groupby(['username', 'is_conflict_related']).size().reset_index(name='n')
    user_interest = user_interest.pivot_table(index='username', columns='is_conflict_related', values='n', fill_value=0).reset_index()
    user_interest['interest'] = user_interest['conflict_related'] / (user_interest['conflict_related'] + user_interest['not_related'])

    # User stance to Israel
    df['conflict_sentiment_rank'] = pd.to_numeric(df['conflict_sentiment_rank'], errors='coerce')
    # Filter out anything that isn't a conflict related stance  - (pro-israeli, pro-palestinian, mixed)
    user_stance_data = df[~df['conflict_sentiment_rank'].isna()]

    user_stance_data = user_stance_data.groupby('username').agg(
        # Average stance
        mean_stance=('conflict_sentiment_rank', 'mean'),
        # Most common stance
        mode_stance=('conflict_sentiment_rank', lambda x: stats.mode(x)[0])
    )

    # Merge all data
    user_stats_merged = user_summary_stat.merge(user_stance_data, on='username', how='left')
    user_stats_merged = user_stats_merged.merge(user_interest, on='username', how='left')
    user_stats_merged = user_stats_merged.merge(metadata, on='username', how='left')  # Adjust 'metadata' as needed


    # Convert 'legislation_body' and 'party' to string if not already, to use in facet_grid
    user_stats_merged['legislation_body'] = user_stats_merged['legislation_body'].astype(str)
    user_stats_merged['party'] = user_stats_merged['party'].astype(str)

    user_stats_merged = user_stats_merged.dropna()



    # Create figure with 4 subplots
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.01, vertical_spacing=0.02, 
                        subplot_titles= ('Democratic Party','Republican Party'), x_title= 'Average Stance Score', y_title= "Interest in % of Tweets")

    # Set party and legislation_body lists for iteration 
    parties = ['Democratic', 'Republican']
    legislation_bodies = ['Senate', 'House of Representatives']


    # Create a subplot for each party in each body
    for party_index, party in enumerate(parties, start=1):
        for body_index, body in enumerate(legislation_bodies, start = 1):

            # Filter data
            party_data = user_stats_merged[user_stats_merged['party'] == party]
            body_data = party_data[party_data['legislation_body'] == body]

            # Create subplot
            fig_party_body = px.scatter(body_data, x= 'mean_stance', y = 'interest', color = 'mean_stance' )
            
        
            
            if party == 'Republican':
                if body == 'Senate': 
                # Fit axes
                    stance_threshold = 0
                    interest_threshold = 0
                    
                    fig.update_xaxes(range=[-1.1, 1.1], row=body_index, col=party_index, showticklabels=False,  tickvals= [-1,-0.5, -0, 0.5, 1] ) #, showgrid = True, zeroline = True)
                    fig.update_yaxes(range=[0, 1], row=body_index, col=party_index, title_text =  "Senate", side = 'right',
                                    tickvals= [0,0.2,0.4,0.6,0.8, 1.1] , ticktext=['0' ,'', '','', '','1']) #showgrid = True, zeroline = False ,
                    
                elif body == 'House of Representatives': 
                    stance_threshold = 0
                    interest_threshold = 0

                    fig.update_xaxes(range=[-1.1, 1.1], row=body_index, col=party_index, zeroline = True, tickvals= [-1,-0.5, -0, 0.5, 1])
                    fig.update_yaxes(range=[0, 1], row=body_index, col=party_index,  title_text =  "House of Representatives", side = 'right',
                                    tickvals= [0,0.2,0.4,0.6,0.8, 1.1], ticktext = ['0' ,'', '','', '','1']) #,showgrid = False, zeroline = False)  
                    
            elif party == 'Democratic': 
                if body == 'Senate': 
                    stance_threshold = -0.1
                    interest_threshold = 0

                    fig.update_xaxes(range=[-1.1, 1.1], row=body_index, col=party_index, showticklabels=False,  tickvals= [-1,-0.5, -0, 0.5, 1]) # showgrid = False, zeroline = True)
                    fig.update_yaxes(range=[0, 1], row=body_index, col=party_index,  showgrid = True, zeroline = True)

                elif body == 'House of Representatives':
                    stance_threshold = -0.2
                    interest_threshold = 0.25

                    fig.update_xaxes(range=[-1.1, 1.1], row=body_index, col=party_index)#showgrid = False, zeroline = True) #title_text = 'Average Stance Score',
                    fig.update_yaxes(range=[0, 1], row=body_index, col=party_index) # showgrid = False, zeroline = True) # title_text =  "Interest in % of Tweets",
            
            # Update figure
            for trace in fig_party_body['data']:
                fig.add_trace(trace, row=body_index, col=party_index)
                
            
            if annotation == True:
                for i, row in body_data.iterrows():
                    if row['mean_stance'] < stance_threshold:
                        if row['interest'] > interest_threshold:
                            fig.add_annotation(
                                x=row['mean_stance'] + 0.25,
                                y=row['interest'] -0.052,
                                text=str(row['username']), 
                                row=body_index,
                                col=party_index, 
                                bgcolor= 'rgb(250,250,250)',
                                bordercolor = 'rgb(0,0,0)',
                                standoff= 100
                            )
            
    # Update figure for better visual appearance
    fig.update_layout(
        width=1100,  
        height=800, 
        coloraxis_colorscale=[[0.0, "#FF1206"],  [0.5, '#B3AFAF'], [1.0, "#3663AD" ]],
        # Fit title to plot
        # Fiddle with margins to make it look better
        margin=dict(l=100, r=10, t=80, b=60),
        plot_bgcolor = 'rgba(220, 225, 221, 0.5)' ,
        #title={'text': 'Average Stance vs. Interest','font_size': 20 ,"x": 0.5},
        ) 
        
    fig.update_layout(
        coloraxis=dict(colorbar=dict(
            x=1.15,  # Adjust the x position of the legend
            y=1.0,  # Adjust the y position of the legend
            xanchor='right',  # Set x anchor to the right side of legend
            yanchor='top'  # Set y anchor to the top of legend
        ))
    )

    return fig

def party_stance_vs_interest(df = pd.DataFrame, metadata = pd.DataFrame, annotation = True):
    
    # Get general stats on each congressperson - tweets,likes, retweets, comments
    user_summary_stat = df.groupby('username').agg(
        mean_like_amount=('like_amount', 'mean'),
        mean_retweet=('retweet_amount', 'mean'),
        mean_comments=('comments_amount', 'mean'),
        n_tweet=('post_id', 'count')  # Assuming there's a tweet_id column
    )

    # User interest in Israel (% of tweets on conflict)
    user_interest = df.groupby(['username', 'is_conflict_related']).size().reset_index(name='n')
    user_interest = user_interest.pivot_table(index='username', columns='is_conflict_related', values='n', fill_value=0).reset_index()
    user_interest['interest'] = user_interest['conflict_related'] / (user_interest['conflict_related'] + user_interest['not_related'])

    # User stance to Israel
    df['conflict_sentiment_rank'] = pd.to_numeric(df['conflict_sentiment_rank'], errors='coerce')
    # Filter out anything that isn't a conflict related stance  - (pro-israeli, pro-palestinian, mixed)
    user_stance_data = df[~df['conflict_sentiment_rank'].isna()]

    user_stance_data = user_stance_data.groupby('username').agg(
        # Average stance
        mean_stance=('conflict_sentiment_rank', 'mean'),
        # Most common stance
        mode_stance=('conflict_sentiment_rank', lambda x: stats.mode(x)[0])
    )

    # Merge all data
    user_stats_merged = user_summary_stat.merge(user_stance_data, on='username', how='left')
    user_stats_merged = user_stats_merged.merge(user_interest, on='username', how='left')
    user_stats_merged = user_stats_merged.merge(metadata, on='username', how='left')  # Adjust 'metadata' as needed


    # Convert 'legislation_body' and 'party' to string if not already, to use in facet_grid
    user_stats_merged['legislation_body'] = user_stats_merged['legislation_body'].astype(str)
    user_stats_merged['party'] = user_stats_merged['party'].astype(str)

    user_stats_merged = user_stats_merged.dropna()



    # Create figure with 4 subplots
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.01, vertical_spacing=0.02, 
                        subplot_titles= ('דמוקרטים','רפובליקנים'), x_title= 'עמדה', y_title= "עניין")

    fig.update_annotations(font_size=25)



    # Set party and legislation_body lists for iteration 
    parties = ['Democratic', 'Republican']
    legislation_bodies = ['Senate', 'House of Representatives']


    # Create a subplot for each party in each body
    for party_index, party in enumerate(parties, start=1):
        for body_index, body in enumerate(legislation_bodies, start = 1):

            # Filter data
            party_data = user_stats_merged[user_stats_merged['party'] == party]
            body_data = party_data[party_data['legislation_body'] == body]

            # Create subplot
            fig_party_body = px.scatter(body_data, x= 'mean_stance', y = 'interest', color = 'mean_stance' )
            
        
            
            if party == 'Republican':
                if body == 'Senate': 
                # Fit axes
                    stance_threshold = 0
                    interest_threshold = 0
                    
                    fig.update_xaxes(range=[-1.1, 1.1], row=body_index, col=party_index, showticklabels=False,  tickvals= [-1,-0.5, -0, 0.5, 1],  zerolinecolor = 'rgb(100,100,100)' ) #, showgrid = True, zeroline = True)
                    fig.update_yaxes(range=[0, 1.05], row=body_index, col=party_index, title_text =  "הסנאט", side = 'right', title_font_size = 25,
                                    tickvals= [0,0.2,0.4,0.6,0.8, 1.1] , ticktext=['0' ,'', '','', '','1']) #showgrid = True, zeroline = False ,
                    
                elif body == 'House of Representatives': 
                    stance_threshold = 0
                    interest_threshold = 0

                    fig.update_xaxes(range=[-1.1, 1.1], row=body_index, col=party_index, zeroline = True, tickvals= [-1,-0.5, -0, 0.5, 1],  zerolinecolor = 'rgb(100,100,100)')
                    fig.update_yaxes(range=[0, 1.05], row=body_index, col=party_index,  title_text =  "בית הנבחרים", side = 'right', title_font_size = 25,
                                    tickvals= [0,0.2,0.4,0.6,0.8, 1.1], ticktext = ['0' ,'', '','', '','1']) #,showgrid = False, zeroline = False)  
                    
            elif party == 'Democratic': 
                if body == 'Senate': 
                    stance_threshold = -0.1
                    interest_threshold = 0

                    fig.update_xaxes(range=[-1.1, 1.1], row=body_index, col=party_index, showticklabels=False,  tickvals= [-1,-0.5, -0, 0.5, 1],  zerolinecolor = 'rgb(100,100,100)') # showgrid = False, zeroline = True)
                    fig.update_yaxes(range=[0, 1.05], row=body_index, col=party_index,  showgrid = True, zeroline = True )

                elif body == 'House of Representatives':
                    stance_threshold = -0.2
                    interest_threshold = 0.25

                    fig.update_xaxes(range=[-1.1, 1.1], row=body_index, col=party_index,  zerolinecolor = 'rgb(100,100,100)')#showgrid = False, zeroline = True) #title_text = 'Average Stance Score',
                    fig.update_yaxes(range=[0, 1.05], row=body_index, col=party_index) # showgrid = False, zeroline = True) # title_text =  "Interest in % of Tweets",
            
            # Update figure
            for trace in fig_party_body['data']:
                fig.add_trace(trace, row=body_index, col=party_index)
                
            
            if annotation == True:
                for i, row in body_data.iterrows():
                    if row['mean_stance'] < stance_threshold:
                        if row['interest'] > interest_threshold:
                            fig.add_annotation(
                                x=row['mean_stance'] + 0.25,
                                y=row['interest'] -0.052,
                                text=str(row['username']), 
                                row=body_index,
                                col=party_index, 
                                bgcolor= 'rgb(250,250,250)',
                                bordercolor = 'rgb(0,0,0)',
                                standoff= 100
                            )
            if party == 'Republican':
                fig.add_trace(go.Scatter(x=[-2,2],y= [2,2],fill='tozeroy', fillcolor= 'rgba(255, 134, 128,0.1)', showlegend = False), row=body_index, col=party_index)
            if party == 'Democratic':    
                fig.add_trace(go.Scatter(x=[-2,2],y= [2,2],fill='tozeroy', fillcolor= 'rgba(154, 188, 244,0.15)', showlegend = False), row=body_index, col=party_index)

            
    # Update figure for better visual appearance
    fig.update_layout(
        width=1100,  
        height=800, 
        coloraxis_colorscale= [[0.0, "#FF1206"],  [0.5, '#B3AFAF'],  [1.0, "#3663AD" ]], #,[1.0,'#30823F']],
        # Fit title to plot
        # Fiddle with margins to make it look better
        margin=dict(l=100, r=10, t=80, b=60),
        plot_bgcolor = 'rgba(220, 225, 221, 0.5)' ,
        #title={'text': 'Average Stance vs. Interest','font_size': 20 ,"x": 0.5},
        ) 
        
    fig.update_layout(
        coloraxis=dict(colorbar=dict(
            x=1.15,  # Adjust the x position of the legend
            y=1.0,  # Adjust the y position of the legend
            xanchor='right',  # Set x anchor to the right side of legend
            yanchor='top'  # Set y anchor to the top of legend
        ))
    )

    return fig

def party_daily_stance_dispersion_eng(df = pd.DataFrame, metadata = pd.DataFrame, filter = 'All'): 

    date_list = df['date'].drop_duplicates()

    def get_latest_position(date):
        date = str(date)
        # Filter and process data for each date
        temp = df[~df['conflict_sentiment_rank'].isna()]
        temp = temp[temp['date'].astype(str) <= date]

        # Group, find the max upload_date for each user, and join with metadata
        latest_tweet_date = temp.groupby('username').agg({'upload_date': 'max'}).reset_index()
        latest_tweets = latest_tweet_date.merge(temp, on=['username', 'upload_date'])
        latest_tweets_with_metadata = latest_tweets.merge(metadata, on='username')
        latest = latest_tweets_with_metadata[latest_tweets_with_metadata['legislation_body'] != 'government']

        # Count the occurrences
        results = (latest.groupby(['conflict_sentiment_chr', 'party', 'legislation_body'])
                .size().reset_index(name='count'))
        results['date'] = date
        return results

    # Apply the function for each date and concatenate results
    latest_position = pd.concat([get_latest_position(date) for date in date_list])
    latest_position['conflict_sentiment_chr'] = latest_position['conflict_sentiment_chr'].str.replace("pro_israeli", "Pro Israeli").str.replace("pro_palestinian", "Pro Palestinian").str.replace("complex_attitude", "Complex Stance")

    if filter == 'All':
        # Create figure with 4 subplots
        fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.15, vertical_spacing=0.1,
                            subplot_titles=('Democratic Party - Senate', 'Republican Party - Senate', 'Democratic Party - House', 'Republican Party - House' ))

        # Set party and legislation_body lists for iteration 
        parties = ['Democratic', 'Republican']
        legislation_bodies = ['Senate', 'House of Representatives']

        # Create a subplot for each party in each body
        for party_index, party in enumerate(parties, start=1):
            for body_index, body in enumerate(legislation_bodies, start = 1):

                # Filter data
                party_data = latest_position[latest_position['party'] == party]
                body_data = party_data[party_data['legislation_body'] == body]
                
                sorted_body_data = body_data.sort_values('count', ascending=False)
                # Create subplot
                fig_party_body = px.bar(sorted_body_data, x= 'date', y = 'count',  barmode = 'stack',
                                        color = 'conflict_sentiment_chr' , color_discrete_map = {'Pro Palestinian': '#d62728','Complex Stance': 'goldenrod','Pro Israeli': 'forestgreen' } )
                
                # Update figure
                for trace in fig_party_body['data']:
                    if party == "Democratic":
                        trace.update(showlegend=False)
                    elif body == 'Senate': 
                        trace.update(showlegend=False)
                    else: pass
                
                    fig.add_trace(trace, row=body_index, col=party_index)

        fig.update_layout(
            barmode = 'stack',
            width=1000,  
            height=800, 
            # Change colorscale into red, gray and blue
            # Fit title to plot
            title={'text': 'Stance Dispersion by day ','font_size': 20 ,"x": 0.5},
            # Fiddle with margins to make it look better
            margin=dict(l=100, r=20, t=80, b=60),
            plot_bgcolor = 'rgb(240,240,240)' 
        )
        return fig
    
    else: 
        if filter == 'Democratic': 
                party = 'Democratic'
        elif filter == 'Republican': 
                party = 'Republican'
    
        party_data = latest_position[latest_position['party'] == party]
        # Order data
        sorted_party_data = party_data.sort_values('count', ascending=False)
        # Create Graph
        fig_party = px.bar(sorted_party_data, x= 'date', y = 'count',  barmode = 'stack',
                                color = 'conflict_sentiment_chr' , color_discrete_map = {'Pro Palestinian': '#d62728','Complex Stance': 'goldenrod','Pro Israeli': 'forestgreen' } )
        fig_party.update_layout( title={'text': '{} Stance Dispersion by day'.format(party) ,'font_size': 20 ,"x": 0.5}, legend = {"title": ""})

        return fig_party
    

def party_daily_stance_dispersion(df = pd.DataFrame, metadata = pd.DataFrame, filter = 'All'): 

    date_list = df['date'].drop_duplicates()

    def get_latest_position(date):
        date = str(date)
        # Filter and process data for each date
        temp = df[~df['conflict_sentiment_rank'].isna()]
        temp = temp[temp['date'].astype(str) <= date]

        # Group, find the max upload_date for each user, and join with metadata
        latest_tweet_date = temp.groupby('username').agg({'upload_date': 'max'}).reset_index()
        latest_tweets = latest_tweet_date.merge(temp, on=['username', 'upload_date'])
        latest_tweets_with_metadata = latest_tweets.merge(metadata, on='username')
        latest = latest_tweets_with_metadata[latest_tweets_with_metadata['legislation_body'] != 'government']

        # Count the occurrences
        results = (latest.groupby(['conflict_sentiment_chr', 'party', 'legislation_body'])
                .size().reset_index(name='count'))
        results['date'] = date
        return results

    # Apply the function for each date and concatenate results
    latest_position = pd.concat([get_latest_position(date) for date in date_list])
    latest_position['conflict_sentiment_chr'] = latest_position['conflict_sentiment_chr'].str.replace("pro_israeli", "פרו ישראלי").str.replace("pro_palestinian", "פרו פלסטיני").str.replace("complex_attitude", "עמדה מורכבת")

    if filter == 'All':
        # Create figure with 4 subplots
        fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.15, vertical_spacing=0.1,
                            subplot_titles=('דמוקרטים בסנאט', 'רפובליקנים בסנאט', 'דמוקרטים בבית הנחברים', 'רפובליקנים בבית הנבחרים' ))

        fig.update_annotations(font_size=25)

        # Set party and legislation_body lists for iteration 
        parties = ['Democratic', 'Republican']
        legislation_bodies = ['Senate', 'House of Representatives']

        # Create a subplot for each party in each body
        for party_index, party in enumerate(parties, start=1):
            for body_index, body in enumerate(legislation_bodies, start = 1):

                # Filter data
                party_data = latest_position[latest_position['party'] == party]
                body_data = party_data[party_data['legislation_body'] == body]
                
                sorted_body_data = body_data.sort_values('count', ascending=False)
                # Create subplot
                fig_party_body = px.bar(sorted_body_data, x= 'date', y = 'count',  barmode = 'stack',
                                        color = 'conflict_sentiment_chr' , color_discrete_map = {'פרו פלסטיני': '#d62728','עמדה מורכבת': 'goldenrod','פרו ישראלי': 'forestgreen' } )
                
                # Update figure
                for trace in fig_party_body['data']:
                    if party == "Democratic":
                        trace.update(showlegend=False)
                    elif body == 'Senate': 
                        trace.update(showlegend=False)
                    else: pass
                
                    fig.add_trace(trace, row=body_index, col=party_index)

        fig.update_layout(
            barmode = 'stack',
            width=1000,  
            height=800, 
            # Change colorscale into red, gray and blue
            # Fit title to plot
            title={'text': 'פילוג עמדות בחלוקה ליום','font_size': 20 ,"x": 0.5},
            # Fiddle with margins to make it look better
            margin=dict(l=100, r=20, t=80, b=60),
            plot_bgcolor = 'rgb(240,240,240)' 
        )
        return fig
    
    else: 
        if filter == 'Democratic': 
                party = 'Democratic'
                party_heb = 'המפלגת הדמוקרטית'
        elif filter == 'Republican': 
                party = 'Republican'
                party_heb = 'המפלגת הרפובליקנית'
    
        party_data = latest_position[latest_position['party'] == party]
        # Order data
        sorted_party_data = party_data.sort_values('count', ascending=False)
        # Create Graph
        fig_party = px.bar(sorted_party_data, x= 'date', y = 'count',  barmode = 'stack',
                                color = 'conflict_sentiment_chr' , color_discrete_map = {'פרו פלסטיני': '#d62728','עמדה מורכבת': 'goldenrod','פרו ישראלי': 'forestgreen' } )
        fig_party.update_layout( title={'text': 'פילוג עמדות של {} בחלוקה לימים'.format(party_heb) ,'font_size': 20 ,"x": 0.5}, legend = {"title": ""})

        return fig_party

def run_graphs(df = pd.DataFrame ,
                metadata = pd.DataFrame, 
                run_daily_tweets_graph = True,
                run_daily_users_conflict_graph = True, 
                run_conflict_relation_graph = True, 
                run_total_stance_graph = True, 
                run_stance_daily_tweets_amount = True, 
                run_stance_freq_by_party = True, 
                run_party_interest_over_time = True, 
                run_party_stance_vs_interest = True,
                run_party_daily_stance_dispersion = True, 
                run_democrat_daily_stance_dispersion = True, 
                run_republican_daily_stance_dispersion = True):
    
    date_to_log = date.today()

    if  run_daily_tweets_graph == True:
        fig = daily_tweets_graph(df)
        fig.write_image('Netivot/notebook/output_graphs/daily_tweets_graph_{}.png'.format(date_to_log), 'png', engine = 'kaleido')

    if  run_daily_users_conflict_graph == True: 
        fig = daily_users_conflict_graph(df)
        fig.write_image('Netivot/notebook/output_graphs/daily_users_conflict_graph_{}.png'.format(date_to_log), 'png', engine = 'kaleido')

    if  run_conflict_relation_graph == True: 
        fig = conflict_relation_graph(df)
        fig.write_image('Netivot/notebook/output_graphs/conflict_relation_graph_{}.png'.format(date_to_log), 'png', engine = 'kaleido')

    if  run_total_stance_graph == True: 
        fig = total_stance_graph(df)
        fig.write_image('Netivot/notebook/output_graphs/total_stance_graph_{}.png'.format(date_to_log), 'png', engine = 'kaleido')

    if  run_stance_daily_tweets_amount == True: 
        fig = stance_daily_tweets_amount(df)
        fig.write_image('Netivot/notebook/output_graphs/stance_daily_tweets_amount_{}.png'.format(date_to_log), 'png', engine = 'kaleido')

    if  run_stance_freq_by_party == True: 
        fig = stance_freq_by_party(df, metadata)
        fig.write_image('Netivot/notebook/output_graphs/stance_freq_by_party_{}.png'.format(date_to_log), 'png', engine = 'kaleido')

    if  run_party_interest_over_time == True: 
        fig = party_interest_over_time(df, metadata)
        fig.write_image('Netivot/notebook/output_graphs/party_interest_over_time_{}.png'.format(date_to_log), 'png', engine = 'kaleido')

    if  run_party_stance_vs_interest == True:
        fig = party_stance_vs_interest(df, metadata)
        fig.write_image('Netivot/notebook/output_graphs/party_stance_vs_interest_{}.png'.format(date_to_log), 'png', engine = 'kaleido')

    if  run_party_daily_stance_dispersion == True:
        fig = party_daily_stance_dispersion(df, metadata)
        fig.write_image('Netivot/notebook/output_graphs/party_daily_stance_dispersion_{}.png'.format(date_to_log), 'png', engine = 'kaleido')
    if  run_democrat_daily_stance_dispersion == True:
        fig = party_daily_stance_dispersion(df, metadata, 'Democratic')
        fig.write_image('Netivot/notebook/output_graphs/democrat_daily_stance_dispersion_{}.png'.format(date_to_log), 'png', engine = 'kaleido')
    if  run_republican_daily_stance_dispersion == True:
        fig = party_daily_stance_dispersion(df, metadata, 'Republican')
        fig.write_image('Netivot/notebook/output_graphs/republican_daily_stance_dispersion_{}.png'.format(date_to_log), 'png', engine = 'kaleido')

    return print('Saved Graphs as Images')