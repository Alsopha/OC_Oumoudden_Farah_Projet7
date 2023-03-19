import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import gzip
import pickle
import sys
import lime
import lime.lime_tabular
from zipfile import ZipFile
import seaborn as sns
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors._dist_metrics import EuclideanDistance



import plotly.express as px
from sklearn.neighbors import _dist_metrics
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

st.title("Credit Risk Prediction Dashboard :credit_card:")
st.markdown("**A dashboard to understand the factors influencing credit risk predictions**")
st.sidebar.header("Feature Update")
st.sidebar.markdown('This section allows you to update the values of key features to see the impact on the credit risk score.')

@st.cache_data
def load_data_unscaled():
    'Function to load unscaled data'
    df = pd.read_csv('data/unscaled_data_small.csv', encoding ='utf-8')
    return df

@st.cache_data
def load_scaled_data():
    'Function to load scaled data used to retrieve model predictions'
    z = ZipFile("data/train_final.zip")
    df_final = pd.read_csv(z.open('train_final.csv'), encoding ='utf-8')
    return df_final

def load_feature_descriptions():
    'Function to load the feature descriptions'
    df_desc = pd.read_csv('data/feature_descriptions.csv', encoding= 'unicode_escape')
    return df_desc

@st.cache_resource
def import_models():
    'Function to import the Logistic Regression and Nearest Neighbours models'
    model = pickle.load(open('models/LRCSmote.obj','rb'))
    z = ZipFile('models/knn.zip', 'r')
    kdt = pickle.loads(z.open('knn.pkl').read())
    #kdt = pickle.load(r.open('knn.pkl','rb'))
    z.close()
    #model = model1[0]
    return model,kdt

def example_ids():
    'Outputs 5 sample IDs of default and non-default clients'
    sample = df_final['SK_ID_CURR'].sample(5).tolist()
    for i in range(0, len(sample)): 
        sample[i] = int(sample[i])
    st.write("Examples of client IDs:")
    st.write(str(sample).replace('[','').replace(']', ''))

def clean_lime_output(df_exp):
    df_exp_full = pd.DataFrame(columns = ['lower boundary', 'lower boundary sign', 'feature', 'upper boundary', 'upper boundary sign'])
    for i in df_exp[0]:
        if (i.count('>') + i.count('<')) > 1:
            split = i.split(' ')
            low_bound = split[0]
            low_bound_sign = split[1]
            up_bound = split[-1]
            up_bound_sign = split[-2]
            if len(low_bound_sign)>1:
                start = 6
            else:
                start = 7
            if len(up_bound_sign)>1:
                end = -8
            else:
                end = -7
            feature = i[start:end]
            df_exp_full = df_exp_full.append({'lower boundary': low_bound,
                                            'lower boundary sign': low_bound_sign,
                                            'upper boundary': up_bound,
                                            'upper boundary sign': up_bound_sign,
                                            'feature': feature
                                            }, ignore_index=True)
        else:
            split = i.split(' ')
            low_bound = np.nan
            low_bound_sign = np.nan
            up_bound = split[-1]
            up_bound_sign = split[-2]
            if len(up_bound_sign)>1:
                end = -8
            else:
                end = -7
            feature = i[:end]
            df_exp_full = df_exp_full.append({'lower boundary': low_bound,
                                            'lower boundary sign': low_bound_sign,
                                            'upper boundary': up_bound,
                                            'upper boundary sign': up_bound_sign,
                                            'feature': feature
                                            }, ignore_index=True)
    return df_exp_full

@st.cache_data
def filter_dataset(client_id: str, df_final: pd.DataFrame):
    'Filters dataset down to a single line, being the chosen client ID'
    df_small = df_final[df_final['SK_ID_CURR'] == int(client_id)]

    return df_small



@st.cache_resource
def interpretation(client):
    'Visualize the interpretation of the prediction generated by the model'
    # Filter the dataset on a single ID
    X = df_final[df_final['SK_ID_CURR'] == int(client)]
    # Drop columns not used in the training of the model
    X.drop(columns = ['SK_ID_CURR', 'TARGET', 'Unnamed: 0'], inplace = True)
    # Generate dataset without unrequired columns
    df_lime = df_final.drop(columns = ['SK_ID_CURR', 'TARGET', 'Unnamed: 0'])
    # Generate explainer object
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data = np.array(df_lime.sample(int(0.1*df_lime.shape[0]), random_state=20)),feature_names = df_lime.columns,training_labels = df_lime.columns.tolist(),verbose=1,random_state=20,mode='classification')
    # Generate specific instance explainer
    exp = explainer.explain_instance(data_row = X.to_numpy().ravel(), predict_fn = model.predict_proba, num_features = 243)
    # Generate dataframe based on Lime output
    df_exp = pd.DataFrame(exp.as_list())
    df_exp_full = clean_lime_output(df_exp)
    df_exp_full = df_exp_full.merge(df_exp[1], left_index = True, right_index = True)
    df_exp_full['default_risk'] = 'decreases'
    #df_exp_full.loc[df_exp_full['importance']>=0, 'default_risk'] = 'increases'
    df_interpret = df_exp_full
    from sklearn.neighbors import KDTree, DistanceMetric

    # Créer un objet distance_metric avec la distance euclidienne
    dist_metric = DistanceMetric.get_metric('euclidean')
    df_interpret2 = df_interpret.drop(["lower boundary","lower boundary sign","default_risk","upper boundary sign","feature"],axis = 1)

    # Utiliser l'objet distance_metric pour la fonction kdtree query
    kdt = KDTree(df_interpret2, metric=dist_metric)
    # Add the customer's value, global average, non-default avg, default avg, and nearest neighbours avg for each feature
    df_interpret['customer_value'] = [X[feature].mean() for feature in df_interpret['feature'].values.tolist()]
    df_interpret['global_average'] = [df_final[feature].mean() for feature in df_interpret['feature'].values.tolist()]
    df_interpret['non_default_average'] = [df_final[df_final['TARGET'] == 0][feature].mean() for feature in df_interpret['feature'].values.tolist()]
    df_interpret['default_average'] = [df_final[df_final['TARGET'] == 1][feature].mean() for feature in df_interpret['feature'].values.tolist()]
    # Columns considered in determining closest neighbours
    cols = ['DAYS_BIRTH', 'AMT_INCOME_TOTAL']
    neighbours = 20
    # Get the IDs of the n closest neighbours
    dist, ind = kdt.query(np.array(X[cols]).reshape(1,-1), k = neighbours)
    nn_idx = ind[0]
    # Generate a n neighbours dataset
    nn_df = df_final[df_final.index.isin(nn_idx)]
    # Average of nearest neighbours
    df_interpret['similar_clients_average'] = [nn_df[feature].mean() for feature in df_interpret['feature'].values.tolist()]
    return df_interpret

def get_prediction():
    'Retrieve prediction either from the Heroku API or from the locally loaded model'
    use_api = False
    if use_api:
        url = 'https://home-credit-risk.herokuapp.com/predict'
        input_data = df_small.to_dict()
        input_data = json.dumps(input_data)
        input_data = json.loads(input_data)
        # Post JSON file
        r = requests.post(url, json = input_data)
        # Visualize response
        st.markdown('**'+r.json()+'**')
    else:
        df_small1=df_small.drop(["SK_ID_CURR",'TARGET','Unnamed: 0'],axis=1)
        prediction = model.predict_proba(df_small1)[:, 1][0]
        prediction = round(prediction,2)*100
        df_small2=df_small.drop(["SK_ID_CURR",'TARGET'],axis=1)
        knn_prediction = kdt.predict(df_small2)
        st.markdown(f'The client is **{prediction}%** at risk of defaulting.')
        st.markdown(f'The client is **{knn_prediction}%** at risk of defaulting according to K-Nearest Neighbors.')

def get_prediction_update():
    'Retrieve updated prediction either from the Heroku API or from the locally loaded model'
    use_api = False
    if use_api:
        url = 'https://home-credit-risk.herokuapp.com/predict'
        input_data = df_small.to_dict()
        input_data = json.dumps(input_data)
        input_data = json.loads(input_data)
        # Post JSON file
        r = requests.post(url, json = input_data)
        # Visualize response
        st.sidebar.markdown('**'+r.json()+'**')
    else:
        df_small1=df_small.drop(["SK_ID_CURR",'TARGET','Unnamed: 0'],axis=1)
        prediction = model.predict_proba(df_small1)[:, 1][0]
        prediction = round(prediction,2)*100
        df_small2=df_small.drop(["SK_ID_CURR",'TARGET'],axis=1)
        knn_prediction = kdt.predict(df_small2)
        st.sidebar.markdown(f'The client is **{prediction}%** at risk of defaulting.')
        st.markdown(f'The client is **{knn_prediction}%** at risk of defaulting according to K-Nearest Neighbors.')

def top_20_credit_requests():
    'Generates a graph showing the 20 largest credit requests'
    # Create table
    credit_df = df.pivot_table('AMT_CREDIT', 'SK_ID_CURR').sort_values(by = 'AMT_CREDIT', ascending = False).head(20)
    y_pos = credit_df.sort_values(by = 'AMT_CREDIT').index.astype(str)
    credit_amount = credit_df.sort_values(by = 'AMT_CREDIT')['AMT_CREDIT']
    # Create graph
    fig = px.bar(x=credit_amount, 
                y=y_pos, 
                orientation='h', 
                labels = dict(x = "Credit Amount", y = "Client ID"), 
                title = "Clients with the highest credit requests",
                width = 800,
                height=600)
    fig.update_yaxes(type='category')
    st.plotly_chart(fig)

def target_amounts():
    'Generates a graph showing the number of people that have default compared to not defaulted'
    # Create table
    default_df = df.pivot_table('SK_ID_CURR', 'TARGET', aggfunc = 'count').reset_index()
    labels = ['Not in default', 'Default']
    # Create graph
    fig = px.pie(default_df, values='SK_ID_CURR',
                names=labels, 
                title='Proportion of clients defaulting vs clients not defaulting',
                hover_data = ['SK_ID_CURR'],
                labels={'SK_ID_CURR': 'Number of Clients'}
                )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)

# Running the dashboard

# Load data and models
df = load_data_unscaled()
df_final = load_scaled_data()
df_desc = load_feature_descriptions()
model, kdt= import_models()
valid_ids = df['SK_ID_CURR'].values.tolist()

#Choose the dashboard page to be displayed
dashboard_page = ''
st.markdown('* The **Global** dashboard provides general information about loans at the Home Credit Group.')
st.markdown('* The **Client-specific** dashboard lets you get a credit risk prediction for a specific client and provides insights into the prediction.')
dashboard_page = st.selectbox('Please select a dashboard below:', options = ['','Global Dashboard', 'Client-Specific Dashboard'])
if dashboard_page == '':
    pass
elif dashboard_page == 'Global Dashboard':
    top_20_credit_requests()
    target_amounts()
else:
    # Print out sample IDs
    example_ids()
    # Get client ID
    client_id = st.text_input('Client ID:')
    if client_id == '':
        st.write('Please enter a Client ID.')
    else:
        if int(client_id) not in valid_ids:
            st.markdown(':exclamation: This ID does not exist. Please enter a valid one.')
        else:
            # Generate filtered dataset
            df_small = filter_dataset(client_id, df_final)
            # Get prediction
            get_prediction()
            with st.spinner('Loading prediction details. This may take a few minutes.'):
                cli = int(client_id)
                df_interpret = interpretation(cli)
                df_interpret = pd.merge(df_interpret,df_desc,left_on='feature', right_on='Row', how='left')
                df_interpret.drop(columns = ['Unnamed: 0', 'Row'], inplace = True)
            st.text("")
            st.text("")
            st.header("Prediction Interpretation")
            st.write("Below are the five features that have the most impact on this client's credit risk prediction, with comparisons to 4 key groups:")
            st.markdown("""
                        * **Average**: Average feature value across all customers 
                    \n* **Avg Non-Default**: Average feature value for clients who have **not** defaulted on their loan
                    \n* **Avg Default**: Average feature value for clients who have defaulted on their loan
                    \n* **Similar Clients**: Average feature value for 20 most similar clients based on age, income, gender, credit length and credit as proportion of income
                    """)
            st.text("")
            # Display written explanation
            for i in range(5):
                row = df_interpret.iloc[i]
                st.subheader(f"{row['feature']}")
                st.markdown(f"_{row['Description']}_")
                if row['lower boundary'] is np.nan:
                    st.markdown(f"When {row['feature']} is {row['upper boundary sign']} {row['upper boundary']}, the risk of the client defaulting **{row['default_risk']}**.")
                else:
                    st.markdown(f"When {row['feature']} is {row['lower boundary sign']} {row['lower boundary']} and {row['upper boundary sign']} {row['upper boundary']}, the risk of the client defaulting **{row['default_risk']}**.")

            # Display graph
                col_df = ['similar_clients_average', 'default_average', 'non_default_average', 'global_average', 'customer_value']
                col_list = ['Similar Clients', 'Avg Default', 'Avg Non-Default', 'Average', 'Target Client']
                fig = px.bar(x = row[col_df], 
                            y = col_list, 
                            orientation='h', 
                            labels = dict(x = row['feature'], y = "Client Group"),
                            color = col_list
                            )
                fig.update_yaxes(type='category')
                st.plotly_chart(fig)
            st.markdown('_Prediction based on Logistic Regression model trained on Home Credit Risk data._')

            # Sidebar - Value update
            features_list = df_interpret['feature'].values.tolist()
            features_list = tuple([''] + features_list)
            ft_to_update = ''
            ft_to_update = st.sidebar.selectbox('Which feature would you like to update', options = features_list)

            if ft_to_update == '':
                default_value = 0
            else:
                st.sidebar.markdown(f"**Description:** _{df_interpret[df_interpret['feature']==ft_to_update]['Description'].values[0]}_")
                default_value = float(df_small[ft_to_update].mean())
                new_value = st.sidebar.slider(ft_to_update, min_value = float(df_final[ft_to_update].min()), max_value = float(df_final[ft_to_update].max()), value = default_value)
                if default_value == new_value:
                    pass
                else:
                    df_small[ft_to_update].values[0] = new_value
                    get_prediction_update()