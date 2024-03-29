import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
import pingouin as pg
from scipy.stats import kruskal
pd.options.display.float_format = '{:.2f}'.format 




st.set_page_config(layout="wide", page_title="Digtal Crope Yield", page_icon= '🌱')
# Read the data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')
                                                                          

# Concatenate the data to over view what's going on ...
train_df['Set'] = 'Train'
test_df['Set'] = 'Test'
df = pd.concat([train_df , test_df])

# file with discribtion for each fatuer
var_def = pd.read_csv('VariableDescription.csv')
# Change the index of the VariableDescription file ,, to esay access leater...
var_def.set_index('Variable' , inplace=True)



# Date time features 
date_col = [x for x in list(df.columns) if str(x).endswith('ate') ] + ['SeedingSowingTransplanting']
# Convert to datetime
for feat in date_col:
    df[feat] = pd.to_datetime(df[feat])

# Extract seasons

def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'
    
season = []
for feat in date_col:
    # Apply function to 'Date' column and create new 'Season' column
    df[f'{feat}_Season'] = df[feat].dt.month.apply(lambda x: get_season(x))
    season += [f'{feat}_Season']
# divide to Zones
def assign_value(District):
    if District == 'Nalanda' or District == 'Gaya':
        return 'zone1'
    elif District == 'Jamui' :
        return 'zone2'
    elif District == 'Vaishali':
        return 'zone3'
    
df['Zone'] = df['District'].apply(assign_value)
train_df['Zone'] = train_df['District'].apply(assign_value)


# Catogrical columns 
cat_col = df.select_dtypes(include=['O' , 'bool']).columns.to_list()


# Numerical featuers 
num_col = df.select_dtypes(exclude=['O' , 'bool' , 'datetime64[ns]']).columns.to_list()

# Function to calculate outliers using IQR method
def detect_outliers_iqr(series, threshold=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (series < lower_bound) | (series > upper_bound)
# Calculate outliers for each column
outliers = df[num_col].apply(detect_outliers_iqr)

# Total number of outliers in the DataFrame
total_outliers = outliers.sum().sum()


# Ther is a smalle confusion about Gurua block matching with Jamui district
# df.loc[(df["District"] == 'Jamui') & (df["Block"] == 'Gurua')].index
df.loc[2177 , "District"] = "Gaya"

# cut off the max for those features
df = df[df['CultLand'] <= 250]
df = df[df['CropCultLand'] <= 250]
df = df[df['SeedlingsPerPit'] != 442]
df = df[(df['TransplantingIrrigationHours'] != 2000.0) & (df['TransplantingIrrigationHours'] != 1000)]
df = df[df['TransIrriCost'] != 6000.0]
df = df[df['1tdUrea'] != 120]
df = df[df['1appDaysUrea'] != 332.0]


def out_iqr(df , column):
    global lower,upper
    q25, q75 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q75 - q25
    cut_off = iqr * 1.96
    lower, upper = q25 - cut_off, q75 + cut_off
    df = df[(df[column] > lower) & (df[column] < upper)]
    return   df 
df = out_iqr(df , 'Yield')
df = out_iqr(df , 'Acre')

# Correlatede features 
def find_highly_correlated_features(data, threshold=0.6):
    """
    Find and display highly correlated features in a dataset.

    Parameters:
    - data: pandas DataFrame
    - threshold: correlation threshold (default is 0.8)

    Returns:
    - List of tuples representing highly correlated feature pairs
    """
    corr_matrix = data.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation above the threshold
    correlated_features = [(col1, col2) for col1 in upper_triangle.columns for col2 in upper_triangle.columns if upper_triangle.loc[col1, col2] > threshold]


    return correlated_features


##################################################################################################
# The Bage

# Title
st.title("Digtal Crope Estimation Problem ")

# Introduction 
st.header("Introduction")
st.markdown("## Problem statment")
st.markdown("#### Digtal crope yield estmation in India ")
st.write("""* Smallholder farmers are crucial contributors to global food production,
          and in India often suffer most from poverty and malnutrition.
          These farmers face challenges such as limited access to modern agriculture, unpredictable weather, and resource constraints. 
          To tackle this issue, Digital Green collected data via surveys,
          offering insights into farming practices, environmental conditions, and crop yields.""")
st.write("""* **The objective of this challenge is to create a machine learning solution to predict the crop yield per acre of rice or wheat crops in India.
          Our goal is to empower these farmers and break the cycle of poverty and malnutrition.**""")
# The data
st.subheader("Data")
st.write("""* *The data was collected through a survey conducted across multiple districts in India. 
         It consists of a variety of factors that could potentially impact the yield of rice crops.*""")
st.write("""* The data source is [Digital Green](https://digitalgreen.org/),
          Which hosted the data as a competition in [zindi platform](https://zindi.africa/competitions/digital-green-crop-yield-estimate-challenge/data).""") 

st.write("* Data structure '.csv' files")
st.write("* Data Shape:", df.shape)

c1 ,c2 = st.columns(2)
with c1 : 
    # check box 
    if st.checkbox("Variables defintion ") :
        st.write(var_def)
with c2 :
    if st.checkbox("Varibles type:") :
            st.write(df.dtypes)

if st.checkbox("DataFrame:") :
            st.write(df.head())



# Workfllow Describtion
st.subheader("Workfllow Describtion")

st.markdown("""#### **The data contain different types of features :**""")


# Date time features 
st.write('Numebr of datetime features : ' , len(date_col))


# Catogrical columns 
st.write('Numebr of catogrical features : ' , len(cat_col))


# Numerical featuers 
st.write('Numebr of numerical features : ' , len(num_col))

st.write("Hint: The catogrical data dosen't contain features of ordinal type.")

st.write("""##### 📍 Note:""")
st.write("""* **All what we see here is coming as a result of data analysis,
          dealing with missing data and  dealing with outliers.**""")
st.write("""* You can find all details in this [Notebook](https://github.com/ahmedalharth/Digtal_crope_app/blob/main/Digtal_crop_1.ipynb).""")

c1 , c2 = st.columns(2)
with c1:
      # Missing values
      st.subheader("""**1- Missing value :**""")
      st.markdown("""### I used the following approach:""")
      st.write("* **Drop any column with more than '40.0%' of missing values.**")
      st.write("* **Imput the catogrical columns with the mode.**")
      st.write("* **Impute the numerical columns with 'KNNImputer'.**")

with c2:
    # Outliers 

    st.subheader("""**2- Outliers :**""")
    st.markdown("""### I use the following approach:""")
    st.write("* **Detect some extrem data points and through them out**")
    st.write("* **Use the IQR to handel the outliers**")
    st.write("* **Outliers in the target variable 'Yield' with residual plot.**")
    


c1 , c2 = st.columns(2)
with c1:
      # Calculate number of missing values for each column
    missing_values = df.isnull().sum()

    # Calculate number of non-missing values for each column
    non_missing_values = df.notnull().sum()

    # Create a DataFrame for the pie chart
    pie_data = pd.DataFrame({
        'Column': missing_values.index,
        'Missing': missing_values.values,
        'Non-Missing': non_missing_values.values
    })

    # Melt the DataFrame to create two columns (Missing and Non-Missing)
    pie_data_melted = pie_data.melt(id_vars=['Column'], var_name='Status', value_name='Count')

    # Create a pie chart using Plotly Express
    fig = px.pie(pie_data_melted, names='Status', values='Count', title='Missing vs Non-Missing Values'  , height=300 , width=400)

    # Display the pie chart
    st.plotly_chart(fig)
      
with c2:
    st.image("output.png", caption="Residual Plot"  , width=400)


drop_col = ['2appDaysUrea', '2tdUrea', 'CropOrgFYM', 'Ganaura', 'BasalUrea']
df.drop(drop_col , axis=1 , inplace= True)
# to drop catogrical 
to_drop_ca = ['ID','TransDetFactor','NursDetFactor','LandPreparationMethod',
             'CropbasalFerts','OrgFertilizers','FirstTopDressFert']
df.drop(to_drop_ca , axis=1 , inplace= True)

cat_col=[x for x in cat_col if x not in drop_col]
num_col=[x for x in num_col if x not in drop_col]
cat_col = [x for x in cat_col if x not in to_drop_ca] 



knn = KNNImputer(n_neighbors=5)
knn.fit(train_df[num_col])
df[num_col] = knn.transform(df[num_col])

for feat in cat_col:
    df[feat].fillna(value= df[feat].mode()[0]  , inplace=True)


# Catogrical features
st.markdown("#### After Data cleainig we end up with")
# Date time features 
st.write('Numebr of datetime features : ' , len(date_col))
# Catogrical columns 
st.write('Numebr of catogrical features : ' , len(cat_col))

# Numerical featuers 
st.write('Numebr of numerical features : ' , len(num_col))

# Analysis
st.header("Analysis")
st.subheader("EDA")
st.markdown('* Date time features :')
if st.checkbox("Seasons"):
    c1 ,c2 = st.columns(2)

    with c1 :
        with st.expander("Dsecribtion"):
            st.write(df[date_col].describe().T)
    with c2:
        with st.expander("Defintions"):
            st.write(var_def.loc[date_col])



    st.markdown('I devide the date time features to seasons, to see rice cultivation cycle.')
    cat_col = [x for x in cat_col if x not in season + ['Set']]
    c1 , c2 = st.columns(2)
    with c1:
        st.write('Rice Cultivation seasons ')
        st.image('seasons1.png' , caption='Cultivation seasons ' , use_column_width=True )

    with c2:
        st.write('Rice Harvrest seasons ')
        st.image('seasons2.png' , caption='Harvrest seasons ' , use_column_width=True )

st.markdown("* Catogrical features :")
if st.checkbox("**District, Block and Zone**") :

    c1 ,c2 ,c3 = st.columns(3)

    with c1 :
        with st.expander("Dsecribtion"):
            st.write(df.groupby('District')['Block'].unique())
        st.write('Pie chart')
        st.plotly_chart(px.histogram(df , y='Block' , color='District', height=300 , width=300))
    with c2:
        with st.expander("Defintions"):
            st.write(var_def.loc[['District' , 'Block']])
        st.write('Count plot')
        st.plotly_chart(px.pie(data_frame=df , names='District', height=300 , width=300))
    with c3:
        with st.expander("zone"):
            st.write(df['Zone'].describe())
        st.write('Count plot')
        st.plotly_chart(px.pie(data_frame=df , names='Zone', height=300 , width=300))
if st.checkbox("Transplantation, harvesting and threshing methods."):
    c1 ,c2  = st.columns(2)

    with c1 :
        with st.expander("Dsecribtion"):
            st.write(df[['CropEstMethod' , 'Harv_method' , 'Threshing_method']].describe().T)
        st.plotly_chart(px.histogram(data_frame=df , x='Harv_method' , y='Yield', height=400 , width=400))
        st.plotly_chart(px.histogram(data_frame=df , x='CropEstMethod' , y='Yield', height=400 , width=400))
    with c2:
        with st.expander("Defintions"):
            st.write(var_def.loc[['CropEstMethod' , 'Harv_method' , 'Threshing_method']])
        st.plotly_chart(px.histogram(data_frame=df , x='Threshing_method' , y='Yield', height=400 , width=400))


if st.checkbox("Source of water and Source of power"):
    c1 ,c2 = st.columns(2)
    with c1 :
        with st.expander("Dsecribtion"):
            st.write(df[['TransplantingIrrigationSource', 'TransplantingIrrigationPowerSource']].describe().T)
        st.plotly_chart(px.histogram(data_frame=df , x= 'TransplantingIrrigationSource' , y="Yield" ,width=400 , height=300))
    with c2:
        with st.expander("Defintions"):
            st.write(var_def.loc[['TransplantingIrrigationSource', 'TransplantingIrrigationPowerSource']])
        st.plotly_chart(px.histogram(data_frame=df , x= 'TransplantingIrrigationPowerSource' , y="Yield" ,width=400 , height=300))

if st.checkbox("Methods of fertilization"):
    c1 ,c2  , c3= st.columns(3)
    with c1 :
        with st.expander("Dsecribtion"):
            st.write(df[['PCropSolidOrgFertAppMethod' ,'MineralFertAppMethod' , 'MineralFertAppMethod.1' , 'Stubble_use']].describe().T)
        st.plotly_chart(px.histogram(df , x='PCropSolidOrgFertAppMethod' , y='Yield' , height=300 , width=400 ))
        st.plotly_chart(px.histogram(df , x='MineralFertAppMethod' , y='Yield' , height=300 , width=400 ))

    with c2:
        with st.expander("Defintions"):
            st.write(var_def.loc[['PCropSolidOrgFertAppMethod' ,'MineralFertAppMethod' ,'MineralFertAppMethod.1','Stubble_use']])
        st.plotly_chart(px.histogram(df , x='MineralFertAppMethod.1' , y='Yield' , height=300 , width=400))
        st.plotly_chart(px.histogram(df , x='Stubble_use', y='Yield',  height=300 , width=400))
st.write("**📍 Note: We dealing with imbalance data.**")


st.markdown("* Numerical features :")

feat = st.selectbox('choose a variable to see it\'s distribution' , options= num_col)
st.write(var_def.loc[feat])
st.plotly_chart(px.histogram(data_frame=df , x=feat  , marginal='box' ))

st.write("**📍 Note: We dealing with alot of outliers, eveing after cuting off the extrem values in the data.**")


st.markdown("* *Bi variante* :")
st.write("Correlated features")
c1 , c2 , c3 = st.columns(3)
with c1:
    r1 = df['CultLand'].corr(df['CropCultLand'])
    st.plotly_chart(px.scatter(data_frame=df , x='CultLand', y='CropCultLand', width=350 , title=f"r= {r1:.2f}")) 
with c2:
    r3 = df['Yield'].corr(df['Acre'])
    st.plotly_chart(px.scatter(data_frame=df , x='Acre', y='Yield' , width=350, title=f"r= {r3:.2f}"))
with c3:
    r2 = df['BasalDAP'].corr(df['1tdUrea'])
    st.plotly_chart(px.scatter(data_frame=df , x='BasalDAP', y= '1tdUrea', width=350, title=f"r= {r2:.2f}"))

st.write("""**📍 Note: Outliers can be saw in this correlated features.**""")


st.subheader("Infrence")
st.write("* **In this section we gonna use some statistics, hypothesis testing**")
st.write("* **We will us noneparametric tests , due the data is not normaly distribuit**")
st.markdown("<h6>1- Mann_wetny U test<h6/>" , unsafe_allow_html=True)
st.write("""* Null hypothesis (H0): There is no difference between the distributions of the two groups.
* Alternative hypothesis (H1): There is a difference between the distributions of the two groups.""")

c1 , c2 ,c3 = st.columns(3)
with c1:
    # 1- Harv_method
    st.write("**Harv_method ('hand' , 'machine')**")
    group1 = train_df[train_df['Harv_method']=="hand"]
    st.write("- Group1 n=",group1.shape[0])
    group2 = train_df[train_df['Harv_method']=="machine"]
    st.write("- Group2 n=",group2.shape[0])
    st.write("H0: Group1[Yield]median = Group2[Yeild]median")
    st.write("H1: Group1[Yield]median < Group2[Yeild]median")

    st.write(pg.mwu(group1["Yield"], group2["Yield"],alternative='less').T)
    st.write("""* Descision : sice P-value < 0.05 level of significant , we rejct H0 and conclud that 
             the median of yield given harv_method == 'hand' is less than the median of yield given harv_method == 'machine' """)
with c2:
    # 1- Threshing_method
    st.write("**Threshing_method ('hand' , 'machine')**")
    group1 = train_df[train_df['Threshing_method']=="hand"]
    st.write("- Group1 n=",group1.shape[0])
    group2 = train_df[train_df['Threshing_method']=="machine"]
    st.write("- Group2 n=",group2.shape[0])

    st.write("H0: Group1[Yield]median = Group2[Yeild]median")
    st.write("H1: Group1[Yield]median < Group2[Yeild]median")
    test = pg.mwu(group1["Yield"], group2["Yield"]).T
    st.dataframe(pg.mwu(group1["Yield"], group2["Yield"],alternative='less'))
    st.write("""* Descision : sice P-value < 0.05 level of significant , we rejct H0 and conclud that 
             the median of yield given Threshing_method == 'hand' is less than the median of yield given Threshing_method == 'machine'""")    

with c3:
    # 1- Stubble_use
    st.write("**Threshing_method ('hand' , 'machine')**")
    group1 = train_df[train_df['Stubble_use']=="plowed_in_soil"]
    st.write("- Group1 n=",group1.shape[0])
    group2 = train_df[train_df['Stubble_use']=="burned"]
    st.write("- Group2 n=",group2.shape[0])

    st.write("H0: Group1[Yield]median = Group2[Yeild]median")
    st.write("H1: Group1[Yield]median ≠ Group2[Yeild]median")

    st.write(pg.mwu(group1["Yield"], group2["Yield"],alternative='two-sided').T)
    st.write("""* Descision : sice P-value < 0.05 level of significant , we rejct H0 and conclud that 
             the median of yield given Threshing_method == 'plowed_in_soil' not equal the median of yield given Threshing_method == 'burned'""")

st.markdown("<h6>2- Kruskal-Wallis test<h6/>" , unsafe_allow_html=True)
st.write("""* Null hypothesis (H0): There is no difference in the medians of the groups being compared.
* Alternative hypothesis (H1): There is a difference in the medians of the groups being compared.""")

st.write("**Zone ('zone1', 'zone2', 'zone3')**")
group1 = train_df[train_df['Zone']=="zone1"]
st.write("- Group1 n=",group1.shape[0])
group2 = train_df[train_df['Zone']=="zone2"]
st.write("- Group2 n=",group2.shape[0])
group3 = train_df[train_df['Zone']=="zone3"]
st.write("- Group3 n=",group3.shape[0])
# Assuming group1, group2, and group3 are pandas DataFrames with a column named "Yield"
result = kruskal(group1["Yield"], group2["Yield"], group3["Yield"])

st.write("Kruskal-Wallis test statistic:", result.statistic)
st.write("p-value:", result.pvalue)
st.write("""* Descision : since p_value < 0.05 level of significant , we rejct H0 and conclud that
         there is a difference in the Yield medians of the groups being compared.""")

st.subheader("Insights")
st.markdown("**I well show the foundings and recommendations :**")
st.write("* The date time features are useless, we can use the seasons and the different between featues in days.")
st.write("* The catogrical columns as we see are imbalance, may be we can use resample techniques.")
st.write("* The numerical features contains alot of outliers and the are very skewed , but it's the nature of this district in rice Yield")

# Sidebar

# Add elements to the sidebar
st.sidebar.title("Content")

st.sidebar.markdown(" # Introduction")

st.sidebar.write(" * Problem statment ")
st.sidebar.write(" * Data ")
st.sidebar.write(" * Workfllow Describtion ")


st.sidebar.markdown(" # Analysis")
st.sidebar.write(" * EDA ")
st.sidebar.write(" * Infrence  ")

st.sidebar.markdown(" # insghts")


# # Define the subjects
# subjects = ["Subject 1", "Subject 2", "Subject 3"]

# # Create a sidebar with subjects as options
# selected_subject = st.sidebar.selectbox("Select a Subject", subjects)

# # Define the content for each subject
# subject_content = {
#     "Subject 1": "This is the content for Subject 1.",
#     "Subject 2": "This is the content for Subject 2.",
#     "Subject 3": "This is the content for Subject 3."
# }

# # Generate anchor links for each subject
# for subject_name in subject_content.keys():
#     st.sidebar.markdown(f'<a href="#{subject_name}">{subject_name}</a>', unsafe_allow_html=True)

# # Display the content based on the selected subject
# for subject_name, content in subject_content.items():
#     st.markdown(f'<h2 id="{subject_name}">{subject_name}</h2>', unsafe_allow_html=True)



