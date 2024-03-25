import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression



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
df.loc[(df["District"] == 'Jamui') & (df["Block"] == 'Gurua')].index
df.loc[2177 , "District"] = "Gaya"

# cut off the max for those features
df = df[df['CultLand'] != 800]
df = df[df['CropCultLand'] != 800]
df = df[df['SeedlingsPerPit'] != 442]
df = df[(df['TransplantingIrrigationHours'] != 2000.0) & (df['TransplantingIrrigationHours'] != 1000)]
df = df[df['TransIrriCost'] != 6000.0]
df = df[df['1tdUrea'] != 120]
df = df[df['1appDaysUrea'] != 332.0]


def out_iqr(df , column):
    global lower,upper
    q25, q75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)
    iqr = q75 - q25
    cut_off = iqr * 1.96
    lower, upper = q25 - cut_off, q75 + cut_off
    # print('The IQR is',iqr)
    # print('The lower bound value is', lower)
    # print('The upper bound value is', upper)
    # df1 = df[df[column] > upper]
    # df2 = df[df[column] < lower]
    df = df[(df[column] > lower) & (df[column] < upper) ]
    # print('Total number of outliers are', df1.shape[0]+ df2.shape[0])
    return   df 
# df[num_col] = out_iqr(df , 'Yield')
# df[num_col] = out_iqr(df , 'Acre')

# # Fit linear regression model
# model = LinearRegression()
# X , y = df[num_col].drop('Yield' , axis=1) , df.Yield
# model.fit(X, y)

# # Calculate residuals
# residuals = y - model.predict(X)


##################################################################################################

# The Bage

# Title
st.title("Digtal Crope Estimation Problem ")

# Introduction 
st.header("Introduction" )
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
    if st.checkbox("Variables defintion and dtype:") :
        st.write(var_def['dtypes'])
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

st.write("Hint: The catogrical data dosen't contain features of order type.")

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
st.markdown("* Date time features :")
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
st.write("**District and Block**")

c1 ,c2 = st.columns(2)

with c1 :
    with st.expander("Dsecribtion"):
        st.write(df.groupby('District')['Block'].unique())
with c2:
    with st.expander("Defintions"):
         st.write(var_def.loc[['District' , 'Block']])

c1 , c2 = st.columns(2)
with c1:
    st.write('Pie chart')
    st.plotly_chart(px.histogram(df , y='Block' , color='District', height=300 , width=400))


with c2:
    st.write('Count plot')
    st.plotly_chart(px.pie(data_frame=df , names='District', height=300 , width=400))

st.write("Transplantation, harvesting and threshing methods.")
c1 ,c2 = st.columns(2)

with c1 :
    with st.expander("Dsecribtion"):
        st.write(df[['CropEstMethod' , 'Harv_method' , 'Threshing_method']].describe().T)
    st.plotly_chart(px.histogram(data_frame=df , x='Harv_method' , y='Yield', height=400 , width=400))
    st.plotly_chart(px.histogram(data_frame=df , x='CropEstMethod' , y='Yield', height=400 , width=400))
with c2:
    with st.expander("Defintions"):
         st.write(var_def.loc[['CropEstMethod' , 'Harv_method' , 'Threshing_method']])
    st.plotly_chart(px.histogram(data_frame=df , x='Threshing_method' , y='Yield', height=400 , width=400))


st.write("Source of water and Source of power")
c1 ,c2 = st.columns(2)
with c1 :
    with st.expander("Dsecribtion"):
        st.write(df[['TransplantingIrrigationSource', 'TransplantingIrrigationPowerSource']].describe().T)
    st.plotly_chart(px.histogram(data_frame=df , x= 'TransplantingIrrigationSource' , y="Yield" ,width=400 , height=300))
with c2:
    with st.expander("Defintions"):
         st.write(var_def.loc[['TransplantingIrrigationSource', 'TransplantingIrrigationPowerSource']])
    st.plotly_chart(px.histogram(data_frame=df , x= 'TransplantingIrrigationPowerSource' , y="Yield" ,width=400 , height=300))

st.write("Methods of fertilization")
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
st.write("**Acre and Yield**")

c1 ,c2 = st.columns(2)

with c1 :
    with st.expander("Dsecribtion"):
        st.write(df[['Acre', 'Yield']].describe().T)
    st.plotly_chart(px.histogram(data_frame=df , x='Yield' , height=300 , width=400))
with c2:
    with st.expander("Defintions"):
         st.write(var_def.loc[['Acre' , 'Yield']])
    st.plotly_chart(px.histogram(data_frame=df , x='Acre' , height=300 , width=400))
    


     

# Radio button
# radio_button = st.radio("Choose an option", ("Option 1", "Option 2", "Option 3"))
# st.write(f"You selected: {radio_button}")

# # Selectbox
# selectbox = st.selectbox("Choose an option", ("Option 1", "Option 2", "Option 3"))
# st.write(f"You selected: {selectbox}")

# # Multiselect
# multiselect = st.multiselect("Choose options", ("Option 1", "Option 2", "Option 3"))
# st.write(f"You selected: {multiselect}")

# # Slider
# slider = st.slider("Choose a value", 0, 10)
# st.write(f"You selected: {slider}")

# # Text input
# text_input = st.text_input("Enter text")
# st.write(f"You entered: {text_input}")

# # Number input
# number_input = st.number_input("Enter a number", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
# st.write(f"You entered: {number_input}")

# # Text area
# text_area = st.text_area("Enter text here")
# st.write(f"You entered:\n{text_area}")

# # Date input
# date_input = st.date_input("Select a date")
# st.write(f"You selected: {date_input}")

# # Time input
# time_input = st.time_input("Select a time")
# st.write(f"You selected: {time_input}")

# File uploader
# file_uploader = st.file_uploader("Upload a file")
# if file_uploader is not None:
#     st.write(f"You uploaded: {file_uploader.name}")

# Display image
# st.image("output.png", caption="Streamlit Logo", use_column_width=True)

# # Display audio
# audio_file = open("audio.mp3", "rb")
# audio_bytes = audio_file.read()
# st.audio(audio_bytes, format="audio/mp3")

# # Display video
# video_file = open("video.mp4", "rb")
# video_bytes = video_file.read()
# st.video(video_bytes)

# Display data
# import pandas as pd
# df = pd.DataFrame({
#     "Name": ["Alice", "Bob", "Charlie"],
#     "Age": [30, 25, 35]
# })
# st.dataframe(df)

# # Plot data
# import matplotlib.pyplot as plt
# import numpy as np
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# plt.plot(x, y)
# st.pyplot()

# # Add columns
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.write("Column 1")
# with col2:
#     st.write("Column 2")
# with col3:
#     st.write("Column 3")

# Expander
# with st.expander("Click to expand"):
#     st.write("This is inside the expander")

# # Display progress
# import time
# with st.spinner("Wait for it..."):
#     time.sleep(5)
#     st.success("Done!")

# # Display error
# st.error("This is an error")

# # Display warning
# st.warning("This is a warning")

# # Display info
# st.info("This is an info message")

# # Display success
# st.success("This is a success message")

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



