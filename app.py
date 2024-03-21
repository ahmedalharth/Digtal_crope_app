import streamlit as st
import pandas as pd
# import plotly.express as px

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
        st.write(var_def)
with c2 :
    if st.checkbox("Varibles type:") :
            st.write(df.dtypes)

if st.checkbox("DataFrame:") :
            st.write(df.head())

st.subheader("Workfllow Describtion")
st.write("""**📍 Note: All what we see here is coming as a result of data analysis,
          dealing with missing data and  dealing with outliers.**""")
st.write("""* You can find all detail in this [Notebook](https://github.com/ahmedalharth/Digtal_crope_app/blob/main/Digtal_crop_1.ipynb). """)
st.markdown("""#### **The data contain different types of features :**""")

# Date time features 
date_col = [x for x in list(df.columns) if str(x).endswith('ate') ] + ['SeedingSowingTransplanting']
st.write('Numebr of datetime features : ' , len(date_col))
# Convert to datetime
for feat in date_col:
    df[feat] = pd.to_datetime(df[feat])

# Catogrical columns 
cat_col = df.select_dtypes(include=['O' , 'bool']).columns.to_list()
st.write('Numebr of catogrical features : ' , len(cat_col))


    # Numerical featuers 
num_col = df.select_dtypes(exclude=['O' , 'bool' , 'datetime64[ns]']).columns.to_list()
st.write('Numebr of numerical features : ' , len(num_col))

st.write("Hint: The catogrical data dosen't contain features of order type.")

# date time features 
st.markdown("#### Datetime features:")
c3 ,c4 = st.columns(2)
with c3:
      st.write(df[date_col].describe().T)

with c4:
      st.write(var_def.loc[date_col])

# Catogrical features
st.markdown("#### Catogrical features:")
st.write("""After analysis we end up with """)






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
# st.image("image.jpg", caption="Streamlit Logo", use_column_width=True)

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

st.sidebar.markdown(" # Analysis")
st.sidebar.write(" * EDA ")
st.sidebar.write(" * Infrence  ")

st.sidebar.markdown(" # insghts")



