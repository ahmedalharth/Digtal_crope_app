import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="HR Case Study ", page_icon=":soccer:")
# df = pd.read_csv('data.csv')
# df['HireDate'] = pd.to_datetime(df['HireDate'])
# df['HireDate_year'] = df['HireDate'].dt.year
# df['HireDate_month'] = df['HireDate'].dt.month
# df['HireDate_day'] = df['HireDate'].dt.day
# df['ReviewDate_temp'] = pd.to_datetime(df['ReviewDate'])
# df['ReviewDate_year'] = df['ReviewDate_temp'].dt.year
# df['ReviewDate_month'] = df['ReviewDate_temp'].dt.month
# df['ReviewDate_day'] = df['ReviewDate_temp'].dt.day
# total_employees=df['EmployeeID'].nunique()
# activeemp = df[df['Attrition'] == 'No']['EmployeeID'].nunique()
# inactive_emp = total_employees - activeemp
# AttritionRate = str(int((inactive_emp / total_employees) * 100))+ ' %'
# tab1,tab2 = st.tabs(['overview' , 'employee'])

# with tab1 :
#     with st.container():
#         c1 , c2, c3,c4  = st.columns(4)
#         c1.metric('total number of employess' , total_employees)
#         c2.metric('activeemp' , activeemp)
#         c3.metric('inactive_emp' , inactive_emp)
#         c4.metric('AttritionRate' , AttritionRate)

#     c1 , c2 = st.columns(2)
#     with c1:
#         x = df.groupby(['HireDate_year' ,'Attrition'])['EmployeeID'].nunique().reset_index()
#         st.plotly_chart(px.bar(x , x = 'HireDate_year' , y = 'EmployeeID' , color = 'Attrition', width=592,height=496))
#     with c2:
#          y= df.groupby(['Department' ,'Attrition'])['EmployeeID'].nunique().reset_index()
#          st.plotly_chart(px.bar(y , y = 'Department' , x = 'EmployeeID' , color = 'Attrition' , orientation ='h', width=500,height=250))
#          m= df.groupby(['State','Attrition'])['EmployeeID'].nunique().reset_index()
#          st.plotly_chart(px.bar(m , y = 'State' , x = 'EmployeeID' , color = 'Attrition' , orientation ='h', width=500,height=250))
# with tab2 :
#     c1, c2,c3,c4 = st.columns(4)
#     with c1:
#         emps = df['EmployeeID'].unique()
#         option = st.selectbox('choose employee',emps)
#         emp = df[df['EmployeeID'] == option].sort_values('ReviewDate')
#         nd = str(emp['ReviewDate_temp'].max() + pd.Timedelta(days=365))
#         name =emp['FirstName'].unique()[0]+ ' ' + emp['LastName'].unique()[0]
#         st.metric('frist review date' , emp['ReviewDate'].min())
#         st.metric('last review date' , emp['ReviewDate'].max())
#         st.metric('next review date',nd[:10] )
#     with c2:
#         st.metric('fullname', str(name))
#         st.plotly_chart(px.line(emp , x='ReviewDate_year' , y='WorkLifeBalance',width=300,height=240))
#         st.plotly_chart(px.line(emp , x='ReviewDate_year' , y='SelfRating',width=300,height=240))

#     with c3:
#         st.metric('AGE', emp['Age'].unique()[0])
#         st.plotly_chart(px.line(emp , x='ReviewDate_year' , y='ManagerRating',width=300,height=240))
#         st.plotly_chart(px.line(emp , x='ReviewDate_year' , y='JobSatisfaction',width=300,height=240))
#     with c4:
#         st.metric('AVG Sal', emp['Salary'].mean())
#         st.plotly_chart(px.line(emp , x='ReviewDate_year' , y='RelationshipSatisfaction',width=300,height=240))
#         st.plotly_chart(px.line(emp , x='ReviewDate_year' , y='EnvironmentSatisfaction',width=300,height=240))
