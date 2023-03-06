# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import numpy as np
import streamlit as st 
from pickle import load
from sklearn.preprocessing import RobustScaler
st.set_option('deprecation.showPyplotGlobalUse', False)

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns








data=pd.read_csv("clean_data.csv")
#array = data.values

X = data.iloc[:,2:] #array[:, 2:]
scaler = RobustScaler()
scaler.fit(X)
df = pd.DataFrame(X)
loaded_model = load(open('xgbclf.sav', 'rb'))












nav = st.sidebar.radio("select Below",["EDA","Predict Churn"])


if nav == "Predict Churn":
    
    st.header("![Alt Text](https://media.giphy.com/media/KfshcG42M0zaDXUCxB/giphy.gif)")
    
    
         
    # creating a function for Prediction
    
    def churn_prediction(input_data):
     
     
         # changing the input_data to numpy array
         #input_data_as_numpy_array = np.asarray(input_data)
         new_input = pd.DataFrame([input_data])
         
         # reshape the array as we are predicting for one instance
         #input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
         
         input_data_reshaped=scaler.transform(new_input)#(input_data_reshaped)
         new_data = pd.DataFrame(input_data_reshaped)
         prediction = loaded_model.predict(new_data)
         print(prediction)
         
         y_pred = loaded_model.predict_proba(input_data_reshaped)[:,1]
         churn_probs = y_pred[:1]*100
         
         
         if (prediction == 1):
           #return 'Churn '
           st.write("Probability of Churn is", round(churn_probs[0],2),"%" )
           st.error("Hence Customer will churn :thumbsdown: ")
         else:
           #return 'Did not Churn '
           st.write("Probability of Churn is", round(churn_probs[0],2),"%")
           st.success("Hence customer will not churn :thumbsup: ")
                      
     
    
                    
    def main():
     
        # giving a title
        st.title('Customer churn prediction: XGBoost Model')
        
        
        # getting the input data from the user
        
        
        number1 = st.number_input('Insert  Duration of Account')
        
        strnumber2 = st.radio('Insert Voice Plan', ['Yes','No'])
        if strnumber2 == 'Yes':
            number2     = 1
        else:
            number2     = 0
        
        number3 = st.number_input('Insert  no. of Voice Messages')
        
        
        strnumber4 = st.radio('Insert International Plan', ['Yes','No'])
        if strnumber4 == 'Yes':
            number4     = 1
        else:
            number4     = 0
        
        
        number5 = st.number_input('Insert  International Minutes')
        number6 = st.number_input('Insert  Total International Calls')
        number7 = st.number_input('Insert  Total International Charge')
        
        number8 = st.number_input('Insert  Total number of calls during the day')
        number9 = st.number_input('Insert  Total number of calls during the evening')
        number10 = st.number_input('Insert Total number of calls during the night')
        number11 = st.number_input('Insert Number of calls to customer service')
        number12 = st.number_input('Insert Total_Charge')
        
        
        
        #     # code for Prediction
        diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Churn Result'):
            diagnosis = churn_prediction([number1, number2, number3,              number4,number5,number6,number7,number8,number9,number10,number11,number12])
        
        
        #st.success(diagnosis)
        
    if __name__ == '__main__':
        main()
        
if nav == "EDA":
    
     
    def main():
	    """Semi Automated ML App with Streamlit """

	    activities = ["EDA","Plots"]	
	    choice = st.sidebar.selectbox("Select Activities",activities)

	    if choice == 'EDA':
		    st.subheader("Exploratory Data Analysis")

		    data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		    if data is not None:
			     df = pd.read_csv(data)
			     st.dataframe(df.head())
			

			     if st.checkbox("Show Shape"):
				     st.write(df.shape)

			     if st.checkbox("Show Columns"):
				     all_columns = df.columns.to_list()
				     st.write(all_columns)

			     if st.checkbox("Summary"):
				     st.write(df.describe())

			     if st.checkbox("Show Selected Columns"):
				     selected_columns = st.multiselect("Select Columns",all_columns)
				     new_df = df[selected_columns]
				     st.dataframe(new_df)

			     if st.checkbox("Show Value Counts"):
				     st.write(df.iloc[:,-1].value_counts())

			     if st.checkbox("Correlation Plot(Matplotlib)"):
				     plt.matshow(df.corr())
				     st.pyplot()

			     if st.checkbox("Correlation Plot(Seaborn)"):
				     st.write(sns.heatmap(df.corr(),annot=True))
				     st.pyplot()


			     if st.checkbox("Pie Plot"):
				     all_columns = df.columns.to_list()
				     column_to_plot = st.selectbox("Select 1 Column",all_columns)
				     pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				     st.write(pie_plot)
				     st.pyplot()

	    elif choice == 'Plots':
		    st.subheader("Data Visualization")
		    data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		    if data is not None:
			    df = pd.read_csv(data)
			    st.dataframe(df.head())

			    if st.checkbox("Show Value Counts"):
				    st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
				    st.pyplot()
		
			    # Customizable Plot

			    all_columns_names = df.columns.tolist()
			    type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
			    selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			    if st.button("Generate Plot"):
				    st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
				    if type_of_plot == 'area':
					    cust_data = df[selected_columns_names]
					    st.area_chart(cust_data)

				    elif type_of_plot == 'bar':
					    cust_data = df[selected_columns_names]
					    st.bar_chart(cust_data)

				    elif type_of_plot == 'line':
					    cust_data = df[selected_columns_names]
					    st.line_chart(cust_data)
                   
				# Custom Plot 
				    elif type_of_plot:
					    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					    st.write(cust_plot)
					    st.pyplot()

    if __name__ == '__main__':
	    main()

    
    
    
    
  
    

    


