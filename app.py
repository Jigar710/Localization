import streamlit as st
from collectData import collectData
from train import train as model_train
import pandas as pd
# Define functions for each tab
def collect_data():
    st.title("Collect Data Tab")
    room_name = st.text_input("Room Name", "")
    room_type = st.text_input("Type", "")
    building_name = st.text_input("Building Name", "")

    # Button to submit the collected data
    if st.button("Submit"):
        # Add your code to process and store the collected data
        with st.spinner("We are collecting data...Please wait..."):
            collectData(room_name, room_type, building_name)  
        st.success(f"Data collected: Room Name - {room_name}, Type - {room_type}, Building Name - {building_name}")

def presentation_data():
    st.title("Presentation Data Tab")
    df = pd.read_csv('spaceData.csv')
    st.dataframe(df,width=1500)
    
def train():
    st.title("Train Tab")
    with st.spinner("training updated data...Please wait..."):
        res_df = model_train()
    # Add your code for training the model here
    st.dataframe(res_df)
    st.success("Model trained successfully!")

def predict_data():
    st.title("Predict Data Tab")
    # Add your code for predicting data here

# Main App
def main():
    # st.title("Streamlit App with Tabs")

    # Create tabs
    tabs = ["Collect Data", "Presentation Data", "Train", "Predict Data"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)

    # Display selected tab
    if selected_tab == "Collect Data":
        collect_data()
    elif selected_tab == "Presentation Data":
        presentation_data()
    elif selected_tab == "Train":
        train()
    elif selected_tab == "Predict Data":
        predict_data()

if __name__ == "__main__":
    main()
