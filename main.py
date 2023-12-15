import streamlit as st
from helper.calculator import ImageCount
from ai import AI
import numpy as np
import pandas as pd

st.title("CNN Image Classification")

# folder_train = 'datasets/training'
# folder_test = 'datasets/test'

with st.expander("See EDA"):
    st.text("Example images for each category:")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text("cat")
        st.image("datasets/training/cat/cat118.jpg")
    with col2:
        st.text("cow")
        st.image("datasets/training/cow/cow90.jpg")
    with col3:
        st.text("dog")
        st.image("datasets/training/dog/dog201.jpg")
    with col4:
        st.text("horse")
        st.image("datasets/training/horse/horse140.jpg")
    with col5:
        st.text("squirrel")
        st.image("datasets/training/squirrel/squirrel135.jpg")

    st.text("Amount of images in each category (training):")
    calc_1 = ImageCount("datasets/training").count()
    labels = ["cat", "cow", "dog", "horse", "squirrel"]
    st.bar_chart(calc_1)

    st.text("Amount of images in each category (test):")
    calc_2 = ImageCount("datasets/test").count()
    st.bar_chart(calc_2)


st.header("Training the model")
epochs = st.slider("Epochs", 1, 10, 1)
dropout_rate = st.slider("Dropout rate", 0.0, 1.0, 0.1)

import altair as alt

if st.button("Train"):
    ai = AI()
    
    with st.spinner("Training the model..."):
        model, history = ai.create_model(epochs, dropout_rate)

    st.success("Model trained!")
    
    # Create a pandas DataFrame for loss
    df_loss = pd.DataFrame({
        'epoch': range(len(history.history['loss'])),
        'Training Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss']
    })
    
    # Create a line chart for loss
    loss_chart = alt.Chart(df_loss.melt('epoch')).mark_line().encode(
        x='epoch',
        y=alt.Y('value', scale=alt.Scale(zero=False)),  # Disable zero-based scale
        color='variable'
    )
    st.altair_chart(loss_chart, use_container_width=True)
    
    # Create a pandas DataFrame for accuracy
    df_accuracy = pd.DataFrame({
        'epoch': range(len(history.history['accuracy'])),
        'Training Accuracy': history.history['accuracy'],
        'Validation Accuracy': history.history['val_accuracy']
    })
    
    # Create a line chart for accuracy
    accuracy_chart = alt.Chart(df_accuracy.melt('epoch')).mark_line().encode(
        x='epoch',
        y=alt.Y('value', scale=alt.Scale(zero=False)),  # Disable zero-based scale
        color='variable'
    )
    st.altair_chart(accuracy_chart, use_container_width=True)
