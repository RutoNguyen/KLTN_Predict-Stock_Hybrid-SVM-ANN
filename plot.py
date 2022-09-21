#from cProfile import label

import streamlit as st
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
#import load_data
#DATA_URL = ('C:/Users/nhat0/Documents/TLCN/MyProject/data/VN_Index_Historical_Data(2011-2022)_C.csv')
#df = load_data.data(DATA_URL)


def plot_x1(df,name):
    ##note
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="Raw Data",
        mode="markers+lines", x=df["date"], y=df["value"],
        marker_symbol="star"
    ))
    fig.add_trace(go.Bar(
        name="Middle-aligned",
        x=df["date"], y=df["value"],
        xperiod="M1",
        xperiodalignment="middle"
    ))
    fig.update_xaxes(showgrid=True, ticklabelmode="period")
    fig.show()
    #
    st.plotly_chart(px.scatter(df, x="sepal_width",
                    y="sepal_length", color="species"))
def plot_x(x, name_of_x):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, label=name_of_x)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Price', fontsize=20)
    ax.set_title('Chart of ' + name_of_x, fontsize=25, fontweight='bold')
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    ax.legend()
    st.line_chart(x)
    st.plotly_chart(fig,use_container_width=True,sharing="streamlit")
    st.plotly_chart(px.scatter(x))
    st.pyplot(fig, clear_figure=True)

def plot_x_y(x, name_of_x, y, name_of_y):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, label=name_of_x)
    ax.plot(y, label=name_of_y)
    ax.set_xlabel('Year', fontsize=20)
    #ax.set_ylabel(name_of_x + ', ' + name_of_y, fontsize = 20)
    ax.set_ylabel('Price', fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=10)
    # ax.tick_params(axis='both', which='minor', labelsize=20)
    #plt.xticks(fontsize=14, rotation=90)
    fig.autofmt_xdate(rotation=45)
    ax.set_title('Chart of ' + name_of_x + ' & ' + name_of_y,
                 fontsize=25)  # ,fontweight = 'bold')
    #plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
    ax.grid(color='green', linestyle='--', linewidth=0.5)
    ax.legend()
    st.plotly_chart(fig,use_container_width=True,sharing="streamlit")
    st.pyplot(fig, clear_figure=True)
def plot_x_y_z(x, name_of_x, y, name_of_y, z, name_of_z):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, label=name_of_x)
    ax.plot(y, label=name_of_y)
    ax.plot(z, label=name_of_z)
    ax.set_xlabel('Year', fontsize=20)
    #ax.set_ylabel(name_of_x + ', ' + name_of_y, fontsize = 20)
    ax.set_ylabel('Price', fontsize=20)
    ax.set_title('Chart of ' + name_of_x + ', ' + name_of_y +
                 ' & ' + name_of_z, fontsize=25, fontweight='bold')
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    ax.legend()
    st.plotly_chart(fig,use_container_width=True,sharing="streamlit")
    st.pyplot(fig, clear_figure=True)
    ###


def plot_x_y_z_t(x, name_of_x, y, name_of_y, z, name_of_z, t, name_of_t):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, label=name_of_x)
    ax.plot(y, label=name_of_y)
    ax.plot(z, label=name_of_z)
    ax.plot(t, label=name_of_t)
    ax.set_xlabel('Year', fontsize=20)
    #ax.set_ylabel(name_of_x + ', ' + name_of_y, fontsize = 20)
    ax.set_ylabel('Price', fontsize=20)
    ax.set_title('Chart of ' + name_of_x + ', ' + name_of_y + ', ' +
                 name_of_z + ' & ' + name_of_t, fontsize=25, fontweight='bold')
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    ax.legend()
    st.plotly_chart(fig,use_container_width=True,sharing="streamlit")
    st.pyplot(fig, clear_figure=True)
    ###
