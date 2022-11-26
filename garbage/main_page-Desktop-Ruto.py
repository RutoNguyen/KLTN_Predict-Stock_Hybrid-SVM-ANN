# import funtion from file
from sklearn.metrics import mean_squared_error, mean_absolute_error

import load_data
import plot
from ANN_model import *
from SVM_model import *
# Library
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import streamlit as st
###
import base64
from email.mime import image
#import plotly.express as px
#@st.experimental_memo
st.set_page_config(
    page_title="Stock Price Prediction", page_icon="📊", initial_sidebar_state="expanded"
)
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# img = get_img_as_base64("./image/photo.jfif")
# imgB = get_img_as_base64("./image/background.jfif")
# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#     background-image: url("data:image/png;base64,{imgB}");
#     background-size: cover;
#     background-position: top left;
#     background-repeat: no-repeat;
#     background-attachment: local;
#     }}
# [data-testid="stSidebar"] > div:first-child {{
#     background-image: url("data:image/png;base64,{img}");
#     background-position: center; 
#     background-repeat: no-repeat;
#     background-attachment: fixed;
#     }}
# [data-testid="stHeader"] {{
#     background: rgba(0,0,0,0);
#     }}
# [data-testid="stToolbar"] {{
#     right: 2rem;
#     }}
# </style>
# """
# st.markdown(page_bg_img, unsafe_allow_html=True)

###Parameters LSTM
# n_Layer = 3
# n_units = 96
# n_Dropout = 0.2
# n_epochs = 50
# n_batch_size = 32
###

# dataset = load_data.train_test_set(df,optionColums,split)
# df,X_train,y_train,X_test,y_test,scaler = dataset[0],dataset[1],dataset[2],dataset[3],dataset[4],dataset[5]
# ###
# svm_df,X_train_svm,y_train_svm,X_test_svm,y_test_svm,scaler_svm = df,X_train,y_train,X_test,y_test,scaler
# ANN_model
####
# @st.cache()
def MODEL_ANN():
    dataset = load_data.train_test_set(df, optionColums, split)
    df_op, X_train, y_train, X_test, y_test, scaler = dataset[
        0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5]
    ###
    # n_Layer = 6
    # n_units = 96
    # n_Dropout = 0.2
    # n_epochs = 50
    # n_batch_size = 32
    ###
    st.header('*** ANN Model ***')
    df_ann,df_forecast_ann = df_model_ann(
        df_op[optionColums], X_train, y_train, X_test, y_test, split, scaler,n_Layer,n_units,n_Dropout,n_epochs,n_batch_size)
    # df_ann = df_ann.drop([optionColums])
    st.write('Actually & Predict ANN Model', df_ann)
    ####
    calculate_ann = load_data.calculate(
        df_ann['y_test'], df_ann['Predict_ann'])
    mape_ann, RegScoreFun, meanAbsoluteError_ann, RMSE_ann = calculate_ann[
        0], calculate_ann[1], calculate_ann[2], calculate_ann[3]
    # MAE:lỗi tuyệt đối đề cập đến mức độ khác biệt giữa dự đoán của một quan sát và giá trị thực sự của quan sát đó
    st.write('RegScoreFun r2_score- độ phù hợp:', RegScoreFun)
    st.write('MAPE-sai số tương đối trung bình:', mape_ann)
    st.write('meanAbsoluteError-MAE_sai số tuyệt đối trung bình:',
             meanAbsoluteError_ann)
    st.write(
        'RMSE mean_squared_error-căn bậc 2 của sai số bình phương trung bình:', RMSE_ann)
    ###
    plot.plot_x_y(df_ann['y_test'], 'Real price',
                  df_ann['Predict_ann'], 'Predict_ann')
    ##
    # plot
    df_train = pd.DataFrame(df_op['Predict'][:-15])
    df_train = df_train[:split]
    plot.plot_x_y_z(df_train['Predict'], 'Train', df_ann['y_test'],
                    'Real price', df_ann['Predict_ann'], 'Predict_ann')
    #
    # forecast ANN
    st.write('forecast 15 days next-ANN', df_forecast_ann)
    return df_ann, df_forecast_ann
def MODEL_SVM():
    dataset = load_data.train_test_set(df, optionColums, split)
    df_op, X_train, y_train, X_test, y_test, scaler = dataset[
        0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5]
    ###
    st.header('*** SVM Model ***')
    df_svm, df_forecast_svm, svm_confidence = df_model_svm(
        df_op[optionColums], X_train, y_train, X_test, y_test, split, scaler)
    # df_svm,df_forecast_svm,svm_confidence = SVM_model.model_svm(svm_df[optionColums],X_train_svm,y_train_svm,X_test_svm,y_test_svm,split_svm,scaler_svm )
    ###
    st.write('Actually & Predict SVM Model', df_svm)
    #
    st.write('svm_confidence', svm_confidence)
    # st.write('accuracy_score',acc_score)
    ####
    calculate_svm = load_data.calculate(
        df_svm['y_test'], df_svm['Predict_svm'])
    mape_svm, RegScoreFun, meanAbsoluteError_svm, RMSE_svm = calculate_svm[
        0], calculate_svm[1], calculate_svm[2], calculate_svm[3]
    # MAE:lỗi tuyệt đối đề cập đến mức độ khác biệt giữa dự đoán của một quan sát và giá trị thực sự của quan sát đó
    st.write('RegScoreFun r2_score- độ phù hợp:', RegScoreFun)
    st.write('MAPE-sai số tương đối trung bình:', mape_svm)
    st.write('meanAbsoluteError-MAE_sai số tuyệt đối trung bình:',
             meanAbsoluteError_svm)
    st.write(
        'RMSE mean_squared_error-căn bậc 2 của sai số bình phương trung bình:', RMSE_svm)
    ###
    plot.plot_x_y(df_svm['y_test'], 'Real price',
                  df_svm['Predict_svm'], 'Predict_svm')
    ##
    # plot
    df_train = pd.DataFrame(df_op['Predict'][:-15])
    df_train = df_train[:split]
    plot.plot_x_y_z(df_train['Predict'], 'Train', df_svm['y_test'],
                    'Real price', df_svm['Predict_svm'], 'Predict_svm')
    #
    # forecast ANN
    st.write('forecast 15 days next-ANN', df_forecast_svm)
    return df_svm, df_forecast_svm
########################--------------------------HYBRID---------------------------######################################

#########################################################################################################################

def Hybrid():
    df_svm, df_forecast_svm = MODEL_SVM()
    df_ann, df_forecast_ann = MODEL_ANN()
    ####
    st.header('Hybrid SVM and ANN')
    st.write('Gọi L(t) là giá trị dự báo của mô hình SVM,N(t)^ là giá trị dự báo của mô hình ANN(LSTM), giá trị dự báo của y được tính như sau')
    st.latex(r'''
        \hat{y}=\alpha{\hat{L}}_t+\left(1-\alpha\right){\hat{N}}_t\ \ \ \ \ \ \ \alpha\in[0,1]
    ''')
    st.write('Để xác định tham số trọng số alpha, chúng ta sẽ tìm giá trị của alpha để hệ số dự báo lỗi MSE là nhỏ nhất.')
    st.latex(r'''
        MSE=\sum_{i=1}^{n}{\left(Y_i-Y_{hybrid,i}\right)^2=\sum_{i=1}^{n}\left(Y_i-\left[\alpha Y_{NN,i}+\left(1-\alpha\right)Y_{DTW,i}\right]\right)^2}
    ''')
    st.write('Trong đó Y(i) là giá trị thực tế tại thời điểm i, Y(NN,i) là giá trị dự báo tại thời điểm i được tạo bởi ANN và ')
    st.write('Y(DTW,i) là giá trị dự báo tại thời điểm i được tạo bởi khớp mẫu trong SVM. Đây là một hàm bậc hai, do đó chúng ta có thể rút ra giá trị của alpha làm cho lỗi dự báo MSE nhỏ nhất như sau:')
    st.latex(r'''
        \alpha=\frac{\sum_{i=1}^{n}\left(Y_{NN,i}-Y_{DTW,i}\right)\left(Y_i-Y_{DTW,i}\right)}{\sum_{i=1}^{n}\left(Y_{NN,i}-Y_{DTW,i}\right)^2}
    ''')
    #####
    df_H = df_ann
    df_H['Predict_svm'] = df_svm['Predict_svm']
    st.header('Pre_ANN_SVM')
    st.write('leng of DF:', len(df_H), df_H)
    ##
    df_H = df_H.reset_index()

    y_ac = df_H[optionColums]
    y_ann = df_H['Predict_ann']
    y_svm = df_H['Predict_svm']
    leng_dfH = len(df_H)
    sum1 = 0
    sum2 = 0
    for i in range(leng_dfH):
        sum1 += (y_ann[i] - y_svm[i]) * (y_ac[i] - y_svm[i])
    for i in range(leng_dfH):
        sum2 += (y_ann[i] - y_svm[i])**2
    alpha = sum1/sum2
    if alpha > 1:
        st.write('Vì alpha =', alpha,
                 ' lớn hơn 1 -> alpha = 1 cho nên  y(alpha) = alpha*y(svm)')
    if alpha < 0:
        st.write('vì alpha =', alpha,
                 'nhỏ hơn 0 -> alpha = 0 cho nên y(alpha) = y(ann)')
    ###
    # df_ann_differrence = df_H[optionColums] - df_H['Predict_ann']
    # df_ann_differrence = df_ann_differrence.abs()
    # mean_ann=df_ann_differrence.mean()

    # df_svm_differrence = df_H[optionColums] - df_H['Predict_svm']
    # df_svm_differrence = df_svm_differrence.abs()
    # mean_svm = df_svm_differrence.mean()

    # RegScoreFun_ann = r2_score(y_test_scaled_ann,y_predictions_ann)
    # RegScoreFun_svm = r2_score(y_test_scaled_svm,svm_prediction)

    # st.write('RegScoreFun_ann:','-',RegScoreFun_ann,' RegScoreFun_svm:','-',RegScoreFun_svm)

    arr = []
    for i in range(leng_dfH):
        col = []
        for j in np.arange(0.1, 1, 0.1):
            col.append(round(j*y_svm[i] + (1-j)*y_ann[i], 3))
        arr.append(col)
    df_arr = pd.DataFrame(arr, columns=['alpha = 0.1', 'alpha = 0.2', 'alpha = 0.3', 'alpha = 0.4',
                          'alpha = 0.5', 'alpha = 0.6', 'alpha = 0.7', 'alpha = 0.8', 'alpha = 0.9'])
    st.write('DF with alpha 0-1', df_arr)
    ###
    columns = ['alpha = 0.1', 'alpha = 0.2', 'alpha = 0.3', 'alpha = 0.4',
               'alpha = 0.5', 'alpha = 0.6', 'alpha = 0.7', 'alpha = 0.8', 'alpha = 0.9']
    # arr_RegScoreFun=[]
    for i in range(9):
        # x_score=r2_score(y_test_scaled_svm,df_arr[columns[i]])
        # arr_RegScoreFun.append(x_score)
        st.write('RegScoreFun', i, ':', r2_score(
            df_H[optionColums], df_arr[columns[i]]))
    # df_arr_RegScoreFun = pd
    df_H['Hybrid'] = df_arr['alpha = 0.1']
    st.header('Predict of ANN,SVM,Hybrid')
    st.write(df_H)
    ###
    calculate_hybrid = load_data.calculate(df_H[optionColums], df_H['Hybrid'])
    mape_hybrid, RegScoreFun_hybrid, meanAbsoluteError_hybrid, RMSE_hybrid = calculate_hybrid[
        0], calculate_hybrid[1], calculate_hybrid[2], calculate_hybrid[3]
    ###
    # MAE:lỗi tuyệt đối đề cập đến mức độ khác biệt giữa dự đoán của một quan sát và giá trị thực sự của quan sát đó
    st.write('MAPE:', mape_hybrid)
    st.write('RegScoreFun r2_score:', RegScoreFun_hybrid)
    st.write('RMSE mean_squared_error:', RMSE_hybrid)
    st.write('meanAbsoluteError-MAE:', meanAbsoluteError_hybrid)
    # plot
    # df_H = pd.DataFrame(df_H)
    # df_H['Date'] = pd.to_datetime(df['Date'])
    df_H.set_index('Date', inplace=True)
    plot.plot_x_y_z_t(df_H[optionColums], optionColums, df_H['Predict_ann'],
                      'Predict_ann', df_H['Predict_svm'], 'Predict_svm', df_H['Hybrid'], 'Hybrid')
    # forecast hybrid
    arr_hy = []
    for i in range(len(df_forecast_svm)):
        y_hy = 0.1*df_forecast_svm['Fore_15_next_svm'][i] + \
            (1-0.1)*df_forecast_ann['Fore_15_next_ann'][i]
        arr_hy.append(y_hy)
    df_forecast_hybrid = pd.DataFrame(df[optionColums][-15:])
    df_forecast_hybrid['Fore_15_next_hybrid'] = arr_hy

    # df_forecast_hybrid = pd.DataFrame(arr_hy,columns=['Fore_15_next_hybrid'])
    st.write('forecast 15 days next', df_forecast_hybrid)
# if __name__ == '__main__':
#     # MODEL_ANN()
#     #MODEL_SVM()
#     Hybrid()

# title
st.title('Stock Forecast App 📈 📉 ')
#st.markdown("# Main page 🎈")
######################################
#******************Sidebar
with st.sidebar:
    st.sidebar.header("Sidebar")
    st.sidebar.subheader('Query parameters')
    # Data URL
    dataset = ('VN_Index_Historical_Data(2011-2022)_C',
            'VN_30_Historical_Data(15-22)_C',
            'VN_Index_Historical_Data_C')
    option = st.selectbox('Select dataset for prediction', dataset)
    DATA_URL = ('./Data/'+option+'.csv')
    ####
    # up load file
    uploaded_file = st.file_uploader("Choose a file .csv", type=['csv'])
    # load data
    ####
    if uploaded_file is not None:
        data_load_state = st.text('Loading data...')
        df = load_data.dataF(uploaded_file)
        data_load_state = st.text('Loading data... done!')
        # df = pd.read_csv(uploaded_file)
        st.write("filename:", uploaded_file.name)
    else:
        df = load_data.dataF(DATA_URL)
    ####
    ###
    # Choose predictive variables
    dataColumns = ('Price', 'Open', 'High', 'Low')
    # dataColumns = list(df.columns)
    optionColums = st.sidebar.selectbox('Choose predictive variables', dataColumns)
    ratio_train = st.sidebar.select_slider(
        'Select percentage of train dataset',
        options=[60, 70, 80, 90],
        value=(80)
    )
    ratio_test = 100 - ratio_train
    if ratio_train == 80:
        ratio_test = 100 - ratio_train
        st.sidebar.write(
            'The percentage of the train dataset is:', ratio_train, '%')
        st.sidebar.write(
            'The percentage of the test dataset is:', ratio_test, '%')
    else:
        ratio_test = 100 - ratio_train
        st.sidebar.write(
            'The percentage of the train dataset is:', ratio_train, '%')
        st.sidebar.write(
            'The percentage of the test dataset is:', ratio_test, '%')
    ###################################################################
    st.markdown("### Parameters LSTM")
    n_Layer = st.number_input(
        label="Number of layers",
        value=3,
        format="%d",
        min_value=1
    )
    n_units = st.number_input(
        label="Number Units, số lượng neural ",
        value=96,
        format="%d",
        min_value=2
    )
    n_Dropout = st.slider(
        "Dropout,bỏ qua một vài neural trong quá trình train.",
        min_value=0.01,
        max_value=0.9,
        value=0.2,
        step=0.01
    )
    n_epochs = st.number_input(
        label="Number Epochs, Số lần thuật toán học tập sẽ chuyển qua toàn bộ tập dữ liệu train, hãy cố gắng tăng thêm >500",
        value=50,
        format="%d",
        min_value=1,
        max_value=1000
    )
    n_batch_size = st.number_input(
        label="batch_size, Số lượng mẫu dữ liệu sẽ sử dụng trên mỗi lần train",
        value=32,
        format="%d",
        min_value=10
    )
with st.container():
    # columns of dataframe
    col_count = len(df.columns)
    st.write('Number of columns of data sets: ', col_count)
    # rows of dataframe
    row_count = len(df)
    st.write('Number of rows of data sets: ', row_count)
    st.dataframe(df)
    # chart
    plot.plot_x(df['Price'],'Close')
    plot.plot_x_y(df['Price'], 'Close', df['Open'], 'Open')
    plot.plot_x_y(df['High'],'High',df['Low'],'Low')
    # plot.plot_x_y(df['High'], 'High', df['Low'], 'Low')
    ###
    ###
    split = int(ratio_train/100 * (len(df)-15))

tab1, tab2, tab3,tab4, tab5 = st.tabs(["SVM Model", "ANN Model", "Hybrid Model","📈 Chart", "🗃 Data"])
with tab1:
    MODEL_SVM()
with tab2:
    MODEL_ANN()
with tab3:
    Hybrid()
# with st.container():
#     #st.header("Big one")
#     Hybrid()