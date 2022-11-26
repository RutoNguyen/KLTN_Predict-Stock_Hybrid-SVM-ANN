from keras.models import Sequential , load_model
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import pandas as pd


def df_model_ann(df,X_train,y_train,X_test,y_test,split,scaler,n_Layer,n_units,n_Dropout,n_epochs,n_batch_size):
    model_ANN = Sequential()
    model_ANN.add(LSTM(units=n_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_ANN.add(Dropout(n_Dropout))
    
    for i in range(n_Layer):
        model_ANN.add(LSTM(units=n_units,return_sequences=True))
        model_ANN.add(Dropout(n_Dropout))
    # model_ANN.add(LSTM(units=96,return_sequences=True))
    # model_ANN.add(Dropout(0.2))
    # model_ANN.add(LSTM(units=96,return_sequences=True))
    # model_ANN.add(Dropout(0.2))   
    
    model_ANN.add(LSTM(units=n_units))
    model_ANN.add(Dropout(n_Dropout))
    
    model_ANN.add(Dense(units=1))
    
    model_ANN.compile(optimizer='adam', loss='mse',metrics=['mae'])
    epochs_hist = model_ANN.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size,  verbose=1, validation_split=0.2)
    model_ANN.save('./save_model/stock_prediction_ANN.h5')
    model_ANN = load_model('./save_model/stock_prediction_ANN.h5')
    #epochs_hist = load_model('stock_prediction_ANN.h5')
    #print(epochs_hist.history.keys())
    
    ###
    y_test_scaled_ann = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_ann = model_ANN.predict(X_test)
    y_predictions_ann = scaler.inverse_transform(predictions_ann.reshape(-1, 1))
    # train_score = r2_score(X_train,y_train)
    # test_score = r2_score(X_test,y_test)
    # create df
    ##
    df_ann = pd.DataFrame(df)
    df_ann = df_ann[split:-15]
    df_ann['y_test'] = y_test_scaled_ann
    df_ann['Predict_ann'] = y_predictions_ann
    ### forecast ANN
    forecast_ann = np.array(df)[-15:]
    forecast_ann = scaler.fit_transform(forecast_ann.reshape(-1,1))

    ann_forecast= model_ANN.predict(forecast_ann)
    ann_forecast = scaler.inverse_transform(ann_forecast.reshape(-1,1))

    #df_forecast_ann = pd.DataFrame(ann_forecast,columns=['Fore_15_next_ann'])

    df_forecast_ann = pd.DataFrame(df[-15:])
    df_forecast_ann['Fore_15_next_ann'] = ann_forecast
    return df_ann,df_forecast_ann
    #st.write('Actually & Predict ANN Model',df_ann)