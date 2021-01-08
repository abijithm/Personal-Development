import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier


################## Multi Output Classifier Model #######################


def label_pitch(row):
   if row['pitch_type_Fastball'] == 1 :
      return 'Fastball'
   if row['pitch_type_Changeup'] == 1 :
      return 'Changeup'
   if row['pitch_type_Curveball'] == 1 :
      return 'Curveball'
   if row['pitch_type_Slider'] == 1:
      return 'Slider'
   return 'Other'

data = pd.read_csv('Data Project Data.csv')
oneHotEncodedPitches = pd.get_dummies(data['pitch_type'], prefix='pitch_type')
oneHotEncodedPitches
pitch_data = data.join(oneHotEncodedPitches)

classified_pitch_data = pitch_data[pitch_data['pitch_type'].notna()]
unclassified_pitch_data = pitch_data[pitch_data['pitch_type'].isna()]

mo_data_A = classified_pitch_data[['pitch_id', 'pitcher_id', 'pitcher_side', 'pitch_type',
                    'pitch_initial_speed_a', 'break_x_a', 'break_z_a',
                    'pitch_type_Changeup', 'pitch_type_Curveball',
                    'pitch_type_Fastball', 'pitch_type_Slider']].dropna()

mo_data_B = classified_pitch_data[['pitch_id', 'pitcher_id', 'pitcher_side', 'pitch_type',
                    'pitch_initial_speed_b', 'spinrate_b', 'break_x_b', 'break_z_b',
                    'pitch_type_Changeup', 'pitch_type_Curveball',
                    'pitch_type_Fastball', 'pitch_type_Slider']].dropna()

mo_training_set_A_features = mo_data_A[['pitch_initial_speed_a', 
                                      'break_x_a', 'break_z_a']]

mo_training_set_A_labels = mo_data_A[['pitch_type_Changeup', 'pitch_type_Curveball',
                                'pitch_type_Fastball', 'pitch_type_Slider']]

mo_training_set_B_features = mo_data_B[['pitch_initial_speed_b', 
                                      'spinrate_b', 'break_x_b', 'break_z_b']]

mo_training_set_B_labels = mo_data_B[['pitch_type_Changeup', 'pitch_type_Curveball',
                                'pitch_type_Fastball', 'pitch_type_Slider']]

mo_X_train_A, mo_X_test_A, mo_t_train_A, mo_t_test_A = train_test_split(mo_training_set_A_features, 
                                                                    mo_training_set_A_labels, 
                                                                    test_size=0.25, random_state=42)

mo_X_train_B, mo_X_test_B, mo_t_train_B, mo_t_test_B = train_test_split(mo_training_set_B_features, 
                                                                    mo_training_set_B_labels,
                                                                    test_size=0.25, random_state=42)

multi_output_class_A = MultiOutputClassifier(KNeighborsClassifier()).fit(mo_X_train_A, mo_t_train_A)
multi_output_class_A_pred = multi_output_class_A.predict(mo_X_test_A)
err_multi_output_class_A = mean_squared_error(mo_t_test_A,multi_output_class_A_pred)

multi_output_class_B = MultiOutputClassifier(KNeighborsClassifier()).fit(mo_X_train_B, mo_t_train_B)
multi_output_class_B_pred = multi_output_class_B.predict(mo_X_test_B)
err_multi_output_class_B = mean_squared_error(mo_t_test_B,multi_output_class_B_pred)

print("Multi Output classification error - System A: ",err_multi_output_class_A)
print("Multi Output classification error - System B: ",err_multi_output_class_B)

df_A = pd.DataFrame(multi_output_class_A_pred, columns = ['pitch_type_Changeup',
                                                        'pitch_type_Curveball',
                                                        'pitch_type_Fastball',
                                                        'pitch_type_Slider'])
df_B = pd.DataFrame(multi_output_class_B_pred, columns = ['pitch_type_Changeup',
                                                        'pitch_type_Curveball',
                                                        'pitch_type_Fastball',
                                                        'pitch_type_Slider'])
class_df_A = df_A.set_index(mo_t_test_A.index)
class_df_B = df_B.set_index(mo_t_test_B.index)

predicted_matrix_A = pd.concat([mo_X_test_A,class_df_A],axis=1)
predicted_matrix_B = pd.concat([mo_X_test_B,class_df_B],axis=1)

predicted_matrix_A['predicted_pitch'] = predicted_matrix_A.apply (lambda row: label_pitch(row), axis=1)
predicted_matrix_B['predicted_pitch'] = predicted_matrix_B.apply (lambda row: label_pitch(row), axis=1)

unclassified_training_data_A = unclassified_pitch_data[['pitch_initial_speed_a', 
                                                      'break_x_a','break_z_a']].dropna()
unclassified_training_data_B = unclassified_pitch_data[['pitch_initial_speed_b', 
                                                        'spinrate_b','break_x_b','break_z_b']].dropna()

unclassified_multi_output_class_A_pred = multi_output_class_A.predict(unclassified_training_data_A)

unclassified_multi_output_class_B_pred = multi_output_class_B.predict(unclassified_training_data_B)

unclass_df_A = pd.DataFrame(unclassified_multi_output_class_A_pred, columns = ['pitch_type_Changeup',
                                                                       'pitch_type_Curveball',
                                                                       'pitch_type_Fastball',
                                                                       'pitch_type_Slider'])
unclass_df_B = pd.DataFrame(unclassified_multi_output_class_B_pred, columns = ['pitch_type_Changeup',
                                                                       'pitch_type_Curveball',
                                                                       'pitch_type_Fastball',
                                                                       'pitch_type_Slider'])
unclass_df_A = unclass_df_A.set_index(unclassified_training_data_A.index)
unclass_df_B = unclass_df_B.set_index(unclassified_training_data_B.index)

unclass_predicted_matrix_A = pd.concat([unclassified_training_data_A,unclass_df_A],axis=1)
unclass_predicted_matrix_B = pd.concat([unclassified_training_data_B,unclass_df_B],axis=1)

unclass_predicted_matrix_A['predicted_pitch_A'] = unclass_predicted_matrix_A.apply (lambda row: label_pitch(row), axis=1)
unclass_predicted_matrix_B['predicted_pitch_B'] = unclass_predicted_matrix_B.apply (lambda row: label_pitch(row), axis=1)

df_1 = unclass_predicted_matrix_A[[
                                                    'pitch_initial_speed_a', 
                                                    'break_x_a', 'break_z_a', 
                                                    'predicted_pitch_A']]

df_2 = unclass_predicted_matrix_B[['pitch_initial_speed_b', 'spinrate_b',
                                'break_x_b', 'break_z_b','predicted_pitch_B']]

final_prediction_data = pd.concat([df_1,df_2],axis=1)
final_prediction_data['predicted_pitch'] = final_prediction_data['predicted_pitch_A'].fillna(
    final_prediction_data['predicted_pitch_B'])

print(final_prediction_data['predicted_pitch'])


################## Multi Output Classifier Model #######################




################## Keras Deep Learning Neural Network #######################
 

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

model_A = Sequential()
model_A.add(Dense(5,activation='relu',input_dim=3))
model_A.add(Dropout(0.5))
model_A.add(Dense(5,activation='relu'))
model_A.add(Dropout(0.5))
model_A.add(Dense(4,activation='softmax'))

model_B = Sequential()
model_B.add(Dense(5,activation='relu',input_dim=4))
model_B.add(Dropout(0.5))
model_B.add(Dense(5,activation='relu'))
model_B.add(Dropout(0.5))
model_B.add(Dense(4,activation='softmax'))

sgd = SGD(lr=0.05,decay=1e-6,momentum=0.9,nesterov=False)

model_A.compile(loss='categorical_crossentropy',
             optimizer=sgd,
             metrics=['accuracy'])
model_B.compile(loss='categorical_crossentropy',
             optimizer=sgd,
             metrics=['accuracy'])


model_A.fit(mo_X_train_A,mo_t_train_A,epochs=1000,batch_size=128)
model_B.fit(mo_X_train_B,mo_t_train_B,epochs=1000,batch_size=128)

score_A = model_A.evaluate(mo_X_test_A,mo_t_test_A)
score_B = model_B.evaluate(mo_X_test_B,mo_t_test_B)

print("Neural Net System A Score: ",score_A)
print("Neural Net System B Score: ",score_B)




################## Keras Deep Learning Neural Network #######################

