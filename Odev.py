# -*- coding: utf-8 -*-

# Gerekli kütüphane importları
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri setini çek
veriset = pd.read_csv('veriseti.csv')

x = veriset.iloc[:,: -1].values

y = veriset.iloc[:, -1].values

# Boş değerleri ortalama ile doldur
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x)
x = imputer.transform(x)

# ENCODİNG İŞLEMİ
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Eğitim ve test verilerini ayır
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.03, random_state=0)

# Özellikleri ölçeklendirme 
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

# yapay Sinir Ağını Oluştur
model = Sequential()

# Giriş katmanı ve 1.Gizli katman
model.add(Dense(units=12, input_dim=12, activation='relu'))

# 2.Gizli katman
model.add(Dense(units=12, activation='relu'))

# 3.Gizli katman
model.add(Dense(units=12, activation='relu'))

# 4.Gizli katman
model.add(Dense(units=6, activation='relu'))

# Çıktı Katmanı
model.add(Dense(units=1, activation='sigmoid'))

# Modeli çalıştır
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=10)

# Test
sonuclar = model.evaluate(x_test, y_test)
print("Accuracy: %.2f%%" %(sonuclar[1]*100))