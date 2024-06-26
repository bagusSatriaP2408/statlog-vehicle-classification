import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Memuat kembali data training dan testing
train_data = pd.read_csv('best_fold_training_data.csv')
test_data = pd.read_csv('best_fold_testing_data.csv')

# Memisahkan atribut dan label untuk data training
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Memisahkan atribut dan label untuk data testing
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Inisialisasi model SVM
svm_model = SVC(kernel='linear', random_state=0)

# Melatih model SVM menggunakan data training
svm_model.fit(X_train, y_train)

# Melakukan prediksi menggunakan data testing
y_pred = svm_model.predict(X_test)

# Menghitung dan menampilkan akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model : {accuracy:.4f}")
