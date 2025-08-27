#Tugas - Single Layer Perceptron with Sigmoid Activation for Binary Classification on Iris Dataset
#Andrian Danar Perdana (23/513040/PA/21917)
#Andreandhiki Riyanta Putra (23/517511/PA/22191)
#Daffa Indra Wibowo (23/518514/PA/22253)
#Muhammad Argya Vityasy (23/522547/PA/22475)
#Rayhan Firdaus Ardian (23/519095/PA/22279)

#Impor libarary yang dibutuhkan
import numpy as np                      #aljabar linier dan operasi numerik
import matplotlib.pyplot as plt         #visualisasi data
import pandas as pd                     #manipulasi data
from sklearn.datasets import load_iris  #dataset iris

#Hyperparameters
RANDOM_SEED = 42        #untuk memastikan hasil yang konsisten
LEARNING_RATE = 0.1     #menentukan learning rate
EPOCHS = 200            #jumlah iterasi pelatihan
TEST_RATIO = 0.2        #proporsi data yang digunakan untuk pengujian (80% pelatihan, 20% pengujian)

#Load Iris Dataset
iris = load_iris()
X = iris["data"]        #fitur
y = iris["target"]      #label

#hanya gunakan kelas 0 dan 1 untuk klasifikasi biner (seperti di gsheets)
mask = y < 2   
X = X[mask]
y = y[mask]

#Train-test split
rng = np.random.default_rng(RANDOM_SEED)    #random number generator
perm = rng.permutation(len(y))              #acak urutan data
X = X[perm]
y = y[perm]

n = len(y)
n_train = int((1 - TEST_RATIO) * n)
#bagi data menjadi data pelatihan dan pengujian
X_train, X_test = X[:n_train], X[n_train:]  
y_train, y_test = y[:n_train], y[n_train:]

#Standardisasi fitur (mean=0, std=1)
mean = X_train.mean(axis=0, keepdims=True)      #hitung mean dari data pelatihan
std = X_train.std(axis=0, keepdims=True) + 1e-8 #hitung std dari data pelatihan, tambahkan epsilon untuk menghindari pembagian dengan nol
X_train_std = (X_train - mean) / std            #standarisasi data pelatihan
X_test_std = (X_test - mean) / std              #standarisasi data pengujian

#preview data
df_preview = pd.DataFrame(
    np.hstack([X_train_std[:8], y_train[:8, None]]),
    columns=["sepal length", "sepal width", "petal length", "petal width", "label"]
)

print(df_preview.to_string(index=False)) #print first 8 deret pertama data pelatihan

#Sigmoid function
def sigmoid(z):                                 
    z = np.clip(z, -500, 500)               #potong nilai z untuk menghindari overflow
    return 1.0 / (1.0 + np.exp(-z))         #rumus matematis sigmoid (1 / (1 + e^(-z)))

#Binary cross-entropy loss
def binary_cross_entropy(y_true, y_pred, eps=1e-12):                                #eps = 1e-12 untuk menghindari log(0)      
    y_pred = np.clip(y_pred, eps, 1 - eps)                                          # avoid log(0) by clipping predictions
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()    #rumus matematis binary cross-entropy loss (-(y*log(y_pred) + (1-y)*log(1-y_pred)))

#Single Layer Perceptron with Sigmoid Activation
class SigmoidPerceptron:
    def __init__(self, n_features, learning_rate=0.1, seed = None):                 #self = instance of the class, n_features = jumlah fitur input, learning_rate = laju pembelajaran, seed = untuk inisialisasi bobot acak
        self.rng = np.random.default_rng(seed)                                      #random number generator   
        self.weights = self.rng.normal(loc=0.0, scale=0.01, size=(n_features,))     #inisialisasi bobot dengan distribusi normal (mean=0, std=0.01)
        self.bias = 0.0                                                             #inisialisasi bias dengan nol 
        self.learning_rate = learning_rate                                          #inisialiasi learning rate       
    
    #fungsi untuk memprediksi probabilitas
    def predict_proba(self, X):
        return sigmoid(X @ self.weights + self.bias) #hitung z = Xw + b, lalu aplikasikan fungsi sigmoid
    
    #fungsi untuk memprediksi kelas biner
    def predict(self, X, threshold=0.5):                            #threshold default = 0.5
        return (self.predict_proba(X) >= threshold).astype(int)     #jika probabilitas >= threshold, kelas = 1, else kelas = 0
    
    #fungsi untuk melatih model
    def fit(self, X, y, epochs=100):                                #X = fitur input, y = label target, epochs = jumlah iterasi pelatihan           
        m, n = X.shape                                              #m = jumlah sampel, n = jumlah fitur
        history = {"loss": [], "accuracy": []}                      #history untuk menyimpan riwayat loss dan akurasi    

        #training loop
        for epoch in range(epochs):
            #forward pass (prediksi)
            y_hat = self.predict_proba(X)                           #prediksi probabilitas  

            #loss dan akurasi
            loss = binary_cross_entropy(y, y_hat)                   #hitung binary cross-entropy loss
            accuracy = ((y_hat >= 0.5).astype(int) == y).mean()     #hitung akurasi

            #backward pass (gradient descent)
            error = y_hat - y                                       #hitung error (y_hat - y)   
            grad_w = (X.T @ error) / m                              #hitung gradien bobot (X^T * error) / m      
            grad_b = error.mean()                                   #hitung gradien bias (mean error)   

            #update bobot dan bias
            self.weights -= self.learning_rate * grad_w             #update bobot (w = w - learning_rate * grad_w)
            self.bias -= self.learning_rate * grad_b                #update bias (b = b - learning_rate * grad_b)

            #simpan riwayat loss dan akurasi
            history["loss"].append(loss)                           
            history["accuracy"].append(accuracy)

        return history                                              #kembalikan riwayat loss dan akurasi
    
#inisialisasi model
model = SigmoidPerceptron(n_features=X_train_std.shape[1], learning_rate=LEARNING_RATE, seed=RANDOM_SEED)       

#latih model
history = model.fit(X_train_std, y_train, epochs=EPOCHS)

#evaluasi model pada data pengujian
test_proba = model.predict_proba(X_test_std)                        #prediksi probabilitas pada data pengujian  
test_predictions = (test_proba >= 0.5).astype(int)                  #prediksi kelas biner pada data pengujian (threshold=0.5)
test_accuracy = (test_predictions == y_test).mean()                 #hitung akurasi pada data pengujian

print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")    #print akurasi akhir pada data pelatihan
print(f"Test accuracy: {test_accuracy:.4f}")                        #print akurasi pada data pengujian

#plot training history
#loss per epoch
plt.figure()
plt.plot(history["loss"], label="Loss")             
plt.title("Binary Cross-Entropy Loss per Epoch (Train Set)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("loss_per_epoch.png")                                   #simpan plot sebagai file PNG
plt.show()

#akurasi per epoch
plt.figure()
plt.plot(history["accuracy"], label="Accuracy")
plt.title("Accuracy per Epoch (Train Set)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("accuracy_per_epoch.png")                               #simpan plot sebagai file PNG  
plt.show()