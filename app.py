from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from flask_cors import CORS  # Import flask_cors untuk mengaktifkan CORS
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)  # Aktifkan CORS untuk semua rute

# Memuat model hanya sekali saat server dimulai
MODEL_PATH = 'static/mobilenetv2_model_with_regularization.h5'
model = load_model(MODEL_PATH)

# Daftar label kelas (sesuai dengan urutan kelas yang digunakan selama pelatihan)
labels = ['Tea', 'Soda', 'Milkshake', 'Coffee', 'Bubble & Boba']

# Fungsi untuk melakukan praproses pada gambar input
def prepare_image(image, target_size=(224, 224)):
    # Ubah gambar ke array numpy
    img = image.resize(target_size)
    img_array = img_to_array(img)
    # Tambahkan dimensi batch dan normalisasi
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Periksa apakah ada file yang dikirim dengan request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Periksa apakah file dikirim
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Baca gambar dari request
    try:
        # Gunakan PIL Image untuk membaca file gambar
        image = Image.open(file)
        processed_image = prepare_image(image)

        # Melakukan prediksi
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        # Menentukan nilai sugar_intake berdasarkan predicted_class
        if predicted_class_label == 'Tea':
            sugar_intake = 21
        elif predicted_class_label == 'Soda':
            sugar_intake = 33
        elif predicted_class_label == 'Milkshake':
            sugar_intake = 20
        elif predicted_class_label == 'Coffee':
            sugar_intake = 21
        elif predicted_class_label == 'Bubble & Boba':
            sugar_intake = 34.36
        else:
            sugar_intake = None  # Default jika tidak ada kecocokan (seharusnya tidak terjadi)

        # Kembalikan hasil prediksi sebagai JSON
        return jsonify({
            'predicted_class': predicted_class_label,
            'confidence': f"{confidence:.2f}%",
            'sugar_intake': sugar_intake
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Menjalankan Flask app
if __name__ == '__main__':
    app.run(debug=True)
