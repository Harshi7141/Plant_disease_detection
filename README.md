### 🌿 Plant Disease Detection

This project uses **Deep Learning** to detect and classify diseases in plants from images of their leaves. By leveraging a **Convolutional Neural Network (CNN)** model, it provides a fast and accurate diagnosis, serving as a valuable tool for farmers and agricultural experts to identify and manage plant health.

---

### ✨ Features

* **Accurate Classification:** Identifies different plant diseases from leaf images with high accuracy.  
* **API-Based Interface:** Built with **FastAPI** to provide a flexible and scalable backend for predictions.  
* **Real-time Predictions:** Accepts image uploads via API and returns instant disease diagnosis.  
* **Modular Architecture:** The project is structured for easy understanding and future enhancements.

---

### 💻 Tech Stack

* **Frameworks & Libraries:**
    * `TensorFlow` / `Keras`: For building and training the deep learning model.  
    * `FastAPI`: For creating the REST API to serve predictions.  
    * `NumPy` & `Pillow`: For data manipulation and image processing.  
* **Language:**
    * `Python`  
* **Tools:**
    * `Uvicorn`: ASGI server for running the FastAPI application.  
    * `Jupyter Notebook`: Used for model training and experimentation.  
    * `Git` / `GitHub`: For version control and project hosting.  

---

📂 Plant_disease_detection/
```
├── main.py # FastAPI application to run the project
├── model/ # Directory to store the trained model
│ └── plant_disease_model.h5 # The pre-trained model file
├── images/ # Folder for sample images
├── requirements.txt # List of all necessary Python packages
└── README.md # Project documentation and instructions
```

---

### 🚀 Getting Started

Follow these simple steps to get a copy of the project up and running on your local machine.

#### **1. Clone the repository**
```bash
git clone https://github.com/Harshi7141/Plant_disease_detection.git
cd Plant_disease_detection
```

#### **2. Install dependencies**


Install all the required Python packages using the requirements.txt file:

```bash
pip install -r requirements.txt
```

#### **3. Run the FastAPI application**

Use Uvicorn to start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will run on http://127.0.0.1:8000. You can test endpoints such as /predict by sending POST requests with leaf images.
