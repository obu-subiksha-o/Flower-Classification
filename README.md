# Flower Classification with CNN & Gradio

This project classifies images of flowers into five categories using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. A Gradio interface allows users to interactively upload images and see predictions.

---

## Dataset

* **Source**: [TensorFlow Flower Dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
* **Classes**:

  * `daisy`
  * `dandelion`
  * `roses`
  * `sunflowers`
  * `tulips`
* Total images: **3670**

---

## Data Loading & Preprocessing

* Dataset is automatically downloaded and extracted.
* Images are resized to **180x180**.
* Split: **80% training**, **20% validation**
* Batching is set to **32**.

```python
tf.keras.preprocessing.image_dataset_from_directory(...)
```

---

## Model Architecture

A Sequential CNN model with:

* `Rescaling` layer for normalization
* 3 convolutional + max pooling blocks
* Flatten and Dense layers
* Output layer with `softmax` for multiclass classification

```python
model = Sequential([
  layers.Rescaling(1./255, input_shape=(180, 180, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(5, activation='softmax')
])
```

---

## Compilation

* Optimizer: `Adam`
* Loss: `SparseCategoricalCrossentropy(from_logits=True)`
  *(Note: Softmax is used, so this raises a warning. Consider setting `from_logits=False`.)*
* Metric: `Accuracy`

---

## Training

Model is trained for **10 epochs**.

```python
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
```

---

## Gradio Interface

* Built with **Gradio 3.50**
* Accepts user-uploaded images
* Predicts and displays the top 5 class probabilities

```python
gr.Interface(fn=predict_image, inputs=image, outputs=label, interpretation='default').launch(debug='True')
```
