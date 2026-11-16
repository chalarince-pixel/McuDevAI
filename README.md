# TinyML Sine Wave Predictor: PyTorch to ESP32-S3 (TFLite Micro)

This project demonstrates a complete end-to-end **TinyML** pipeline on the **ESP32-S3** microcontroller. The system runs local, hardware-accelerated **INT8 quantized** neural network inference to predict a sine wave ($y = \sin(x)$), generating a real-time ASCII visualization plot over the serial monitor.

---

## 1. The TinyML Pipeline Architecture

The workflow follows the industry-standard deep learning deployment pipeline for edge devices:

```
[ PC / GPU / Cloud ]
   1. Train Model (PyTorch)
          ↓
   2. Export Graph (ONNX)
          ↓
   3. Convert Model (TensorFlow Lite Float32)
          ↓
   4. INT8 Quantization (Post-Training Calibration)
          ↓
   5. Serialize Model (Convert to C Header Byte Array)
          ↓
[ ESP32-S3 Microcontroller ]
   6. Load C Byte Array (Flash Memory / ROM)
          ↓
   7. Allocate Interpreter & Tensor Arena (Internal SRAM)
          ↓
   8. Quantize Input (Float x → Int8)
          ↓
   9. Invoke Hardware-Accelerated Inference (esp-nn Coprocessor)
          ↓
  10. Dequantize Output (Int8 → Float y) & Plot Results
```

---

## 2. PC-Side Pipeline: What Happens in `train_and_convert.py`

The educational script [train_and_convert.py](file:///Users/rince_hurtle/Documents/PlatformIO/Projects/ChalaMCUS3/main/train_and_convert.py) executes the offline training and preparation pipeline. The steps are:

### Step 1: Generate Training Data
We generate 1,000 coordinate points of $y = \sin(x)$ where $x \in [0, 2\pi]$. This represents our mathematical ground truth that the model must learn to approximate.

### Step 2: Define and Train PyTorch MLP
We build a 3-layer Multi-Layer Perceptron (MLP) in PyTorch:
* **Input Layer:** 1 node (takes the $x$ coordinate value).
* **Hidden Layer 1:** 16 nodes with a Rectified Linear Unit (ReLU) activation function.
* **Hidden Layer 2:** 16 nodes with a ReLU activation function.
* **Output Layer:** 1 node (predicts the $y$ coordinate value).

The model starts with completely random weights and is trained using the **Mean Squared Error (MSE)** loss function and the **Adam** optimizer for 500 epochs until the error is minimized.

### Step 3: Export to ONNX
PyTorch's dynamic graph is exported to **ONNX (Open Neural Network Exchange)** format. This formats the neural network as a standardized computational graph, making it independent of PyTorch.

### Step 4: Convert to TensorFlow Lite (Float32)
The ONNX graph is converted into a TensorFlow Lite (`.tflite`) FlatBuffer. This is an optimized, flat data structure designed for lightweight mobile and embedded parsers.

### Step 5: Post-Training INT8 Quantization
Standard microcontrollers lack high-performance floating-point units. To optimize execution speed and reduce memory consumption by **$4\times$**:
1. We feed a **Representative Dataset** (a subset of $x$ values) into the converter.
2. The converter monitors the range of floating-point values running through the network's activations and weights.
3. It maps the floating-point values ($[-1.0, 1.0]$) to signed 8-bit integers ($[-128, 127]$).
4. The output is a fully quantized **INT8 TFLite model** (`sine_model_quant.tflite`).

### Step 6: Generate C Header Byte Array (xxd)
Because microcontrollers lack traditional operating systems and file systems, we cannot read a `.tflite` file from a hard drive. 
The script parses the binary model file and converts it into a C byte array (`g_model[]` in `sine_model_data.cc`). This compiles the model directly into the ESP32-S3's **read-only Flash memory (ROM)**.

---

## 3. On-Chip Pipeline: Why `main.cpp` is Needed

While the model is trained and frozen on a PC, [main.cpp](file:///Users/rince_hurtle/Documents/PlatformIO/Projects/ChalaMCUS3/main/main.cpp) serves as the **runtime execution engine** on the ESP32-S3 hardware. 

The microcontroller cannot parse PyTorch, ONNX, or standard TensorFlow code. `main.cpp` is necessary to bridge this gap:

### 1. Bootstrapping & OS Context (FreeRTOS)
The chip runs **FreeRTOS** instead of Windows or macOS. `main.cpp` defines `app_main(void)` as the startup thread context, allocating memory and scheduling resources safely.

### 2. Loading the TFLite Micro Interpreter
It initializes the lightweight C++ interpreter (`tflite::MicroInterpreter`) using:
* **The Model Pointer:** Maps to the static `g_model[]` array in Flash memory.
* **The Operator Resolver:** Registers only the specific mathematical operations required by our model (in this case, `AddFullyConnected()`), minimizing binary size.
* **The Tensor Arena:** Allocates a small, aligned memory buffer ($3\text{ KB}$) in the ESP32-S3's fast **internal SRAM** to act as working memory for network layers.

### 3. Hardware-Software Co-Design (`esp-nn`)
`main.cpp` binds directly to Espressif's optimized **`esp-nn`** libraries under the hood. During `interpreter.Invoke()`, calculations are offloaded to the S3's proprietary **128-bit vector coprocessor (PIE)**, accelerating matrix multiplications on the silicon level.

### 4. Input Quantization & Output De-quantization
Since the model inside Flash memory only understands integers ($[-128, 127]$) but the hardware generates floats (like $x = 1.57$), `main.cpp` performs the runtime scaling:
* **Quantization:** `x_quantized = (x / scale) + zero_point`
* **De-quantization:** `y_pred = (y_quantized - zero_point) * scale`

### 5. Peripheral Communication & Visualization
It coordinates printing results to the serial UART port (`printf`) and plots a real-time ASCII visualization curve, showing the predictions tracking mathematical ground truth at an average speed of **$110\ \mu\text{s}$ per inference**.

---

## How to Build and Run

Run these commands in your project folder to flash the firmware and view the running visual inference:

```bash
# 1. Flash the project onto your ESP32-S3
~/.platformio/penv/bin/pio run -t upload

# 2. Start the serial monitor
~/.platformio/penv/bin/pio device monitor
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details.

Copyright © 2026 [Chala Rince](https://github.com/chalarince-pixel). All rights reserved.

