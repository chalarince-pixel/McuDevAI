#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_camera.h"
#include <math.h>
#include <stdio.h>

// Common ESP32-S3 Camera Pinout (e.g., ESP32-S3-EYE / Freenove)
#define CAM_PIN_PWDN -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK 15
#define CAM_PIN_SIOD 4
#define CAM_PIN_SIOC 5
#define CAM_PIN_D7 16
#define CAM_PIN_D6 17
#define CAM_PIN_D5 18
#define CAM_PIN_D4 12
#define CAM_PIN_D3 10
#define CAM_PIN_D2 8
#define CAM_PIN_D1 9
#define CAM_PIN_D0 11
#define CAM_PIN_VSYNC 6
#define CAM_PIN_HREF 7
#define CAM_PIN_PCLK 13

static const char *TAG = "tinyml";

esp_err_t init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_sscb_sda = CAM_PIN_SIOD;
    config.pin_sscb_scl = CAM_PIN_SIOC;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG; 
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    // Camera init
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE("CAMERA", "Camera init failed with error 0x%x", err);
        return err;
    }

    sensor_t * s = esp_camera_sensor_get();
    // OV3660 specific settings
    if (s && s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }
    
    ESP_LOGI("CAMERA", "Camera initialized successfully!");
    return ESP_OK;
}

// TensorFlow Lite Micro Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Pre-trained, pre-quantized INT8 model data
#include "sine_model_data.h"

extern "C" void app_main(void) {
  printf("\n=== ESP32-S3 TinyML Sine Predictor ===\n");

  if (init_camera() != ESP_OK) {
      printf("Camera check FAILED. Check wiring and pins.\n");
  } else {
      printf("Camera check PASSED!\n");
  }

  printf("Initializing TensorFlow Lite Micro...\n");

  // 1. Map the model into a usable data structure.
  const tflite::Model *model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(
        TAG,
        "Model schema version %d is not supported by TFLM schema version %d.",
        static_cast<int>(model->version()), TFLITE_SCHEMA_VERSION);
    return;
  }

  // 2. Pull in only the operator implementations we need (FullyConnected for
  // MLP).
  static tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    ESP_LOGE(TAG, "Failed to add FullyConnected operator to resolver.");
    return;
  }

  // 3. Define the tensor arena (memory where inputs, outputs, and intermediate
  // states live). 16-byte alignment is strictly required by the vector
  // accelerator.
  constexpr int kTensorArenaSize = 3000;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  // 4. Build the interpreter to execute the model.
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                              kTensorArenaSize);

  // 5. Allocate memory from the tensor arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() failed.");
    return;
  }

  // 6. Obtain pointers to the input and output tensors.
  TfLiteTensor *input = interpreter.input(0);
  TfLiteTensor *output = interpreter.output(0);

  ESP_LOGI(TAG, "Interpreter initialization complete.");
  printf("\n--- Starting Continuous Local Inference ---\n");
  printf("Legend: [o] Actual Sine  [*] Neural Network Prediction  [x] Perfect "
         "Overlap\n\n");

  int step = 0;
  const int total_steps = 40;
  const float x_range = 2.0f * 3.14159265f;
  const float x_increment = x_range / total_steps;

  while (true) {
    float x = step * x_increment;

    // 7. INT8 Quantization: Convert input float to 8-bit signed integer
    // (int8_t) Scale and zero_point are loaded dynamically from the TFLite
    // model metadata.
    int8_t x_quantized =
        static_cast<int8_t>(x / input->params.scale + input->params.zero_point);
    input->data.int8[0] = x_quantized;

    // 8. Invoke local hardware-accelerated inference (esp-nn coprocessor
    // executes this)
    int64_t start_time = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter.Invoke();
    int64_t inference_time = esp_timer_get_time() - start_time;

    if (invoke_status != kTfLiteOk) {
      ESP_LOGE(TAG, "Inference execution failed on x = %f", x);
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    // 9. De-quantization: Convert 8-bit output back into a floating-point
    // prediction
    int8_t y_quantized = output->data.int8[0];
    float y_pred =
        (y_quantized - output->params.zero_point) * output->params.scale;
    float y_actual = sinf(x);

    // 10. Generate a beautiful real-time ASCII visualization plot!
    // Map floating values [-1.0f, 1.0f] to columns [0, 50]
    int col_pred = static_cast<int>((y_pred + 1.1f) * 22.0f);
    int col_actual = static_cast<int>((y_actual + 1.1f) * 22.0f);

    // Boundaries check
    if (col_pred < 0)
      col_pred = 0;
    if (col_pred > 48)
      col_pred = 48;
    if (col_actual < 0)
      col_actual = 0;
    if (col_actual > 48)
      col_actual = 48;

    char line[51];
    for (int i = 0; i < 50; ++i) {
      line[i] = ' ';
    }
    line[50] = '\0';

    line[24] = '|'; // Draw the center axis line

    line[col_actual] = 'o'; // Draw actual sine point
    line[col_pred] = '*';   // Draw predicted sine point

    if (col_actual == col_pred) {
      line[col_pred] = 'x'; // Mark 'x' where they overlap!
    }

    printf("x: %4.2f | Actual: %5.2f | Pred: %5.2f | (Time: %lld us) | %s\n", x,
           y_actual, y_pred, static_cast<long long>(inference_time), line);

    step++;
    if (step >= total_steps) {
      step = 0;
      printf("\n--- Cycle Complete. Restarting Wave... ---\n\n");
      vTaskDelay(pdMS_TO_TICKS(1200));
    } else {
      vTaskDelay(pdMS_TO_TICKS(120));
    }
  }
}
