
#include "esp_camera.h"
#include "esp_http_server.h"
#include <ESPmDNS.h>
#include <WiFi.h>

// Power management headers
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#include "secrets.h"
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// ===================
// WIFI CREDENTIALS
// ===================
const char *ssid = WIFI_SSID;
const char *password = WIFI_PASS;

#define ALARM_PIN 12 // Moved from 13 to avoid HS2_DATA3 (SD Card) conflict
#define FLASH_PIN 4

httpd_handle_t stream_httpd = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ===================
// SECURITY HELPERS
// ===================

bool check_auth(httpd_req_t *req) {
  char buf[64];
  char query[128];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
    if (httpd_query_key_value(query, "auth", buf, sizeof(buf)) == ESP_OK) {
      if (strcmp(buf, API_KEY) == 0) {
        return true;
      }
    }
  }
  Serial.print("Unauthorized access attempt! Key was: ");
  Serial.println(buf);
  httpd_resp_set_status(req, "401 Unauthorized");
  httpd_resp_send(req, "401 Unauthorized: Invalid API Key", -1);
  return false;
}

// ===================
// ENDPOINT HANDLERS
// ===================

esp_err_t index_handler(httpd_req_t *req) {
  if (!check_auth(req)) return ESP_OK;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, "<h1>ESP32-CAM Safety System Online</h1><p>Auth Status: Verified</p><p>Available endpoints: /stream, /status, /alarm</p>", -1);
}

esp_err_t favicon_handler(httpd_req_t *req) {
  httpd_resp_set_status(req, "204 No Content");
  return httpd_resp_send(req, NULL, 0);
}

esp_err_t alarm_handler(httpd_req_t *req) {
  if (!check_auth(req)) return ESP_OK;
  char buf[32];
  char query[64];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
    if (httpd_query_key_value(query, "state", buf, sizeof(buf)) == ESP_OK) {
      if (strcmp(buf, "on") == 0) {
        digitalWrite(ALARM_PIN, HIGH);
      } else {
        digitalWrite(ALARM_PIN, LOW);
      }
    }
  }
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, "OK", 2);
}

esp_err_t status_handler(httpd_req_t *req) {
  if (!check_auth(req)) return ESP_OK;
  static char json_response[156];
  int rssi = WiFi.RSSI();
  unsigned long uptime = millis() / 1000;
  float sensor_val = analogRead(34) * (3.3 / 4095.0);

  snprintf(json_response, sizeof(json_response),
           "{\"rssi\": %d, \"uptime\": %lu, \"sensor\": %.2f, \"alarm\": %d}",
           rssi, uptime, sensor_val, digitalRead(ALARM_PIN));

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json_response, strlen(json_response));
}

esp_err_t stream_handler(httpd_req_t *req) {
  if (!check_auth(req)) return ESP_OK;
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t *_jpg_buf = NULL;
  char *part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
    } else {
      _jpg_buf_len = fb->len;
      _jpg_buf = fb->buf;
    }
    if (res == ESP_OK) {
      size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY,
                                  strlen(_STREAM_BOUNDARY));
    }
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
    }
    if (res != ESP_OK) {
      break;
    }
    
    // Stability Optimization: Yield to TCP/IP stack
    delay(1); 
  }
  return res;
}

void setup() {
  // Power stability: Disable brownout detector
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  pinMode(ALARM_PIN, OUTPUT);
  digitalWrite(ALARM_PIN, LOW);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA; 
  config.jpeg_quality = 15; 
  
  Serial.println("Initialzing ESP32-CAM Safety System (Optimized)...");

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    while (true) {
      digitalWrite(ALARM_PIN, HIGH); delay(100);
      digitalWrite(ALARM_PIN, LOW); delay(100);
    }
  }
  Serial.println("Camera OK!");

  Serial.printf("Connecting to WiFi: %s\n", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Setup mDNS Discovery
  String hostname = "esp32-safety-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  if (MDNS.begin(hostname.c_str())) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("mDNS Responder Started: %s.local\n", hostname.c_str());
  }

  httpd_config_t http_config = HTTPD_DEFAULT_CONFIG();
  http_config.server_port = 80;

  httpd_uri_t index_uri = {.uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = NULL};
  httpd_uri_t favicon_uri = {.uri = "/favicon.ico", .method = HTTP_GET, .handler = favicon_handler, .user_ctx = NULL};
  httpd_uri_t stream_uri = {.uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL};
  httpd_uri_t status_uri = {.uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = NULL};
  httpd_uri_t alarm_uri = {.uri = "/alarm", .method = HTTP_GET, .handler = alarm_handler, .user_ctx = NULL};

  if (httpd_start(&stream_httpd, &http_config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
    httpd_register_uri_handler(stream_httpd, &favicon_uri);
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &status_uri);
    httpd_register_uri_handler(stream_httpd, &alarm_uri);
    Serial.println("Web server started successfully!");
  }
}

void loop() { 
  // Resilience: Check WiFi connectivity and reconnect if necessary
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi connection lost. Reconnecting...");
    WiFi.disconnect();
    WiFi.begin(ssid, password);
    unsigned long start_ms = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start_ms < 15000) {
      delay(500);
      Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\nWiFi Restored!");
    }
  }
  delay(10000); 
}
