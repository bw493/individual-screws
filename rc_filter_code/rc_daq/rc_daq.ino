// rc_daq.ino
// ESP32 data acquisition for the cascaded RC filter project (Physics 4BL).
//
// Two acquisition modes, both paced by the microsecond timer so the samples are evenly spaced:
//
//   STREAM  one pin, continuous, for the 20 s frequency sweeps and the reference sweeps.
//           ASCII (default, SerialPlot compatible): one value per line, 64 us period (15.6 kS/s).
//           Binary (for daq.py): 2 bytes per sample, 25 us period (40 kS/s), covers the full 10 kHz sweep.
//
//   BURST   two pins in alternation (GPIO 34 first, then GPIO 35), stored in RAM at full speed and
//           printed afterwards as "ch34,ch35" lines. Used for the sampling-rate and delay calibration,
//           the triangle-wave linearity check, the phase points, and the square-wave bursts.
//           Default: 16000 pairs every 49 us (20.4 kS/s per channel, 0.78 s).
//           49 us is chosen on purpose: with a 50 us period the aliases of the square wave's high
//           harmonics land exactly on 500, 1500, 2500 and 3500 Hz and bias the measured ratios by up
//           to 3 %. With 49 us they fall between harmonics and contribute less than 0.05 %.
//
// The board boots straight into an ASCII stream of GPIO 35, so SerialPlot shows data immediately.
// Send a command (Arduino Serial Monitor or the SerialPlot "Commands" panel, ending with \n) to
// change mode. Any incoming byte stops a running stream.
//
//   s [pin]            stream one pin (default 35)
//   b [pairs] [us]     burst two pins (defaults 16000, 49)
//   p <us>             set the stream period in microseconds
//   f a | f b          stream format: ASCII or binary (binary also sets the period to 25 us)
//   i                  print settings and the measured analogRead time
//   x                  stop and stay idle
//
// Lines that begin with '#' carry metadata. SerialPlot ignores them and daq.py records them.
//
// Wiring: GPIO 34 = input terminal (Vin, generator side of R1 or C3), GPIO 35 = filter output (Vout).
// Both are ADC1 pins. Keep every signal between about 0.15 V and 2.45 V. Share the ground.

#include <Arduino.h>
#include "esp_timer.h"

static const int PIN_IN = 34;
static const int PIN_OUT = 35;
static const uint32_t BAUD = 921600;

static const uint32_t ASCII_PERIOD_US = 64;   // 15625 S/s, ~85 % of the 921600 baud link
static const uint32_t BINARY_PERIOD_US = 25;  // 40000 S/s, ~87 % of the 921600 baud link
static const uint32_t DEFAULT_PAIRS = 16000;
static const uint32_t DEFAULT_PAIR_US = 49;
static const uint32_t MAX_PAIRS = 25000;      // two 50 kB buffers

static uint32_t streamPeriodUs = ASCII_PERIOD_US;
static bool binaryStream = false;
static int streamPin = PIN_OUT;
static float readTimeUs = 0.0f;

enum Mode { IDLE, STREAM };
static Mode mode = STREAM;

static inline void waitUntil(int64_t t) {
  while (esp_timer_get_time() < t) {
  }
}

static float measureReadTime(int pin) {
  for (int i = 0; i < 50; i++) analogRead(pin);
  const int n = 2000;
  int64_t t0 = esp_timer_get_time();
  volatile uint32_t sink = 0;
  for (int i = 0; i < n; i++) sink += analogRead(pin);
  int64_t t1 = esp_timer_get_time();
  (void)sink;
  return (float)(t1 - t0) / n;
}

static void printInfo() {
  Serial.printf("# info baud=%lu stream_pin=%d stream_period_us=%lu stream_format=%s "
                "pair_period_us_default=%lu pairs_default=%lu analogRead_us=%.2f\n",
                (unsigned long)BAUD, streamPin, (unsigned long)streamPeriodUs,
                binaryStream ? "binary" : "ascii", (unsigned long)DEFAULT_PAIR_US,
                (unsigned long)DEFAULT_PAIRS, readTimeUs);
  if (readTimeUs * 2.0f + 2.0f > DEFAULT_PAIR_US) {
    Serial.printf("# warning: two reads take %.1f us, longer than the %lu us pair period. "
                  "Use a longer period, for example: b 16000 %d\n",
                  2.0f * readTimeUs, (unsigned long)DEFAULT_PAIR_US, (int)(2.0f * readTimeUs + 8.0f));
  }
  if (readTimeUs + 3.0f > streamPeriodUs) {
    Serial.printf("# warning: analogRead takes %.1f us, close to the %lu us stream period\n",
                  readTimeUs, (unsigned long)streamPeriodUs);
  }
}

// Writes v as decimal digits plus '\n' into p and returns the length.
static inline int formatValue(uint16_t v, char *p) {
  char tmp[6];
  int i = 0;
  do {
    tmp[i++] = (char)('0' + v % 10);
    v /= 10;
  } while (v);
  int len = 0;
  while (i) p[len++] = tmp[--i];
  p[len++] = '\n';
  return len;
}

static void runStream() {
  const uint32_t period = streamPeriodUs;
  const int pin = streamPin;
  uint32_t samples = 0, late = 0;
  char buf[8];
  for (int i = 0; i < 20; i++) analogRead(pin);
  if (!binaryStream) {
    Serial.printf("# stream pin=%d period_us=%lu format=ascii\n", pin, (unsigned long)period);
  }
  int64_t next = esp_timer_get_time() + 200;
  while (true) {
    int64_t now = esp_timer_get_time();
    if (now > next + 2) {
      late++;
      if (now > next + (int64_t)period) next = now;  // fell more than one period behind: resync
    }
    waitUntil(next);
    uint16_t v = (uint16_t)analogRead(pin);
    next += period;
    if (binaryStream) {
      uint8_t b[2] = {(uint8_t)(0x80 | (v >> 7)), (uint8_t)(v & 0x7F)};
      Serial.write(b, 2);
    } else {
      int len = formatValue(v, buf);
      Serial.write((const uint8_t *)buf, len);
    }
    samples++;
    if ((samples & 0xFF) == 0 && Serial.available()) break;
  }
  Serial.flush();
  Serial.printf("\n# stream stopped samples=%lu late=%lu period_us=%lu pin=%d\n",
                (unsigned long)samples, (unsigned long)late, (unsigned long)period, pin);
}

static void runBurst(uint32_t pairs, uint32_t period) {
  if (pairs == 0 || pairs > MAX_PAIRS) {
    Serial.printf("# error: pairs must be between 1 and %lu\n", (unsigned long)MAX_PAIRS);
    return;
  }
  uint16_t *a = (uint16_t *)malloc(pairs * sizeof(uint16_t));
  uint16_t *b = (uint16_t *)malloc(pairs * sizeof(uint16_t));
  if (!a || !b) {
    free(a);
    free(b);
    Serial.println("# error: not enough RAM for this many pairs");
    return;
  }
  for (int i = 0; i < 20; i++) {
    analogRead(PIN_IN);
    analogRead(PIN_OUT);
  }
  uint32_t late = 0;
  int64_t next = esp_timer_get_time() + 500;
  const int64_t tStart = next;
  for (uint32_t i = 0; i < pairs; i++) {
    if (esp_timer_get_time() > next + 1) late++;
    waitUntil(next);
    a[i] = (uint16_t)analogRead(PIN_IN);
    b[i] = (uint16_t)analogRead(PIN_OUT);
    next += period;
  }
  const int64_t elapsed = next - tStart;
  Serial.printf("# burst pairs=%lu period_us=%lu late=%lu elapsed_us=%lld analogRead_us=%.2f\n",
                (unsigned long)pairs, (unsigned long)period, (unsigned long)late,
                (long long)elapsed, readTimeUs);
  Serial.println("ch34,ch35");
  char line[16];
  for (uint32_t i = 0; i < pairs; i++) {
    int n = snprintf(line, sizeof(line), "%u,%u\n", a[i], b[i]);
    Serial.write((const uint8_t *)line, n);
  }
  Serial.println("# end");
  free(a);
  free(b);
}

static void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  char c = cmd.charAt(0);
  String args = cmd.substring(1);
  args.trim();
  long v1 = -1, v2 = -1;
  if (args.length()) {
    int sp = args.indexOf(' ');
    v1 = args.substring(0, sp < 0 ? args.length() : sp).toInt();
    if (sp >= 0) v2 = args.substring(sp + 1).toInt();
  }
  switch (c) {
    case 's':
      streamPin = (v1 == PIN_IN) ? PIN_IN : PIN_OUT;
      mode = STREAM;
      break;
    case 'b':
      runBurst(v1 > 0 ? (uint32_t)v1 : DEFAULT_PAIRS, v2 > 0 ? (uint32_t)v2 : DEFAULT_PAIR_US);
      mode = IDLE;
      break;
    case 'p':
      if (v1 >= 10) streamPeriodUs = (uint32_t)v1;
      Serial.printf("# stream_period_us=%lu\n", (unsigned long)streamPeriodUs);
      mode = IDLE;
      break;
    case 'f':
      if (args.startsWith("b")) {
        binaryStream = true;
        streamPeriodUs = BINARY_PERIOD_US;
      } else {
        binaryStream = false;
        streamPeriodUs = ASCII_PERIOD_US;
      }
      Serial.printf("# stream_format=%s stream_period_us=%lu\n", binaryStream ? "binary" : "ascii",
                    (unsigned long)streamPeriodUs);
      mode = IDLE;
      break;
    case 'i':
      printInfo();
      mode = IDLE;
      break;
    case 'x':
      mode = IDLE;
      Serial.println("# idle");
      break;
    default:
      Serial.println("# commands: s [pin] | b [pairs] [us] | p <us> | f a|b | i | x");
      mode = IDLE;
  }
}

void setup() {
  Serial.setTxBufferSize(8192);
  Serial.setRxBufferSize(256);
  Serial.begin(BAUD);
  Serial.setTimeout(50);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);  // default range, roughly 0.15 V to 2.45 V linear
  delay(200);
  readTimeUs = measureReadTime(PIN_OUT);
  Serial.println();
  Serial.println("# ready rc_daq");
  printInfo();
}

void loop() {
  if (mode == STREAM) {
    runStream();  // returns when a byte arrives
    mode = IDLE;
  }
  if (Serial.available()) {
    handleCommand(Serial.readStringUntil('\n'));
  }
}
