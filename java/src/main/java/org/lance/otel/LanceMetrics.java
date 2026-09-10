/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.otel;

import org.lance.JniLoader;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.api.metrics.BatchCallback;
import io.opentelemetry.api.metrics.Meter;
import io.opentelemetry.api.metrics.MeterProvider;
import io.opentelemetry.api.metrics.ObservableDoubleMeasurement;
import io.opentelemetry.api.metrics.ObservableMeasurement;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/** Bridges Lance's internal Rust metrics into OpenTelemetry observable instruments. */
public final class LanceMetrics {
  private static final Logger LOGGER = Logger.getLogger(LanceMetrics.class.getName());
  private static final String METER_NAME = "lance";
  private static final List<AutoCloseable> INSTRUMENTS = new ArrayList<>();
  private static boolean instrumented;
  private static MeterProvider instrumentedProvider;

  private LanceMetrics() {}

  /**
   * Register Lance metrics on the global OpenTelemetry {@link MeterProvider}.
   *
   * @return true if the recorder is installed and instruments are registered, false if a different
   *     Rust metrics recorder is already installed in this process
   */
  public static synchronized boolean instrument() {
    return instrument(GlobalOpenTelemetry.get().getMeterProvider());
  }

  /**
   * Register Lance metrics as OpenTelemetry observable instruments.
   *
   * <p>This installs a process-global Rust metrics recorder. If a different Rust metrics recorder
   * is already installed, this returns false and does not create instruments. Repeated successful
   * calls with the same provider are safe and return true without creating duplicate instruments.
   * Calls with a different provider close the existing instruments and register new callbacks on
   * the supplied provider.
   *
   * @param meterProvider the OpenTelemetry meter provider that owns the instruments
   * @return true if the recorder is installed and instruments are registered, false otherwise
   */
  public static synchronized boolean instrument(MeterProvider meterProvider) {
    Objects.requireNonNull(meterProvider, "meterProvider");
    JniLoader.ensureLoaded();

    if (!registerLanceMetricsRecorderNative()) {
      LOGGER.warning(
          "Could not install the Lance metrics recorder: another Rust metrics recorder is already"
              + " installed in this process. Lance metrics will not be exported via"
              + " OpenTelemetry.");
      return false;
    }

    if (instrumented) {
      if (instrumentedProvider == meterProvider) {
        return true;
      }
      closeInstruments();
    }

    Meter meter = meterProvider.meterBuilder(METER_NAME).build();
    Map<String, RegisteredMetric> registeredMetrics = new HashMap<>();
    List<ObservableMeasurement> measurements = new ArrayList<>();

    for (MetricDescription desc : supportedMetrics(catalog())) {
      String unit = desc.getUnit() == null ? "" : desc.getUnit();
      String description = desc.getDescription();
      switch (desc.getKind()) {
        case "counter":
          ObservableDoubleMeasurement counter =
              meter
                  .counterBuilder(desc.getName())
                  .ofDoubles()
                  .setUnit(unit)
                  .setDescription(description)
                  .buildObserver();
          registeredMetrics.put(desc.getName(), RegisteredMetric.scalar(counter));
          measurements.add(counter);
          break;
        case "gauge":
          ObservableDoubleMeasurement gauge =
              meter
                  .gaugeBuilder(desc.getName())
                  .setUnit(unit)
                  .setDescription(description)
                  .buildObserver();
          registeredMetrics.put(desc.getName(), RegisteredMetric.scalar(gauge));
          measurements.add(gauge);
          break;
        case "histogram":
          ObservableDoubleMeasurement buckets =
              meter
                  .counterBuilder(desc.getName() + "_bucket")
                  .ofDoubles()
                  .setDescription(description + " (cumulative buckets)")
                  .buildObserver();
          ObservableDoubleMeasurement count =
              meter
                  .counterBuilder(desc.getName() + "_count")
                  .ofDoubles()
                  .setDescription(description + " (count)")
                  .buildObserver();
          ObservableDoubleMeasurement sum =
              meter
                  .counterBuilder(desc.getName() + "_sum")
                  .ofDoubles()
                  .setUnit(unit)
                  .setDescription(description + " (sum)")
                  .buildObserver();
          registeredMetrics.put(desc.getName(), RegisteredMetric.histogram(buckets, count, sum));
          measurements.add(buckets);
          measurements.add(count);
          measurements.add(sum);
          break;
        default:
          LOGGER.warning(
              "Skipping Lance metric " + desc.getName() + " with unknown kind: " + desc.getKind());
          continue;
      }
    }

    if (!measurements.isEmpty()) {
      ObservableMeasurement first = measurements.get(0);
      ObservableMeasurement[] rest =
          measurements.subList(1, measurements.size()).toArray(new ObservableMeasurement[0]);
      BatchCallback callback =
          meter.batchCallback(() -> recordSnapshot(registeredMetrics), first, rest);
      INSTRUMENTS.add(callback);
    }

    instrumented = true;
    instrumentedProvider = meterProvider;
    return true;
  }

  /** Close registered OpenTelemetry callbacks. Rust metric state remains process-global. */
  public static synchronized void close() {
    closeInstruments();
  }

  /** Return the catalog of Lance metrics described by the native recorder. */
  public static List<MetricDescription> catalog() {
    JniLoader.ensureLoaded();
    return Collections.unmodifiableList(lanceMetricsCatalogNative());
  }

  /** Return a point-in-time snapshot of all recorded Lance metric series. */
  public static List<MetricPoint> snapshot() {
    JniLoader.ensureLoaded();
    return Collections.unmodifiableList(snapshotLanceMetricsNative());
  }

  static List<MetricDescription> supportedMetrics(List<MetricDescription> catalog) {
    List<MetricDescription> supported = new ArrayList<>();
    for (MetricDescription desc : catalog) {
      switch (desc.getKind()) {
        case "counter":
        case "gauge":
        case "histogram":
          supported.add(desc);
          break;
        default:
          LOGGER.warning(
              "Skipping Lance metric " + desc.getName() + " with unknown kind: " + desc.getKind());
      }
    }
    return supported;
  }

  private static void recordSnapshot(Map<String, RegisteredMetric> registeredMetrics) {
    for (MetricPoint point : snapshot()) {
      RegisteredMetric metric = registeredMetrics.get(point.getName());
      if (metric == null) {
        continue;
      }
      metric.record(point);
    }
  }

  private static void closeInstruments() {
    for (AutoCloseable instrument : INSTRUMENTS) {
      try {
        instrument.close();
      } catch (Exception e) {
        LOGGER.warning("Failed to close Lance OpenTelemetry instrument: " + e.getMessage());
      }
    }
    INSTRUMENTS.clear();
    instrumented = false;
    instrumentedProvider = null;
  }

  private static AttributesBuilder attributesBuilder(Map<String, String> values) {
    AttributesBuilder builder = Attributes.builder();
    for (Map.Entry<String, String> entry : values.entrySet()) {
      builder.put(entry.getKey(), entry.getValue());
    }
    return builder;
  }

  private static Attributes attributes(Map<String, String> values) {
    return attributesBuilder(values).build();
  }

  private static final class RegisteredMetric {
    private final ObservableDoubleMeasurement scalar;
    private final ObservableDoubleMeasurement buckets;
    private final ObservableDoubleMeasurement count;
    private final ObservableDoubleMeasurement sum;

    private RegisteredMetric(
        ObservableDoubleMeasurement scalar,
        ObservableDoubleMeasurement buckets,
        ObservableDoubleMeasurement count,
        ObservableDoubleMeasurement sum) {
      this.scalar = scalar;
      this.buckets = buckets;
      this.count = count;
      this.sum = sum;
    }

    private static RegisteredMetric scalar(ObservableDoubleMeasurement measurement) {
      return new RegisteredMetric(measurement, null, null, null);
    }

    private static RegisteredMetric histogram(
        ObservableDoubleMeasurement buckets,
        ObservableDoubleMeasurement count,
        ObservableDoubleMeasurement sum) {
      return new RegisteredMetric(null, buckets, count, sum);
    }

    private void record(MetricPoint point) {
      if (scalar != null && point.getValue() != null) {
        scalar.record(point.getValue(), attributes(point.getAttributes()));
      }
      if (buckets != null && point.getBuckets() != null) {
        for (MetricBucket bucket : point.getBuckets()) {
          AttributesBuilder attributes = attributesBuilder(point.getAttributes());
          attributes.put("le", bucket.getLe());
          buckets.record(bucket.getCumulativeCount(), attributes.build());
        }
      }
      if (count != null && point.getCount() != null) {
        count.record(point.getCount().doubleValue(), attributes(point.getAttributes()));
      }
      if (sum != null && point.getSum() != null) {
        sum.record(point.getSum(), attributes(point.getAttributes()));
      }
    }
  }

  private static native boolean registerLanceMetricsRecorderNative();

  private static native List<MetricDescription> lanceMetricsCatalogNative();

  private static native List<MetricPoint> snapshotLanceMetricsNative();
}
