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

import org.lance.Dataset;
import org.lance.TestUtils;

import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.metrics.data.DoublePointData;
import io.opentelemetry.sdk.metrics.data.MetricData;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LanceMetricsTest {
  private static final String REQUESTS = "lance_object_store_requests_total";
  private static final String DURATION = "lance_object_store_request_duration_seconds";
  private static final String RETRYABLE = "lance_object_store_retryable_responses_total";
  private static final String IN_FLIGHT = "lance_object_store_in_flight_requests";

  @AfterEach
  void closeLanceMetrics() {
    LanceMetrics.close();
  }

  @Test
  void testInstrumentLanceMetricsExportsObjectStoreMetrics(@TempDir Path tempDir) {
    InMemoryMetricReader reader = InMemoryMetricReader.create();
    SdkMeterProvider provider = SdkMeterProvider.builder().registerMetricReader(reader).build();

    try {
      assertTrue(LanceMetrics.instrument(provider));

      Map<String, MetricDescription> catalog =
          LanceMetrics.catalog().stream()
              .collect(Collectors.toMap(MetricDescription::getName, Function.identity()));
      assertTrue(catalog.containsKey(REQUESTS));
      assertEquals("histogram", catalog.get(DURATION).getKind());
      assertEquals("counter", catalog.get(RETRYABLE).getKind());
      assertEquals("gauge", catalog.get(IN_FLIGHT).getKind());

      generateObjectStoreMetrics(tempDir.resolve("otel_metrics.lance"));

      MetricPoint requests =
          LanceMetrics.snapshot().stream()
              .filter(point -> REQUESTS.equals(point.getName()))
              .findFirst()
              .orElseThrow(() -> new AssertionError("expected object store request metric"));
      assertNotNull(requests.getValue());
      assertTrue(requests.getValue() > 0);
      assertTrue(requests.getAttributes().containsKey("operation"));
      assertTrue(requests.getAttributes().containsKey("base"));

      Collection<MetricData> metrics = reader.collectAllMetrics();
      Map<String, MetricData> metricsByName =
          metrics.stream().collect(Collectors.toMap(MetricData::getName, Function.identity()));
      Set<String> names = metricsByName.keySet();
      assertTrue(names.contains(REQUESTS));
      assertTrue(names.contains(DURATION + "_bucket"));
      assertTrue(names.contains(DURATION + "_count"));
      assertTrue(names.contains(DURATION + "_sum"));

      Collection<DoublePointData> bucketPoints =
          metricsByName.get(DURATION + "_bucket").getDoubleSumData().getPoints();
      Collection<DoublePointData> countPoints =
          metricsByName.get(DURATION + "_count").getDoubleSumData().getPoints();
      Collection<DoublePointData> sumPoints =
          metricsByName.get(DURATION + "_sum").getDoubleSumData().getPoints();

      assertFalse(bucketPoints.isEmpty());
      assertTrue(
          bucketPoints.stream()
              .allMatch(point -> point.getAttributes().get(AttributeKey.stringKey("le")) != null));

      Map<Attributes, Double> countsByAttributes =
          countPoints.stream()
              .collect(Collectors.toMap(DoublePointData::getAttributes, DoublePointData::getValue));
      Map<Attributes, Double> infiniteBucketsByAttributes = new HashMap<>();
      for (DoublePointData bucketPoint : bucketPoints) {
        if ("+Inf".equals(bucketPoint.getAttributes().get(AttributeKey.stringKey("le")))) {
          Attributes attributes =
              bucketPoint.getAttributes().toBuilder().remove(AttributeKey.stringKey("le")).build();
          infiniteBucketsByAttributes.put(attributes, bucketPoint.getValue());
        }
      }
      assertEquals(countsByAttributes, infiniteBucketsByAttributes);
      assertTrue(sumPoints.stream().anyMatch(point -> point.getValue() > 0));
    } finally {
      provider.close();
    }
  }

  @Test
  void testInstrumentReRegistersWithNewProvider(@TempDir Path tempDir) {
    InMemoryMetricReader firstReader = InMemoryMetricReader.create();
    InMemoryMetricReader secondReader = InMemoryMetricReader.create();
    SdkMeterProvider firstProvider =
        SdkMeterProvider.builder().registerMetricReader(firstReader).build();
    SdkMeterProvider secondProvider =
        SdkMeterProvider.builder().registerMetricReader(secondReader).build();

    try {
      assertTrue(LanceMetrics.instrument(firstProvider));
      generateObjectStoreMetrics(tempDir.resolve("first_provider.lance"));
      assertTrue(metricNames(firstReader).contains(REQUESTS));

      assertTrue(LanceMetrics.instrument(secondProvider));
      assertTrue(metricNames(secondReader).contains(REQUESTS));
      assertFalse(metricNames(firstReader).contains(REQUESTS));
    } finally {
      firstProvider.close();
      secondProvider.close();
    }
  }

  @Test
  void testSupportedMetricsSkipsUnknownKinds() {
    MetricDescription counter = new MetricDescription("counter", "counter", null, "counter");
    MetricDescription unknown = new MetricDescription("unknown", "summary", null, "unknown");

    List<MetricDescription> supported =
        LanceMetrics.supportedMetrics(Arrays.asList(counter, unknown));

    assertEquals(1, supported.size());
    assertEquals("counter", supported.get(0).getName());
  }

  private static void generateObjectStoreMetrics(Path datasetPath) {
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath.toString());
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(0, dataset.countRows());
      }
    }
  }

  private static Set<String> metricNames(InMemoryMetricReader reader) {
    Collection<MetricData> metrics = reader.collectAllMetrics();
    return metrics.stream().map(MetricData::getName).collect(Collectors.toSet());
  }
}
