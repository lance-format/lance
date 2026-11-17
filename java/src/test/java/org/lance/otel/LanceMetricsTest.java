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
import org.lance.index.IndexBuildProgress;
import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.ScalarIndexParams;

import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.metrics.data.DoublePointData;
import io.opentelemetry.sdk.metrics.data.MetricData;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
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
  private static final List<String> MEMORY_GAUGES =
      List.of(
          "lance_memory_rss_bytes",
          "lance_memory_rss_peak_bytes",
          "lance_memory_cgroup_bytes",
          "lance_memory_cgroup_limit_bytes",
          "lance_memory_component_bytes");
  private static final String MEMORY_STAGE_PEAK = "lance_memory_stage_peak_bytes";
  private static final String MEMORY_OPERATION_PEAK = "lance_memory_operation_peak_bytes";

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
      // Memory metrics are described whether or not memory observation is on;
      // they stay empty until it is first switched on.
      for (String gauge : MEMORY_GAUGES) {
        assertEquals("gauge", catalog.get(gauge).getKind(), gauge);
      }
      assertEquals("histogram", catalog.get(MEMORY_STAGE_PEAK).getKind());
      assertEquals("histogram", catalog.get(MEMORY_OPERATION_PEAK).getKind());

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
  @EnabledOnOs(OS.LINUX)
  void testSetMemoryObservationSwitchesAtRuntime(@TempDir Path tempDir) {
    InMemoryMetricReader reader = InMemoryMetricReader.create();
    SdkMeterProvider provider = SdkMeterProvider.builder().registerMetricReader(reader).build();

    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      assertTrue(LanceMetrics.instrument(provider));
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(
              allocator, tempDir.resolve("memory_switch.lance").toString());
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 16)) {
        long builds = observedPeaks(MEMORY_OPERATION_PEAK);
        long stages = observedPeaks(MEMORY_STAGE_PEAK);

        LanceMetrics.setMemoryObservation(true);
        assertTrue(memoryGauge("lance_memory_rss_bytes") > 0);

        // The build checks the switch before its first stage, so it reports in
        // full although observation is switched off while it runs.
        AtomicLong buildsAtSwitch = new AtomicLong(-1);
        dataset.createIndex(btreeOptions(), switchingProgress(false, buildsAtSwitch));
        assertEquals(builds, buildsAtSwitch.get());
        assertEquals(builds + 1, observedPeaks(MEMORY_OPERATION_PEAK));
        assertTrue(observedPeaks(MEMORY_STAGE_PEAK) > stages);
        stages = observedPeaks(MEMORY_STAGE_PEAK);

        // A build that starts while observation is off reports nothing, even if
        // observation is switched on while it runs.
        dataset.createIndex(btreeOptions());
        dataset.createIndex(btreeOptions(), switchingProgress(true, new AtomicLong()));
        assertEquals(builds + 1, observedPeaks(MEMORY_OPERATION_PEAK));
        assertEquals(stages, observedPeaks(MEMORY_STAGE_PEAK));
      }
    } finally {
      LanceMetrics.setMemoryObservation(false);
      provider.close();
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

  private static IndexOptions btreeOptions() {
    IndexParams params =
        IndexParams.builder()
            .setScalarIndexParams(ScalarIndexParams.create("btree", "{\"zone_size\": 2048}"))
            .build();
    return IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, params)
        .replace(true)
        .build();
  }

  /** Progress that switches memory observation as a stage starts. */
  private static IndexBuildProgress switchingProgress(boolean enabled, AtomicLong buildsAtSwitch) {
    return new IndexBuildProgress() {
      @Override
      public void stageStart(String stage, Optional<Long> total, String unit) {
        buildsAtSwitch.set(observedPeaks(MEMORY_OPERATION_PEAK));
        LanceMetrics.setMemoryObservation(enabled);
      }

      @Override
      public void stageProgress(String stage, long completed) {}

      @Override
      public void stageComplete(String stage) {}
    };
  }

  /** How many BTree builds, or stages of them, have reported a peak to {@code name}. */
  private static long observedPeaks(String name) {
    return LanceMetrics.snapshot().stream()
        .filter(point -> name.equals(point.getName()))
        .filter(point -> "create_index".equals(point.getAttributes().get("operation")))
        .filter(point -> "btree".equals(point.getAttributes().get("index_type")))
        .mapToLong(MetricPoint::getCount)
        .sum();
  }

  private static double memoryGauge(String name) {
    return LanceMetrics.snapshot().stream()
        .filter(point -> name.equals(point.getName()))
        .map(MetricPoint::getValue)
        .findFirst()
        .orElseThrow(() -> new AssertionError("expected the gauge " + name));
  }

  private static Set<String> metricNames(InMemoryMetricReader reader) {
    Collection<MetricData> metrics = reader.collectAllMetrics();
    return metrics.stream().map(MetricData::getName).collect(Collectors.toSet());
  }
}
