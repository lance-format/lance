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
package org.lance;

import org.lance.index.DistanceType;
import org.lance.index.IndexParams;
import org.lance.index.vector.HnswBuildParams;
import org.lance.index.vector.IvfBuildParams;
import org.lance.index.vector.PQBuildParams;
import org.lance.index.vector.SQBuildParams;
import org.lance.index.vector.VectorIndexParams;
import org.lance.ipc.ApproxMode;
import org.lance.ipc.Query;
import org.lance.test.JniTestHelper;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class JNITest {
  @Test
  public void testInts() {
    JniTestHelper.parseInts(Arrays.asList(1, 2, 3));
  }

  @Test
  public void testLongs() {
    JniTestHelper.parseLongs(Arrays.asList(1L, 2L, 3L, Long.MAX_VALUE));
  }

  @Test
  public void testIntsOpt() {
    JniTestHelper.parseIntsOpt(Optional.of(Arrays.asList(1, 2, 3)));
  }

  @Test
  public void testQuery() {
    Query defaultQuery =
        new Query.Builder().setColumn("column").setKey(new float[] {1.0f, 2.0f, 3.0f}).build();
    assertEquals(ApproxMode.NORMAL, defaultQuery.getApproxMode());

    JniTestHelper.parseQuery(
        Optional.of(
            new Query.Builder()
                .setColumn("column")
                .setKey(new float[] {1.0f, 2.0f, 3.0f})
                .setK(10)
                .setNprobes(20)
                .setEf(30)
                .setCentroidEf(30)
                .setRefineFactor(40)
                .setDistanceType(DistanceType.L2)
                .setUseIndex(true)
                .setQueryParallelism(-1)
                .setApproxMode(ApproxMode.ACCURATE)
                .build()));
  }

  @Test
  public void testIvfFlatIndexParams() {
    JniTestHelper.parseIndexParams(
        IndexParams.builder()
            .setVectorIndexParams(VectorIndexParams.ivfFlat(10, DistanceType.L2))
            .build());
  }

  @Test
  public void testIvfPqIndexParams() {
    JniTestHelper.parseIndexParams(
        IndexParams.builder()
            .setVectorIndexParams(VectorIndexParams.ivfPq(10, 8, 4, DistanceType.L2, 50))
            .build());
  }

  @Test
  public void testIvfPqWithCustomParamsIndexParams() {
    IvfBuildParams ivf =
        new IvfBuildParams.Builder()
            .setNumPartitions(20)
            .setMaxIters(100)
            .setSampleRate(512)
            .build();
    PQBuildParams pq =
        new PQBuildParams.Builder()
            .setNumSubVectors(8)
            .setNumBits(8)
            .setMaxIters(100)
            .setKmeansRedos(3)
            .setSampleRate(1024)
            .build();

    JniTestHelper.parseIndexParams(
        IndexParams.builder()
            .setVectorIndexParams(VectorIndexParams.withIvfPqParams(DistanceType.Cosine, ivf, pq))
            .build());
  }

  @Test
  public void testIvfHnswPqIndexParams() {
    IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(15).build();
    HnswBuildParams hnsw =
        new HnswBuildParams.Builder()
            .setMaxLevel((short) 10)
            .setM(30)
            .setEfConstruction(200)
            .setPrefetchDistance(3)
            .build();
    PQBuildParams pq = new PQBuildParams.Builder().setNumSubVectors(16).setNumBits(8).build();

    JniTestHelper.parseIndexParams(
        IndexParams.builder()
            .setVectorIndexParams(
                VectorIndexParams.withIvfHnswPqParams(DistanceType.L2, ivf, hnsw, pq))
            .build());
  }

  @Test
  public void testIvfHnswSqIndexParams() {
    IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(25).build();
    HnswBuildParams hnsw =
        new HnswBuildParams.Builder()
            .setMaxLevel((short) 8)
            .setM(25)
            .setEfConstruction(175)
            .build();
    SQBuildParams sq =
        new SQBuildParams.Builder().setNumBits((short) 16).setSampleRate(512).build();

    JniTestHelper.parseIndexParams(
        IndexParams.builder()
            .setVectorIndexParams(
                VectorIndexParams.withIvfHnswSqParams(DistanceType.Dot, ivf, hnsw, sq))
            .build());
  }

  @Test
  public void testInvalidCombinationPqAndSq() {
    IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(10).build();
    PQBuildParams pq = new PQBuildParams.Builder().build();
    SQBuildParams sq = new SQBuildParams.Builder().build();

    assertThrows(
        IllegalArgumentException.class,
        () -> {
          new VectorIndexParams.Builder(ivf)
              .setDistanceType(DistanceType.L2)
              .setPqParams(pq)
              .setSqParams(sq)
              .build();
        });
  }

  @Test
  public void testInvalidCombinationHnswWithoutPqOrSq() {
    IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(10).build();
    HnswBuildParams hnsw = new HnswBuildParams.Builder().build();

    assertThrows(
        IllegalArgumentException.class,
        () -> {
          new VectorIndexParams.Builder(ivf)
              .setDistanceType(DistanceType.L2)
              .setHnswParams(hnsw)
              .build();
        });
  }

  @Test
  public void testValidCombinationIvfSqWithoutHnsw() {
    IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(10).build();
    SQBuildParams sq = new SQBuildParams.Builder().build();

    JniTestHelper.parseIndexParams(
        IndexParams.builder()
            .setVectorIndexParams(
                new VectorIndexParams.Builder(ivf)
                    .setDistanceType(DistanceType.L2)
                    .setSqParams(sq)
                    .build())
            .build());
  }
}
