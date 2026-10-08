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
package org.lance.index;

public enum IndexType {
  SCALAR(0),
  BTREE(1),
  BITMAP(2),
  LABEL_LIST(3),
  INVERTED(4),
  NGRAM(5),
  FRAGMENT_REUSE(6),
  MEM_WAL(7),
  ZONEMAP(8),
  BLOOM_FILTER(9),
  RTREE(10),
  VECTOR(100),
  IVF_FLAT(101),
  IVF_SQ(102),
  IVF_PQ(103),
  IVF_HNSW_SQ(104),
  IVF_HNSW_PQ(105),
  IVF_HNSW_FLAT(106),
  IVF_RQ(107),
  IVF_HNSW_RQ(108);

  private final int value;

  IndexType(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }
}
