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
package org.lance.test;

import org.lance.JniLoader;
import org.lance.index.IndexParams;
import org.lance.ipc.Query;

import java.util.List;
import java.util.Optional;

/**
 * Used by the JNI test to test the JNI FFI functionality. Note that if ffi parsing errors out, the
 * whole JVM will crash or all tests will show as UnsatisfiedLinkError.
 */
public class JniTestHelper {
  static {
    JniLoader.ensureLoaded();
  }

  /**
   * JNI parse ints test.
   *
   * @param intsList the given list of integers
   */
  public static native void parseInts(List<Integer> intsList);

  /**
   * JNI parse longs test.
   *
   * @param longsList the given list of longs
   */
  public static native void parseLongs(List<Long> longsList);

  /**
   * JNI parse ints opts test.
   *
   * @param intsOpt the given optional of list of integers
   */
  public static native void parseIntsOpt(Optional<List<Integer>> intsOpt);

  /**
   * JNI parse query test.
   *
   * @param query the optional query
   */
  public static native void parseQuery(Optional<Query> query);

  public static native void parseIndexParams(IndexParams indexParams);

  public static native void parseOrderedIndexMode(String name);
}
