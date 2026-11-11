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

/**
 * Thrown when a write in {@link WriteParams.WriteMode#CREATE} mode targets a location that already
 * holds a Lance dataset.
 *
 * <p>It extends {@link IllegalArgumentException}, which was thrown for this case before, so
 * existing handlers keep working. Catch it specifically to tell an existing dataset apart from
 * other invalid input.
 */
public class DatasetAlreadyExistsException extends IllegalArgumentException {
  /**
   * Creates the exception with the message of the native error, which names the dataset location.
   *
   * @param message description of the failure
   */
  public DatasetAlreadyExistsException(String message) {
    super(message);
  }
}
