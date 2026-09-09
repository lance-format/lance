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
 * Thrown when a Lance operation fails and its API cannot expose a checked exception.
 *
 * <p>For example, scanner I/O failures can be handled separately from programming errors:
 *
 * <pre>{@code
 * try (ArrowReader reader = scanner.scanBatches()) {
 *   // Consume batches.
 * } catch (LanceException e) {
 *   // Handle the failed Lance operation.
 * }
 * }</pre>
 */
public class LanceException extends RuntimeException {
  /**
   * Creates an exception with a message describing the failed operation.
   *
   * @param message description of the failure
   */
  public LanceException(String message) {
    super(message);
  }

  /**
   * Creates an exception with a message and its underlying cause.
   *
   * @param message description of the failure
   * @param cause underlying cause of the failure
   */
  public LanceException(String message, Throwable cause) {
    super(message, cause);
  }
}
