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
package org.lance.compaction;

/**
 * Controls how compaction planning applies source budgets to cumulative totals across the tasks
 * selected for one plan.
 */
public enum SourceBudgetMode {
  /** Reject a task when adding it would make any cumulative source total exceed its budget. */
  HARD("hard"),
  /**
   * Always admit the first indivisible task. If it exceeds a cumulative budget, the plan stops
   * there; otherwise later tasks use the same cumulative checks as {@link #HARD}. If a byte budget
   * is configured but the first task has a source file without a recorded size, admit that task and
   * stop; hard mode reports an error.
   */
  SOFT("soft");

  private final String value;

  SourceBudgetMode(String value) {
    this.value = value;
  }

  public String getValue() {
    return value;
  }
}
