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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;

import java.util.Optional;

public class Ref {
  private final Optional<Long> versionNumber;
  private final Optional<String> branchName;
  private final Optional<String> tagName;

  public Ref(Optional<Long> versionNumber, Optional<String> branchName, Optional<String> tagName) {
    this.versionNumber = versionNumber;
    this.branchName = branchName;
    this.tagName = tagName;
  }

  public Optional<Long> getVersionNumber() {
    return versionNumber;
  }

  public Optional<String> getBranchName() {
    return branchName;
  }

  public Optional<String> getTagName() {
    return tagName;
  }

  /** Creates a reference to a specific version on the default branch. */
  public static Ref ofMain(long versionNumber) {
    Preconditions.checkArgument(versionNumber > 0, "versionNumber must be greater than 0");
    return new Ref(Optional.of(versionNumber), Optional.empty(), Optional.empty());
  }

  /** Creates a reference to the latest version on the default branch. */
  public static Ref ofMain() {
    return new Ref(Optional.empty(), Optional.empty(), Optional.empty());
  }

  /**
   * Creates a reference to the latest version on a branch.
   *
   * <p>In reference contexts, {@code "main"} is an alias for the default branch.
   */
  public static Ref ofBranch(String branchName) {
    Preconditions.checkArgument(
        branchName != null && !branchName.isEmpty(), "branchName must not be empty");
    return new Ref(Optional.empty(), Optional.of(branchName), Optional.empty());
  }

  /**
   * Creates a reference to a specific version on a branch.
   *
   * <p>In reference contexts, {@code "main"} is an alias for the default branch.
   */
  public static Ref ofBranch(String branchName, long versionNumber) {
    Preconditions.checkArgument(
        branchName != null && !branchName.isEmpty(), "branchName must not be empty");
    Preconditions.checkArgument(versionNumber > 0, "versionNumber must be greater than 0");
    return new Ref(Optional.of(versionNumber), Optional.of(branchName), Optional.empty());
  }

  public static Ref ofTag(String tagName) {
    Preconditions.checkArgument(tagName != null && !tagName.isEmpty(), "tagName must not be empty");
    return new Ref(Optional.empty(), Optional.empty(), Optional.of(tagName));
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("versionNumber", versionNumber.orElse(null))
        .add("branchName", branchName.orElse(null))
        .add("tagName", tagName.orElse(null))
        .toString();
  }
}
