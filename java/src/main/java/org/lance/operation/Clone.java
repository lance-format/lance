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
package org.lance.operation;

import com.google.common.base.MoreObjects;

import java.util.Objects;
import java.util.Optional;

/** Clone a dataset version into a new dataset. */
public final class Clone implements Operation {
  private final boolean shallow;
  private final Optional<String> refName;
  private final long refVersion;
  private final String refPath;
  private final Optional<String> branchName;

  private Clone(
      boolean shallow, String refName, long refVersion, String refPath, String branchName) {
    this.shallow = shallow;
    this.refName = Optional.ofNullable(refName);
    this.refVersion = refVersion;
    this.refPath = Objects.requireNonNull(refPath, "refPath must not be null");
    this.branchName = Optional.ofNullable(branchName);
  }

  public boolean isShallow() {
    return shallow;
  }

  public Optional<String> getRefName() {
    return refName;
  }

  public long getRefVersion() {
    return refVersion;
  }

  public String getRefPath() {
    return refPath;
  }

  public Optional<String> getBranchName() {
    return branchName;
  }

  @Override
  public String name() {
    return "Clone";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("shallow", shallow)
        .add("refName", refName)
        .add("refVersion", refVersion)
        .add("refPath", refPath)
        .add("branchName", branchName)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Clone clone = (Clone) o;
    return shallow == clone.shallow
        && refVersion == clone.refVersion
        && Objects.equals(refName, clone.refName)
        && Objects.equals(refPath, clone.refPath)
        && Objects.equals(branchName, clone.branchName);
  }

  @Override
  public int hashCode() {
    return Objects.hash(shallow, refName, refVersion, refPath, branchName);
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private boolean shallow;
    private String refName;
    private long refVersion;
    private String refPath;
    private String branchName;

    public Builder shallow(boolean shallow) {
      this.shallow = shallow;
      return this;
    }

    public Builder refName(String refName) {
      this.refName = refName;
      return this;
    }

    public Builder refVersion(long refVersion) {
      this.refVersion = refVersion;
      return this;
    }

    public Builder refPath(String refPath) {
      this.refPath = refPath;
      return this;
    }

    public Builder branchName(String branchName) {
      this.branchName = branchName;
      return this;
    }

    public Clone build() {
      return new Clone(shallow, refName, refVersion, refPath, branchName);
    }
  }
}
