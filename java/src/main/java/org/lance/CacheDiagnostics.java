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

import org.lance.protobuf.CacheDiagnosticsProto;

import com.google.protobuf.InvalidProtocolBufferException;

import java.util.List;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;
import java.util.stream.Collectors;

/**
 * Immutable diagnostics for both caches owned by a {@link Session}.
 *
 * <p>Activity counts are cumulative for the lifetime of a cache wrapper and do not reset when the
 * cache is cleared. Backend accounting belongs to a physical cache pool. Equal pool identifiers
 * mean two snapshots refer to the same pool, whose occupancy must not be summed twice.
 *
 * <p>Fields are sampled independently and may change while cache operations are running. Optional
 * values are empty when a backend cannot report the measurement. Accounted bytes are cache policy
 * weights; they are not process resident memory.
 */
public final class CacheDiagnostics {
  private final CacheDiagnosticsProto.SessionCacheDiagnostics proto;

  private CacheDiagnostics(CacheDiagnosticsProto.SessionCacheDiagnostics proto) {
    this.proto = proto;
  }

  static CacheDiagnostics fromBytes(byte[] bytes) {
    try {
      return new CacheDiagnostics(CacheDiagnosticsProto.SessionCacheDiagnostics.parseFrom(bytes));
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalStateException("Native cache diagnostics returned invalid protobuf", e);
    }
  }

  /** Returns diagnostics for opened indices and index payloads. */
  public Snapshot getIndex() {
    return new Snapshot(proto.getIndex());
  }

  /** Returns diagnostics for dataset and file metadata. */
  public Snapshot getMetadata() {
    return new Snapshot(proto.getMetadata());
  }

  /** One logical cache wrapper and its physical backend. */
  public static final class Snapshot {
    private final CacheDiagnosticsProto.CacheDiagnosticsSnapshot proto;

    private Snapshot(CacheDiagnosticsProto.CacheDiagnosticsSnapshot proto) {
      this.proto = proto;
    }

    /** Returns cumulative wrapper activity. */
    public Activity getActivity() {
      return new Activity(proto.getActivity());
    }

    /** Returns shared physical backend accounting. */
    public Backend getBackend() {
      return new Backend(proto.getBackend());
    }

    /**
     * Returns accounted occupancy divided by configured capacity.
     *
     * <p>The value is absent for zero or unknown capacity, or unavailable occupancy. It is not
     * clamped.
     */
    public OptionalDouble getUtilization() {
      return proto.hasUtilization()
          ? OptionalDouble.of(proto.getUtilization())
          : OptionalDouble.empty();
    }

    /**
     * Returns explicitly requested per-type detail.
     *
     * <p>The value is absent for ordinary constant-cost snapshots.
     */
    public Optional<ByType> getByType() {
      return proto.hasByType() ? Optional.of(new ByType(proto.getByType())) : Optional.empty();
    }
  }

  /** Bounded activity and explicitly scanned occupancy grouped by stable cache key type. */
  public static final class ByType {
    private final CacheDiagnosticsProto.CacheByTypeDiagnostics proto;

    private ByType(CacheDiagnosticsProto.CacheByTypeDiagnostics proto) {
      this.proto = proto;
    }

    /** Returns cumulative activity rows, including an {@code other} row after local overflow. */
    public List<TypeActivity> getActivity() {
      return proto.getActivityList().stream()
          .map(TypeActivity::new)
          .collect(Collectors.toUnmodifiableList());
    }

    /** Returns scanned occupancy, or empty when the backend cannot enumerate tagged records. */
    public Optional<OccupancyByType> getOccupancy() {
      return proto.hasOccupancy()
          ? Optional.of(new OccupancyByType(proto.getOccupancy()))
          : Optional.empty();
    }

    /** Returns activity events assigned to the bounded exported {@code other} label. */
    public long getTypeLabelOverflowEvents() {
      return proto.getTypeLabelOverflowEvents();
    }
  }

  /** Activity for one stable cache key type. */
  public static final class TypeActivity {
    private final CacheDiagnosticsProto.CacheTypeActivityDiagnostics proto;

    private TypeActivity(CacheDiagnosticsProto.CacheTypeActivityDiagnostics proto) {
      this.proto = proto;
    }

    /** Returns the stable cache key type name. */
    public String getTypeName() {
      return proto.getTypeName();
    }

    /** Returns cumulative activity for this type. */
    public Activity getActivity() {
      return new Activity(proto.getActivity());
    }
  }

  /** Approximate occupancy collected by scanning tagged backend records. */
  public static final class OccupancyByType {
    private final CacheDiagnosticsProto.CacheOccupancyByTypeDiagnostics proto;

    private OccupancyByType(CacheDiagnosticsProto.CacheOccupancyByTypeDiagnostics proto) {
      this.proto = proto;
    }

    /** Returns occupancy rows for tagged resident entries. */
    public List<TypeOccupancy> getTypes() {
      return proto.getTypesList().stream()
          .map(TypeOccupancy::new)
          .collect(Collectors.toUnmodifiableList());
    }

    /** Returns accounted weight for entries written through an untagged backend API. */
    public long getUntaggedSizeBytes() {
      return proto.getUntaggedSizeBytes();
    }

    /** Returns resident entries written through an untagged backend API. */
    public long getUntaggedNumEntries() {
      return proto.getUntaggedNumEntries();
    }
  }

  /** Approximate resident occupancy for one stable cache key type. */
  public static final class TypeOccupancy {
    private final CacheDiagnosticsProto.CacheTypeOccupancyDiagnostics proto;

    private TypeOccupancy(CacheDiagnosticsProto.CacheTypeOccupancyDiagnostics proto) {
      this.proto = proto;
    }

    /** Returns the stable cache key type name. */
    public String getTypeName() {
      return proto.getTypeName();
    }

    /** Returns accounted backend weight in bytes. */
    public long getSizeBytes() {
      return proto.getSizeBytes();
    }

    /** Returns the number of resident entries carrying this type tag. */
    public long getNumEntries() {
      return proto.getNumEntries();
    }
  }

  /** Cumulative activity for a logical cache wrapper. */
  public static final class Activity {
    private final CacheDiagnosticsProto.CacheActivityDiagnostics proto;

    private Activity(CacheDiagnosticsProto.CacheActivityDiagnostics proto) {
      this.proto = proto;
    }

    /** Returns successful cache lookups. */
    public long getHits() {
      return proto.getHits();
    }

    /** Returns cache lookups that did not find a value. */
    public long getMisses() {
      return proto.getMisses();
    }

    /** Returns cache lookups that failed before loading a value. */
    public long getLookupErrors() {
      return proto.getLookupErrors();
    }

    /** Returns lookups whose stored value had the requested key but a different type. */
    public long getTypeMismatches() {
      return proto.getTypeMismatches();
    }

    /** Returns loader executions started by this cache wrapper. */
    public long getLoadsStarted() {
      return proto.getLoadsStarted();
    }

    /** Returns loader executions that completed successfully. */
    public long getLoadsSucceeded() {
      return proto.getLoadsSucceeded();
    }

    /** Returns loader executions that returned an error. */
    public long getLoadsFailed() {
      return proto.getLoadsFailed();
    }

    /** Returns loader executions cancelled before completion. */
    public long getLoadsCancelled() {
      return proto.getLoadsCancelled();
    }

    /** Returns loader executions currently in progress. */
    public long getLoadsInFlight() {
      return proto.getLoadsInFlight();
    }

    /** Returns cumulative nanoseconds spent in successful loader executions. */
    public long getLoadSuccessDurationNs() {
      return proto.getLoadSuccessDurationNs();
    }

    /** Returns cumulative nanoseconds spent in loader executions that returned an error. */
    public long getLoadErrorDurationNs() {
      return proto.getLoadErrorDurationNs();
    }

    /** Returns cumulative nanoseconds spent in cancelled loader executions. */
    public long getLoadCancelledDurationNs() {
      return proto.getLoadCancelledDurationNs();
    }

    /** Returns work attributed to explicit prewarming. */
    public WarmActivity getWarm() {
      return new WarmActivity(proto.getWarm());
    }
  }

  /** Cumulative work attributed to explicit prewarming. */
  public static final class WarmActivity {
    private final CacheDiagnosticsProto.CacheWarmActivityDiagnostics proto;

    private WarmActivity(CacheDiagnosticsProto.CacheWarmActivityDiagnostics proto) {
      this.proto = proto;
    }

    /** Returns warm cache calls, including hits, loads, and direct insertions. */
    public long getAttempts() {
      return proto.getAttempts();
    }

    /** Returns warm calls served without starting a new materialization. */
    public long getHits() {
      return proto.getHits();
    }

    /** Returns warm materializations started, including direct insertions. */
    public long getLoadsStarted() {
      return proto.getLoadsStarted();
    }

    /** Returns warm materializations successfully submitted to the cache. */
    public long getLoadsSucceeded() {
      return proto.getLoadsSucceeded();
    }

    /** Returns warm loader executions that returned an error. */
    public long getLoadsFailed() {
      return proto.getLoadsFailed();
    }

    /** Returns warm loader executions cancelled before completion. */
    public long getLoadsCancelled() {
      return proto.getLoadsCancelled();
    }

    /** Returns accounted bytes from successful warm materializations. */
    public long getLoadBytes() {
      return proto.getLoadBytes();
    }

    /** Returns failed warm cache calls, including loader failures. */
    public long getErrors() {
      return proto.getErrors();
    }
  }

  /** Bounded physical backend implementation class. */
  public enum BackendKind {
    /** Lance's synchronous weighted cache. */
    QUICK,
    /** The Moka asynchronous weighted cache. */
    MOKA,
    /** A cache backend supplied through the backend registry. */
    CUSTOM
  }

  /** Shared physical backend accounting and support information. */
  public static final class Backend {
    private final CacheDiagnosticsProto.CacheBackendDiagnostics proto;

    private Backend(CacheDiagnosticsProto.CacheBackendDiagnostics proto) {
      this.proto = proto;
    }

    /** Returns the backend implementation class. */
    public BackendKind getKind() {
      switch (proto.getKind()) {
        case QUICK:
          return BackendKind.QUICK;
        case MOKA:
          return BackendKind.MOKA;
        case CUSTOM:
          return BackendKind.CUSTOM;
        default:
          throw new IllegalStateException("Unknown cache backend kind: " + proto.getKind());
      }
    }

    /** Returns the physical pool identifier, when the backend provides one. */
    public OptionalLong getPoolId() {
      return optionalLong(proto.hasPoolId(), proto.getPoolId());
    }

    /** Returns the configured weighted capacity in bytes, when known. */
    public OptionalLong getCapacityBytes() {
      return optionalLong(proto.hasCapacityBytes(), proto.getCapacityBytes());
    }

    /** Returns whether writes can be admitted, when the backend reports this state. */
    public Optional<Boolean> getEnabled() {
      return proto.hasEnabled() ? Optional.of(proto.getEnabled()) : Optional.empty();
    }

    /** Returns the currently accounted weight in bytes, when available. */
    public OptionalLong getSizeBytes() {
      return optionalLong(proto.hasSizeBytes(), proto.getSizeBytes());
    }

    /** Returns the current number of resident entries, when available. */
    public OptionalLong getNumEntries() {
      return optionalLong(proto.hasNumEntries(), proto.getNumEntries());
    }

    /** Returns cumulative backend write attempts, when available. */
    public OptionalLong getWriteAttempts() {
      return optionalLong(proto.hasWriteAttempts(), proto.getWriteAttempts());
    }

    /** Returns cumulative attempted write weight in bytes, when available. */
    public OptionalLong getWriteBytes() {
      return optionalLong(proto.hasWriteBytes(), proto.getWriteBytes());
    }

    /** Returns cumulative removals caused by weighted capacity, when available. */
    public OptionalLong getSizeRemovals() {
      return optionalLong(proto.hasSizeRemovals(), proto.getSizeRemovals());
    }

    /** Returns cumulative weight removed because of capacity, when available. */
    public OptionalLong getSizeRemovedBytes() {
      return optionalLong(proto.hasSizeRemovedBytes(), proto.getSizeRemovedBytes());
    }

    /** Returns writes rejected because the backend was disabled, when available. */
    public OptionalLong getDisabledWriteRejections() {
      return optionalLong(proto.hasDisabledWriteRejections(), proto.getDisabledWriteRejections());
    }

    /** Returns loads that bypassed a disabled backend, when available. */
    public OptionalLong getDisabledBypasses() {
      return optionalLong(proto.hasDisabledBypasses(), proto.getDisabledBypasses());
    }

    /** Returns writes rejected after losing their loading placeholder, when available. */
    public OptionalLong getLostPlaceholderRejections() {
      return optionalLong(
          proto.hasLostPlaceholderRejections(), proto.getLostPlaceholderRejections());
    }

    /** Returns values whose weight exceeded the accounting range, when available. */
    public OptionalLong getWeightSaturations() {
      return optionalLong(proto.hasWeightSaturations(), proto.getWeightSaturations());
    }

    /** Returns whether write rejection counters cover every rejection reason. */
    public boolean getWriteRejectionsComplete() {
      return proto.getWriteRejectionsComplete();
    }

    /** Returns removals of resident entries, when the backend reports them. */
    public OptionalLong getResidentEvictions() {
      return optionalLong(proto.hasResidentEvictions(), proto.getResidentEvictions());
    }

    /** Returns admitted writes, when the backend reports them. */
    public OptionalLong getAdmissions() {
      return optionalLong(proto.hasAdmissions(), proto.getAdmissions());
    }

    /** Returns loads shared by concurrent requests, when the backend reports them. */
    public OptionalLong getCoalescedLoads() {
      return optionalLong(proto.hasCoalescedLoads(), proto.getCoalescedLoads());
    }

    private static OptionalLong optionalLong(boolean present, long value) {
      return present ? OptionalLong.of(value) : OptionalLong.empty();
    }
  }
}
