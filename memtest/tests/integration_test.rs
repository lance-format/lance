use libc::{c_void, size_t};
use std::ptr;

// Import from the library we're testing
use memtest::{MemtestStats, memtest_get_stats, memtest_reset_stats};

extern "C" {
    fn malloc(size: size_t) -> *mut c_void;
    fn calloc(count: size_t, element_size: size_t) -> *mut c_void;
    fn realloc(ptr: *mut c_void, size: size_t) -> *mut c_void;
    fn free(ptr: *mut c_void);
    fn memalign(alignment: size_t, size: size_t) -> *mut c_void;
    fn posix_memalign(memptr: *mut *mut c_void, alignment: size_t, size: size_t) -> i32;
    fn aligned_alloc(alignment: size_t, size: size_t) -> *mut c_void;
}

fn get_stats() -> MemtestStats {
    let mut stats = MemtestStats {
        total_allocations: 0,
        total_deallocations: 0,
        total_bytes_allocated: 0,
        total_bytes_deallocated: 0,
        current_bytes: 0,
        peak_bytes: 0,
    };
    unsafe {
        memtest_get_stats(&mut stats as *mut MemtestStats);
    }
    stats
}

fn reset_stats() {
    memtest_reset_stats();
}

#[test]
fn test_malloc_free() {
    unsafe {
        reset_stats();
        let stats_after_reset = get_stats();

        let size = 1024;
        let ptr = malloc(size);
        assert!(!ptr.is_null());

        let stats_after_alloc = get_stats();
        // Check delta from reset
        assert_eq!(
            stats_after_alloc
                .total_allocations
                .saturating_sub(stats_after_reset.total_allocations),
            1
        );
        assert_eq!(
            stats_after_alloc
                .total_bytes_allocated
                .saturating_sub(stats_after_reset.total_bytes_allocated),
            size as u64
        );

        free(ptr);

        let stats_after_free = get_stats();
        assert_eq!(
            stats_after_free
                .total_deallocations
                .saturating_sub(stats_after_reset.total_deallocations),
            1
        );
        assert_eq!(
            stats_after_free
                .total_bytes_deallocated
                .saturating_sub(stats_after_reset.total_bytes_deallocated),
            size as u64
        );
    }
}

#[test]
fn test_calloc_free() {
    unsafe {
        reset_stats();
        let stats_baseline = get_stats();

        let count = 10;
        let element_size = 100;
        let total_size = count * element_size;

        let ptr = calloc(count, element_size);
        assert!(!ptr.is_null());

        // Verify memory is zeroed
        let slice = std::slice::from_raw_parts(ptr as *const u8, total_size);
        assert!(slice.iter().all(|&b| b == 0));

        let stats = get_stats();
        assert_eq!(
            stats
                .total_allocations
                .saturating_sub(stats_baseline.total_allocations),
            1
        );
        assert_eq!(
            stats
                .total_bytes_allocated
                .saturating_sub(stats_baseline.total_bytes_allocated),
            total_size as u64
        );

        free(ptr);

        let stats = get_stats();
        assert_eq!(
            stats
                .total_deallocations
                .saturating_sub(stats_baseline.total_deallocations),
            1
        );
    }
}

#[test]
fn test_realloc() {
    reset_stats();

    unsafe {
        // Start with malloc
        let ptr1 = malloc(100);
        assert!(!ptr1.is_null());

        let stats = get_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_bytes_allocated, 100);

        // Grow the allocation
        let ptr2 = realloc(ptr1, 200);
        assert!(!ptr2.is_null());

        let stats = get_stats();
        assert_eq!(stats.total_allocations, 2); // realloc counts as new allocation
        assert_eq!(stats.total_deallocations, 1); // old allocation freed
        assert_eq!(stats.total_bytes_allocated, 300); // 100 + 200
        assert_eq!(stats.total_bytes_deallocated, 100);
        assert_eq!(stats.current_bytes, 200);

        // Shrink the allocation
        let ptr3 = realloc(ptr2, 50);
        assert!(!ptr3.is_null());

        let stats = get_stats();
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.total_deallocations, 2);
        assert_eq!(stats.current_bytes, 50);

        free(ptr3);

        let stats = get_stats();
        assert_eq!(stats.current_bytes, 0);
    }
}

#[test]
fn test_realloc_null_is_malloc() {
    reset_stats();

    unsafe {
        // realloc(NULL, size) should behave like malloc
        let ptr = realloc(ptr::null_mut(), 100);
        assert!(!ptr.is_null());

        let stats = get_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_bytes_allocated, 100);

        free(ptr);
    }
}

#[test]
fn test_peak_tracking() {
    unsafe {
        reset_stats();
        let stats_baseline = get_stats();

        let ptr1 = malloc(1000);
        let ptr2 = malloc(500);
        let ptr3 = malloc(2000);

        let stats = get_stats();
        let current_bytes = stats
            .current_bytes
            .saturating_sub(stats_baseline.current_bytes);
        let peak_bytes = stats.peak_bytes.saturating_sub(stats_baseline.peak_bytes);
        assert_eq!(current_bytes, 3500);
        assert_eq!(peak_bytes, 3500);

        free(ptr3);

        let stats = get_stats();
        let current_bytes = stats
            .current_bytes
            .saturating_sub(stats_baseline.current_bytes);
        let peak_bytes = stats.peak_bytes.saturating_sub(stats_baseline.peak_bytes);
        assert_eq!(current_bytes, 1500);
        assert_eq!(peak_bytes, 3500); // Peak should remain

        let ptr4 = malloc(1000);

        let stats = get_stats();
        let current_bytes = stats
            .current_bytes
            .saturating_sub(stats_baseline.current_bytes);
        let peak_bytes = stats.peak_bytes.saturating_sub(stats_baseline.peak_bytes);
        assert_eq!(current_bytes, 2500);
        assert_eq!(peak_bytes, 3500); // Still the peak

        free(ptr1);
        free(ptr2);
        free(ptr4);
    }
}

#[test]
fn test_memalign() {
    unsafe {
        reset_stats();
        let stats_baseline = get_stats();

        let alignment = 128;
        let size = 1024;

        let ptr = memalign(alignment, size);
        assert!(!ptr.is_null());

        // Verify alignment
        assert_eq!(ptr as usize % alignment, 0);

        let stats = get_stats();
        assert_eq!(
            stats
                .total_allocations
                .saturating_sub(stats_baseline.total_allocations),
            1
        );
        assert_eq!(
            stats
                .total_bytes_allocated
                .saturating_sub(stats_baseline.total_bytes_allocated),
            size as u64
        );

        free(ptr);

        let stats = get_stats();
        assert_eq!(
            stats
                .total_deallocations
                .saturating_sub(stats_baseline.total_deallocations),
            1
        );
    }
}

#[test]
fn test_posix_memalign() {
    unsafe {
        reset_stats();
        let stats_baseline = get_stats();

        let alignment = 256;
        let size = 2048;
        let mut ptr: *mut c_void = ptr::null_mut();

        let ret = posix_memalign(&mut ptr as *mut *mut c_void, alignment, size);
        assert_eq!(ret, 0);
        assert!(!ptr.is_null());

        // Verify alignment
        assert_eq!(ptr as usize % alignment, 0);

        let stats = get_stats();
        assert_eq!(
            stats
                .total_allocations
                .saturating_sub(stats_baseline.total_allocations),
            1
        );
        assert_eq!(
            stats
                .total_bytes_allocated
                .saturating_sub(stats_baseline.total_bytes_allocated),
            size as u64
        );

        free(ptr);
    }
}

#[test]
fn test_aligned_alloc() {
    reset_stats();

    unsafe {
        let alignment = 64;
        let size = 512;

        let ptr = aligned_alloc(alignment, size);
        assert!(!ptr.is_null());

        // Verify alignment
        assert_eq!(ptr as usize % alignment, 0);

        let stats = get_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_bytes_allocated, size as u64);

        free(ptr);
    }
}

#[test]
fn test_large_alignment() {
    reset_stats();

    unsafe {
        // Test with page-sized alignment (4096 bytes)
        let alignment = 4096;
        let size = 8192;

        let ptr = memalign(alignment, size);
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % alignment, 0);

        // Write to the memory to ensure it's actually usable
        let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, size);
        slice[0] = 42;
        slice[size - 1] = 43;
        assert_eq!(slice[0], 42);
        assert_eq!(slice[size - 1], 43);

        let stats = get_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_bytes_allocated, size as u64);

        free(ptr);

        let stats = get_stats();
        assert_eq!(stats.current_bytes, 0);
    }
}

#[test]
fn test_mixed_aligned_unaligned() {
    reset_stats();

    unsafe {
        let ptr1 = malloc(1000); // Unaligned
        let ptr2 = memalign(128, 2000); // Aligned
        let ptr3 = malloc(500); // Unaligned
        let ptr4 = aligned_alloc(64, 1500); // Aligned

        let stats = get_stats();
        assert_eq!(stats.total_allocations, 4);
        assert_eq!(stats.total_bytes_allocated, 5000);
        assert_eq!(stats.current_bytes, 5000);

        // Verify alignments
        assert_eq!(ptr2 as usize % 128, 0);
        assert_eq!(ptr4 as usize % 64, 0);

        free(ptr1);
        free(ptr2);
        free(ptr3);
        free(ptr4);

        let stats = get_stats();
        assert_eq!(stats.total_deallocations, 4);
        assert_eq!(stats.current_bytes, 0);
    }
}

#[test]
fn test_free_null() {
    reset_stats();

    unsafe {
        // Freeing null should not crash or affect stats
        free(ptr::null_mut());

        let stats = get_stats();
        assert_eq!(stats.total_deallocations, 0);
    }
}

#[test]
fn test_reset_stats() {
    unsafe {
        let ptr1 = malloc(1000);
        let ptr2 = malloc(2000);

        let stats = get_stats();
        assert!(stats.total_allocations > 0);
        assert!(stats.total_bytes_allocated > 0);

        reset_stats();

        let stats = get_stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.total_deallocations, 0);
        assert_eq!(stats.total_bytes_allocated, 0);
        assert_eq!(stats.total_bytes_deallocated, 0);
        assert_eq!(stats.current_bytes, 0);
        assert_eq!(stats.peak_bytes, 0);

        // Clean up (stats won't count these since we reset)
        free(ptr1);
        free(ptr2);
    }
}

#[test]
fn test_alignment_with_write() {
    reset_stats();

    unsafe {
        // Test that aligned allocations are actually writable
        let alignment = 256;
        let size = 1024;

        let ptr = memalign(alignment, size);
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % alignment, 0);

        // Write pattern to memory
        let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, size);
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Verify pattern
        for (i, byte) in slice.iter().enumerate() {
            assert_eq!(*byte, (i % 256) as u8);
        }

        free(ptr);
    }
}
