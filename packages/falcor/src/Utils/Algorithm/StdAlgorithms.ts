/**
 * libstdc++'s std::sort (introsort) and std::nth_element (introselect), step for step: native results
 * that depend on how equal elements end up ordered (alias tables, BVH splits) come out the same.
 */


function medianOf3<T>(arr: T[], a: number, b: number, c: number, less: (x: T, y: T) => boolean): number {
    // libstdc++ __median(a,b,c): returns iterator of median value.
    if (less(arr[a]!, arr[b]!)) {
        if (less(arr[b]!, arr[c]!)) return b;
        else if (less(arr[a]!, arr[c]!)) return c;
        else return a;
    } else if (less(arr[a]!, arr[c]!)) return a;
    else if (less(arr[b]!, arr[c]!)) return c;
    else return b;
}

function insertionSort<T>(arr: T[], first: number, last: number, less: (x: T, y: T) => boolean): void {
    for (let i = first + 1; i < last; i++) {
        const val = arr[i]!;
        let j = i - 1;
        while (j >= first && less(val, arr[j]!)) {
            arr[j + 1] = arr[j]!;
            j--;
        }
        arr[j + 1] = val;
    }
}

function heapSelect<T>(arr: T[], first: number, middle: number, last: number, less: (x: T, y: T) => boolean): void {
    // Partial sort [first, middle): make heap of [first, middle), then sift.
    const heapLen = middle - first;
    const siftDown = (start: number, end: number) => {
        let root = start;
        for (;;) {
            let child = 2 * (root - first) + 1 + first;
            if (child >= end) break;
            if (child + 1 < end && less(arr[child]!, arr[child + 1]!)) child++;
            if (less(arr[root]!, arr[child]!)) {
                const t = arr[root]!;
                arr[root] = arr[child]!;
                arr[child] = t;
                root = child;
            } else break;
        }
    };
    for (let start = first + Math.floor((heapLen - 2) / 2); start >= first; start--) siftDown(start, first + heapLen);
    for (let i = middle; i < last; i++) {
        if (less(arr[i]!, arr[first]!)) {
            const t = arr[first]!;
            arr[first] = arr[i]!;
            arr[i] = t;
            siftDown(first, first + heapLen);
        }
    }
}

function unguardedPartition<T>(arr: T[], first: number, last: number, pivotIdx: number, less: (x: T, y: T) => boolean): number {
    // libstdc++ __unguarded_partition_pivot: swap pivot to first, partition (first+1, last) with pivot arr[first].
    const t0 = arr[first]!;
    arr[first] = arr[pivotIdx]!;
    arr[pivotIdx] = t0;
    const pivot = arr[first]!;
    let lo = first + 1;
    let hi = last - 1;
    for (;;) {
        while (less(arr[lo]!, pivot)) lo++;
        while (less(pivot, arr[hi]!)) hi--;
        if (lo >= hi) return lo;
        const t = arr[lo]!;
        arr[lo] = arr[hi]!;
        arr[hi] = t;
        lo++;
        hi--;
    }
}

/** libstdc++ std::nth_element (__introselect) over arr[first, last). */
export function nthElement<T>(arr: T[], first: number, nth: number, last: number, less: (x: T, y: T) => boolean): void {
    if (first === last || nth === last) return;
    let depthLimit = 2 * Math.floor(Math.log2(last - first));
    while (last - first > 3) {
        if (depthLimit === 0) {
            // __heap_select(first, nth+1, last) then swap first/nth.
            heapSelect(arr, first, nth + 1, last, less);
            const t = arr[first]!;
            arr[first] = arr[nth]!;
            arr[nth] = t;
            return;
        }
        depthLimit--;
        const mid = first + ((last - first) >> 1);
        const pivotIdx = medianOf3(arr, first + 1, mid, last - 1, less);
        // libstdc++ passes the median VALUE via iter refs: it swaps arr[first+? ]...
        // __introselect uses __unguarded_partition_pivot(first, last) with
        // __median(*(first+1), *(first+(last-first)/2), *(last-1)) moved to first.
        const cut = unguardedPartition(arr, first, last, pivotIdx, less);
        if (cut <= nth) first = cut;
        else last = cut;
    }
    insertionSort(arr, first, last, less);
}

/** libstdc++ std::sort over arr[first, last): __introsort_loop (threshold 16) then __final_insertion_sort. */
export function stdSort<T>(arr: T[], less: (x: T, y: T) => boolean, first = 0, last = arr.length): void {
    if (last - first < 2) return;
    const introsortLoop = (lo: number, hi: number, depthLimit: number) => {
        while (hi - lo > 16) {
            if (depthLimit === 0) {
                // __partial_sort(lo, hi, hi): heap select then sort the heap.
                heapSelect(arr, lo, hi, hi, less);
                for (let end = hi - 1; end > lo; end--) {
                    const t = arr[lo]!;
                    arr[lo] = arr[end]!;
                    arr[end] = t;
                    heapSelect(arr, lo, end, end, less);
                }
                return;
            }
            depthLimit--;
            const mid = lo + ((hi - lo) >> 1);
            const cut = unguardedPartition(arr, lo, hi, medianOf3(arr, lo + 1, mid, hi - 1, less), less);
            introsortLoop(cut, hi, depthLimit);
            hi = cut;
        }
    };
    introsortLoop(first, last, 2 * Math.floor(Math.log2(last - first)));
    insertionSort(arr, first, last, less);
}
