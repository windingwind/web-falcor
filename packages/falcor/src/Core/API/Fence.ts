/**
 * GPU fence mirroring Falcor/Core/API/Fence.h.
 *
 * WebGPU exposes no user-visible timeline semaphores; fence values are tracked
 * host-side and signaled via queue.onSubmittedWorkDone() promises. Semantics
 * (monotonic 64-bit values, wait-for-value) are preserved for host<->GPU sync;
 * GPU-side waits between queues don't exist (single queue).
 */

export const kAutoSignalValue = 0xffffffffffffffffn;

export class Fence {
    private signaledValue = 0n;
    private pending: { value: bigint; promise: Promise<void> }[] = [];

    constructor(private readonly queue: GPUQueue, public readonly initialValue = 0n) {
        this.signaledValue = initialValue;
    }

    /** Mirrors Fence::signal: value is considered signaled once currently-submitted work completes. */
    signal(value: bigint = kAutoSignalValue): bigint {
        const v = value === kAutoSignalValue ? this.signaledValue + 1n : value;
        const promise = this.queue.onSubmittedWorkDone().then(() => {
            if (v > this.signaledValue) this.signaledValue = v;
            this.pending = this.pending.filter((p) => p.value > this.signaledValue);
        });
        this.pending.push({ value: v, promise });
        return v;
    }

    /** Mirrors Fence::wait on host (async divergence, DESIGN.md §9). */
    async wait(value?: bigint): Promise<void> {
        const target = value ?? this.maxPendingValue();
        while (this.signaledValue < target) {
            const next = this.pending.find((p) => p.value >= target) ?? this.pending[0];
            if (!next) break;
            await next.promise;
        }
    }

    getSignaledValue(): bigint {
        return this.signaledValue;
    }

    private maxPendingValue(): bigint {
        return this.pending.reduce((m, p) => (p.value > m ? p.value : m), this.signaledValue);
    }
}
