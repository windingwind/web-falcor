/**
 * Mirrors Falcor's TaskManager / BS::thread_pool for scene-build CPU work: a pool of module
 * workers running the tasks in ./Tasks.ts. Where workers are unavailable (Node, tests), tasks
 * run inline on the calling thread.
 */

import { kTasks, type TaskName } from "./Tasks.js";

type TaskArgs<T extends TaskName> = Parameters<(typeof kTasks)[T]>[0];
type TaskValue<T extends TaskName> = ReturnType<(typeof kTasks)[T]>["value"];

interface Pending {
    resolve: (v: unknown) => void;
    reject: (e: Error) => void;
}

export class WorkerPool {
    private static instance: WorkerPool | null = null;
    private readonly workers: Worker[] = [];
    private readonly load: number[] = [];
    private readonly pending = new Map<number, Pending>();
    private nextId = 1;

    /** The shared pool (hardwareConcurrency - 1 workers, at most 8). */
    static get(): WorkerPool {
        return (WorkerPool.instance ??= new WorkerPool());
    }

    private constructor() {
        if (typeof Worker === "undefined") return;
        const count = Math.max(1, Math.min(8, (globalThis.navigator?.hardwareConcurrency ?? 4) - 1));
        for (let i = 0; i < count; i++) {
            const worker = new Worker(new URL("./TaskWorker.ts", import.meta.url), { type: "module", name: `WorkerPool ${i}` });
            worker.onmessage = (e: MessageEvent<{ id: number; value?: unknown; error?: string }>) => {
                const p = this.pending.get(e.data.id);
                if (!p) return;
                this.pending.delete(e.data.id);
                this.load[i]!--;
                if (e.data.error !== undefined) p.reject(new Error(e.data.error));
                else p.resolve(e.data.value);
            };
            this.workers.push(worker);
            this.load.push(0);
        }
    }

    get threadCount(): number {
        return this.workers.length;
    }

    /** Runs `task` on the least-loaded worker (inline without workers). Buffers in `transfer` move to the worker. */
    run<T extends TaskName>(task: T, args: TaskArgs<T>, transfer: Transferable[] = []): Promise<TaskValue<T>> {
        if (this.workers.length === 0) return Promise.resolve((kTasks[task] as (a: TaskArgs<T>) => { value: TaskValue<T> })(args).value);
        let w = 0;
        for (let i = 1; i < this.workers.length; i++) if (this.load[i]! < this.load[w]!) w = i;
        const id = this.nextId++;
        this.load[w]!++;
        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
            this.workers[w]!.postMessage({ id, task, args }, transfer);
        });
    }
}
