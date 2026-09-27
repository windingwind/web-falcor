/**
 * Mirrors Falcor's TaskManager / BS::thread_pool for scene-build CPU work: a pool of module
 * workers running the tasks in ./Tasks.ts. Where workers are unavailable (Node, tests) or their
 * script fails to load, tasks run inline on the calling thread.
 */

import { kTasks, type TaskName } from "./Tasks.js";
import { Logger } from "../Logger.js";

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
    /** Resolves true once the workers loaded, false if one failed to (then tasks run inline). */
    private ready: Promise<boolean> = Promise.resolve(false);

    /** The shared pool (hardwareConcurrency - 1 workers, at most 8). */
    static get(): WorkerPool {
        return (WorkerPool.instance ??= new WorkerPool());
    }

    private constructor() {
        if (typeof Worker === "undefined") return;
        const count = Math.max(1, Math.min(8, (globalThis.navigator?.hardwareConcurrency ?? 4) - 1));
        const loaded: Promise<boolean>[] = [];
        for (let i = 0; i < count; i++) {
            const worker = new Worker(new URL("./TaskWorker.ts", import.meta.url), { type: "module", name: `WorkerPool ${i}` });
            let signal: (ok: boolean) => void = () => {};
            loaded.push(new Promise((r) => (signal = r)));
            // A load failure (e.g. the script is not served) arrives as an error event before the ready message.
            worker.onerror = () => signal(false);
            worker.onmessage = (e: MessageEvent<{ id: number; value?: unknown; error?: string; ready?: boolean }>) => {
                if (e.data.ready) return signal(true);
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
        this.ready = Promise.all(loaded).then((ok) => {
            if (ok.every((v) => v)) return true;
            Logger.warning("WorkerPool: the task worker failed to load; scene-build tasks run on the main thread.");
            for (const w of this.workers) w.terminate();
            this.workers.length = 0;
            return false;
        });
    }

    get threadCount(): number {
        return this.workers.length;
    }

    /** Runs `task` on the least-loaded worker (inline without workers). Buffers in `transfer` move to the worker. */
    async run<T extends TaskName>(task: T, args: TaskArgs<T>, transfer: Transferable[] = []): Promise<TaskValue<T>> {
        if (!(await this.ready)) return (kTasks[task] as (a: TaskArgs<T>) => { value: TaskValue<T> })(args).value;
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
