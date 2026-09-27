/** Worker entry for WorkerPool: runs kTasks by name. */

import { kTasks, type TaskName } from "./Tasks.js";
import { loadMikkTSpace } from "../../Scene/TangentSpace.js";

// Tasks are synchronous; the MikkTSpace wasm (tangentsAndMerge) loads once before the first one runs.
const ready = loadMikkTSpace().catch(() => undefined);

interface Request {
    id: number;
    task: TaskName;
    args: unknown;
}

self.onmessage = async (e: MessageEvent<Request>) => {
    await ready;
    const { id, task, args } = e.data;
    try {
        const result = (kTasks[task] as (a: unknown) => { value: unknown; transfer?: Transferable[] })(args);
        (self as unknown as Worker).postMessage({ id, value: result.value }, result.transfer ?? []);
    } catch (err) {
        (self as unknown as Worker).postMessage({ id, error: String((err as Error)?.message ?? err) });
    }
};

// The pool holds tasks until this arrives (a worker script that fails to load never sends it).
(self as unknown as Worker).postMessage({ ready: true });
