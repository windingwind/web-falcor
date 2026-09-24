/** Worker entry for WorkerPool: runs kTasks by name. */

import { kTasks, type TaskName } from "./Tasks.js";

interface Request {
    id: number;
    task: TaskName;
    args: unknown;
}

self.onmessage = (e: MessageEvent<Request>) => {
    const { id, task, args } = e.data;
    try {
        const result = (kTasks[task] as (a: unknown) => { value: unknown; transfer?: Transferable[] })(args);
        (self as unknown as Worker).postMessage({ id, value: result.value }, result.transfer ?? []);
    } catch (err) {
        (self as unknown as Worker).postMessage({ id, error: String((err as Error)?.message ?? err) });
    }
};
