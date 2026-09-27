import type { Plugin } from "vite";

/** Serves (dev) and copies (build) web-falcor's runtime assets at `<base><path>`; `path` defaults to "web-falcor/". */
export declare function webFalcor(options?: { path?: string }): Plugin;
