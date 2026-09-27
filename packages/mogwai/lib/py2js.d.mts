export declare function convertFiles(options: { inputs: string[]; out?: string; root?: string; check?: boolean; core: unknown; pyodideDir: string; log?: Pick<Console, "log" | "warn" | "error"> }): Promise<number>;
export declare function parseArgs(argv: string[]): { inputs: string[]; out?: string; root?: string; check: boolean };
export declare function pythonParser(pyodideDir: string, kAstDumper: string): Promise<(src: string) => unknown>;
