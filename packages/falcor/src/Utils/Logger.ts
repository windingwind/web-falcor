/**
 * Logging facility mirroring Falcor/Utils/Logger.h.
 */

export enum LogLevel {
    Disabled = 0,
    Fatal = 1,
    Error = 2,
    Warning = 3,
    Info = 4,
    Debug = 5,
}

export class Logger {
    static level: LogLevel = LogLevel.Info;

    static fatal(msg: string): void {
        if (Logger.level >= LogLevel.Fatal) console.error(`(Fatal) ${msg}`);
    }
    static error(msg: string): void {
        if (Logger.level >= LogLevel.Error) console.error(`(Error) ${msg}`);
    }
    static warning(msg: string): void {
        if (Logger.level >= LogLevel.Warning) console.warn(`(Warning) ${msg}`);
    }
    static info(msg: string): void {
        if (Logger.level >= LogLevel.Info) console.info(`(Info) ${msg}`);
    }
    static debug(msg: string): void {
        if (Logger.level >= LogLevel.Debug) console.debug(`(Debug) ${msg}`);
    }
}
