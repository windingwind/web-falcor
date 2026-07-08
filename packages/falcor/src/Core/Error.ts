/**
 * Error types mirroring Falcor/Core/Error.h.
 */

/** Base class for all web-falcor exceptions (Falcor::Exception). */
export class FalcorError extends Error {
    constructor(message: string) {
        super(message);
        this.name = this.constructor.name;
    }
}

/** Runtime error, e.g. invalid API usage detected at runtime (Falcor::RuntimeError). */
export class RuntimeError extends FalcorError {}

/** Argument error (Falcor::ArgumentError). */
export class ArgumentError extends FalcorError {}

/** Assertion failure (FALCOR_ASSERT). */
export class AssertionError extends FalcorError {}

/** Throws AssertionError if condition does not hold. */
export function assert(condition: unknown, message = "Assertion failed"): asserts condition {
    if (!condition) throw new AssertionError(message);
}

/** Marks Falcor features that cannot be supported on the web platform. See docs parity matrix. */
export class UnsupportedFeatureError extends FalcorError {
    constructor(feature: string, reason: string) {
        super(`Unsupported feature '${feature}': ${reason}`);
    }
}
