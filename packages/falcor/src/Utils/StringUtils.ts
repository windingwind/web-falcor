/**
 * String helpers mirroring Falcor/Utils/StringUtils.h, plus the path helpers hasExtension and
 * getExtensionFromPath from Core/Platform/OS.h.
 */

const kWhitespace = " \n\r\t";

export function removeLeadingWhitespace(str: string, whitespace = kWhitespace): string {
    let i = 0;
    while (i < str.length && whitespace.includes(str[i]!)) i++;
    return str.slice(i);
}

export function removeTrailingWhitespace(str: string, whitespace = kWhitespace): string {
    let i = str.length;
    while (i > 0 && whitespace.includes(str[i - 1]!)) i--;
    return str.slice(0, i);
}

export function removeLeadingTrailingWhitespace(str: string, whitespace = kWhitespace): string {
    return removeTrailingWhitespace(removeLeadingWhitespace(str, whitespace), whitespace);
}

/** Replaces every character of `characters` in `str` by `replacement`. */
export function replaceCharacters(str: string, characters: string, replacement: string): string {
    return Array.from(str, (c) => (characters.includes(c) ? replacement : c)).join("");
}

/** "1023 B", "1.00 kB" ... "10.00 TB", with 1024 steps as natively. */
export function formatByteSize(size: number): string {
    if (size < 1024) return `${size} B`;
    const units = ["kB", "MB", "GB", "TB"];
    let value = size / 1024;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit++;
    }
    return `${value.toFixed(2)} ${units[unit]}`;
}

export function encodeBase64(data: Uint8Array): string {
    let binary = "";
    for (const b of data) binary += String.fromCharCode(b);
    return btoa(binary);
}

export function decodeBase64(encoded: string): Uint8Array {
    return Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0));
}

/** Mirrors decodeURI: %XX bytes and '+' as a space, never throwing (unlike the JS builtins). */
export function decodeURI(input: string): string {
    const bytes: number[] = [];
    const utf8 = new TextEncoder();
    for (let i = 0; i < input.length; i++) {
        const c = input[i]!;
        if (c === "%") {
            // Like native: a '%' without two following characters is dropped.
            if (i + 2 < input.length) {
                bytes.push(parseInt(input.slice(i + 1, i + 3), 16) & 0xff);
                i += 2;
            }
        } else if (c === "+") bytes.push(0x20);
        else bytes.push(...utf8.encode(c));
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
}

/** Lower-case extension without the dot; "" for none or a dot file (std::filesystem semantics). */
export function getExtensionFromPath(path: string): string {
    const name = path.slice(path.lastIndexOf("/") + 1);
    const dot = name.lastIndexOf(".");
    return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

/** Case-insensitive; `ext` may start with a dot, and "" matches paths without an extension. */
export function hasExtension(path: string, ext: string): boolean {
    return getExtensionFromPath(path) === (ext.startsWith(".") ? ext.slice(1) : ext).toLowerCase();
}
