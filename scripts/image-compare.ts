/**
 * Command-line ImageCompare (mirrors Falcor/Source/Tools/ImageCompare):
 *   npm run image-compare -- [-l] [-m metric] [-t threshold] [-a] [-e heatmap] image1 image2
 * Prints the error and exits with 0 when it is within the threshold (default 0), else 1.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { basename } from "node:path";
import { compareImages, encodeCompareImage, errorMetrics, generateHeatMap, loadCompareImage } from "../packages/falcor/src/Utils/Image/ImageCompare.js";

/** std::cout's default formatting of a double (%g, 6 significant digits). */
function formatDouble(x: number): string {
    if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
    if (x === 0) return "0";
    const exp = Math.floor(Math.log10(Math.abs(Number(x.toPrecision(6)))));
    if (exp < -4 || exp >= 6) {
        const [m, e] = x.toExponential(5).split("e");
        const mant = m!.includes(".") ? m!.replace(/0+$/, "").replace(/\.$/, "") : m!;
        const n = Number(e);
        return `${mant}e${n < 0 ? "-" : "+"}${String(Math.abs(n)).padStart(2, "0")}`;
    }
    return String(Number(x.toPrecision(6)));
}

const usage = "Utility to compare images.\n  image-compare [-l] [-m metric] [-t threshold] [-a] [-e filename] image1 image2";
const printMetrics = (out: (s: string) => void) => {
    out("Available error metrics:");
    for (const m of errorMetrics) out(`  ${m.name} - ${m.desc}`);
};

async function main(argv: string[]): Promise<number> {
    let metric = errorMetrics[0]!.name;
    let threshold = 0;
    let alpha = false;
    let heatMap = "";
    const images: string[] = [];
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i]!;
        if (a === "-h" || a === "--help") return console.log(usage), 0;
        else if (a === "-l") return printMetrics(console.log), 0;
        else if (a === "-m") metric = argv[++i]!;
        else if (a === "-t") threshold = Number(argv[++i]);
        else if (a === "-a") alpha = true;
        else if (a === "-e") heatMap = argv[++i]!;
        else images.push(a);
    }
    if (images.length !== 2) return console.error(usage), 1;
    if (!errorMetrics.some((m) => m.name === metric)) {
        console.error(`Unknown error metric '${metric}'.`);
        printMetrics(console.error);
        return 1;
    }
    const load = async (path: string) => {
        try {
            return await loadCompareImage(new Uint8Array(readFileSync(path)), basename(path));
        } catch (e) {
            console.error(`Cannot load image from '${path}' (Error: ${(e as Error).message}).`);
            return null;
        }
    };
    const a = await load(images[0]!);
    const b = a && (await load(images[1]!));
    if (!a || !b) return 1;
    if (a.width !== b.width || a.height !== b.height) return console.error("Cannot compare images with different resolutions."), 1;
    const { error, errorMap } = compareImages(a, b, metric, alpha);
    if (heatMap) {
        try {
            writeFileSync(heatMap, await encodeCompareImage(generateHeatMap(a.width, a.height, errorMap), heatMap));
        } catch (e) {
            console.error(`Cannot save image to '${heatMap}' (Error: ${(e as Error).message}).`);
        }
    }
    console.log(formatDouble(error));
    // Treat nans and infs as errors.
    return Number.isFinite(error) && error <= threshold ? 0 : 1;
}

process.exit(await main(process.argv.slice(2)));
