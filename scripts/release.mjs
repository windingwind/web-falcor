#!/usr/bin/env node
/**
 * Bumps every package to one version, commits, tags vX.Y.Z and pushes: the tag starts .github/workflows/release.yml,
 * which tests and publishes @web-falcor/falcor, /render-passes and /mogwai.
 *
 *   npm run release -- <X.Y.Z | patch | minor | major> [--no-push] [--retag]
 *
 * --no-push  commit and tag locally only (push later with `git push && git push origin vX.Y.Z`).
 * --retag    reuse an existing tag whose version npm doesn't have yet (e.g. a release that failed before publishing).
 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const kPackages = ["falcor", "render-passes", "mogwai"];

const args = process.argv.slice(2);
const push = !args.includes("--no-push");
const retag = args.includes("--retag");
const spec = args.find((a) => !a.startsWith("--"));

const fail = (msg) => {
    console.error(`error: ${msg}`);
    process.exit(1);
};
const git = (...a) => execFileSync("git", a, { cwd: repoRoot, encoding: "utf8" }).trim();
const run = (cmd, ...a) => execFileSync(cmd, a, { cwd: repoRoot, stdio: "inherit" });

if (!spec) fail("usage: npm run release -- <X.Y.Z | patch | minor | major> [--no-push] [--retag]");

// The version: explicit, or the current one bumped.
const current = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8")).version;
const [maj, min, pat] = current.split(".").map(Number);
const version = { major: `${maj + 1}.0.0`, minor: `${maj}.${min + 1}.0`, patch: `${maj}.${min}.${pat + 1}` }[spec] ?? spec;
if (!/^\d+\.\d+\.\d+(-[\w.]+)?$/.test(version)) fail(`'${spec}' is not a version or patch/minor/major`);
const tag = `v${version}`;

// A clean main: the release commit contains only the version bump.
if (git("status", "--porcelain")) fail("uncommitted changes: commit or stash them first");
const branch = git("rev-parse", "--abbrev-ref", "HEAD");
if (branch !== "main") fail(`releases are cut from main (on '${branch}')`);

// An existing tag is reused only with --retag, and only if that version never reached npm.
const localTag = git("tag", "--list", tag);
const remoteTag = push ? git("ls-remote", "--tags", "origin", `refs/tags/${tag}`) : "";
if ((localTag || remoteTag) && !retag) fail(`${tag} already exists (--retag reuses it if npm doesn't have ${version} yet)`);
if (retag) {
    for (const p of kPackages) {
        let published = "";
        try {
            published = execFileSync("npm", ["view", `@web-falcor/${p}@${version}`, "version"], { cwd: repoRoot, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
        } catch {
            // not on the registry
        }
        if (published) fail(`@web-falcor/${p}@${version} is already on npm: release a new version instead`);
    }
}

console.log(`Releasing ${current} -> ${version}`);
if (version !== current) run("npm", "version", version, "--workspaces", "--include-workspace-root", "--no-git-tag-version", "--allow-same-version");
// Only the version files (npm version also chmods workspace bin scripts, which isn't part of a release).
const versionFiles = ["package.json", "package-lock.json", ...git("ls-files", "packages/*/package.json").split("\n")].filter(Boolean);
run("git", "add", "--", ...versionFiles);
if (git("diff", "--cached", "--name-only")) run("git", "commit", "-m", `chore: release ${version}`);
run("git", "tag", ...(retag ? ["-f"] : []), tag);

if (push) {
    run("git", "push", "origin", "HEAD");
    run("git", "push", ...(retag ? ["-f"] : []), "origin", tag);
    console.log(`\nPushed ${tag}: the Release workflow publishes it (https://github.com/windingwind/web-falcor/actions).`);
} else {
    console.log(`\nTagged ${tag} locally. Publish with: git push origin HEAD && git push ${retag ? "-f " : ""}origin ${tag}`);
}
