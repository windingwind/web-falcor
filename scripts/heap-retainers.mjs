// Prints strong retainer paths to GC roots for objects of a constructor name in a .heapsnapshot
// (tests/gpu/harness: log "#HEAPSNAPSHOT <file>"). Usage: node --max-old-space-size=8000 scripts/heap-retainers.mjs <snapshot> <ClassName>
import fs from "fs";
const snap = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const target = process.argv[3] ?? "Scene";
const m = snap.snapshot.meta, nf = m.node_fields.length, ef = m.edge_fields.length;
const nodeTypes = m.node_types[0], edgeTypes = m.edge_types[0];
const N = snap.nodes, E = snap.edges, S = snap.strings;
const iType = m.node_fields.indexOf("type"), iName = m.node_fields.indexOf("name"), iEC = m.node_fields.indexOf("edge_count"), iSize = m.node_fields.indexOf("self_size");
const eType = m.edge_fields.indexOf("type"), eName = m.edge_fields.indexOf("name_or_index"), eTo = m.edge_fields.indexOf("to_node");
const count = N.length / nf;
const firstEdge = new Uint32Array(count + 1);
for (let i = 0, e = 0; i < count; i++) { firstEdge[i] = e; e += N[i * nf + iEC] * ef; firstEdge[i + 1] = e; }
const rev = Array.from({ length: count }, () => []);
for (let i = 0; i < count; i++) for (let e = firstEdge[i]; e < firstEdge[i + 1]; e += ef) {
    const t = edgeTypes[E[e + eType]]; if (t === "weak" || t === "shortcut") continue;
    rev[E[e + eTo] / nf].push([i, e]);
}
const name = (i) => `${nodeTypes[N[i * nf + iType]]}:${S[N[i * nf + iName]]}`.slice(0, 80);
const edgeName = (e) => { const t = edgeTypes[E[e + eType]]; return t === "element" || t === "hidden" ? `[${E[e + eName]}]` : S[E[e + eName]]; };
const targets = []; for (let i = 0; i < count; i++) if (nodeTypes[N[i * nf + iType]] === "object" && S[N[i * nf + iName]] === target) targets.push(i);
console.log(`${targets.length} ${target} objects`);
for (const t of targets.slice(0, 3)) {
    // BFS toward root (node 0).
    const prev = new Map([[t, null]]); const q = [t]; let found = -1;
    while (q.length) { const x = q.shift(); if (x === 0) { found = x; break; } for (const [p, e] of rev[x]) if (!prev.has(p)) { prev.set(p, [x, e]); q.push(p); } }
    if (found < 0) { console.log("no path"); continue; }
    const path = []; let x = 0; while (x !== t) { const [c, e] = prev.get(x); path.push(`${name(x)} --${edgeName(e)}-->`); x = c; }
    console.log(path.join("\n  ") + "\n  " + name(t));
}
