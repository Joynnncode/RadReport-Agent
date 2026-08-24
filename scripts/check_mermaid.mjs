// Validate every ```mermaid block in the repo's Markdown, the way GitHub does.
//
// Why this exists: GitHub renders Mermaid server-side and shows a raw parser
// error in place of the diagram when it fails. The README architecture diagram
// shipped broken because a label contained a literal "." -- Mermaid's dotted
// link syntax is `-. text .->`, so ".npz" inside the label collided with the
// ".->" terminator. Nothing in a Python test suite could catch that, and it was
// the first thing on the page.
//
//   npm install mermaid@11 jsdom
//   node scripts/check_mermaid.mjs README.md docs/*.md
//
// Exits non-zero if any block fails to parse.

import { JSDOM } from 'jsdom';
import fs from 'fs';

const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
Object.defineProperty(globalThis, 'navigator',
  { value: dom.window.navigator, configurable: true });
globalThis.Element = dom.window.Element;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.SVGElement = dom.window.SVGElement;

const mermaid = (await import('mermaid')).default;
mermaid.initialize({ startOnLoad: false });

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: node scripts/check_mermaid.mjs <file.md> [...]');
  process.exit(2);
}

let blocks = 0;
let failures = 0;

for (const file of files) {
  if (!fs.existsSync(file)) continue;
  const md = fs.readFileSync(file, 'utf8');
  const found = [...md.matchAll(/```mermaid\n([\s\S]*?)```/g)].map(m => m[1]);
  for (const [i, code] of found.entries()) {
    blocks++;
    try {
      await mermaid.parse(code);
      console.log(`ok    ${file} block ${i + 1}`);
    } catch (e) {
      failures++;
      console.log(`FAIL  ${file} block ${i + 1}`);
      console.log('      ' + String(e.message || e).split('\n').slice(0, 5).join('\n      '));
    }
  }
}

console.log(`\n${blocks} block(s) checked, ${failures} failing`);
process.exit(failures ? 1 : 0);
