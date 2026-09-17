"""Exercise the static gateway's scoped connection address and logout handling."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("secure_context", [True, False])
def test_native_builder_preserves_scoped_url_and_clears_it_on_logout(secure_context):
    node = shutil.which("node")
    if not node:
        pytest.skip(
            "Node is used only to test browser JavaScript, not by the device app"
        )
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const [source, html, secure] = process.argv.slice(1);
const elements = new Map();
for (const match of fs.readFileSync(html, 'utf8').matchAll(/id="([^"]+)"/g)) {
  elements.set(match[1], {
    hidden: true, value: '', textContent: '', handlers: {}, contentWindow: {},
    addEventListener(type, fn) { this.handlers[type] = fn; },
    removeAttribute(name) { delete this[name]; },
    focus() { this.focused = true; }, select() { this.selected = true; },
  });
}
const get = id => {
  assert(elements.has(id), 'Missing actual HTML element: ' + id);
  return elements.get(id);
};
const listeners = {};
const copied = [];
const origin = 'http://192.0.2.1:9001';
const runtime = origin + '/ui/runtime/temporary-scoped-grant';
const sandbox = {
  URL, console,
  location: {origin, pathname: '/build/edit/test-workflow'},
  history: {pushState() {}},
  document: {getElementById: get, execCommand: () => false},
  navigator: {clipboard: {writeText: async value => copied.push(value)}},
  window: {isSecureContext: secure === 'true', addEventListener: (name, fn) => listeners[name] = fn},
  EdgeUI: {
    refreshSession: async () => ({}),
    request: async path => {
      assert.equal(path, '/ui/builder-session');
      return {origin: 'https://app.roboflow.com', runtime_path: '/ui/runtime/temporary-scoped-grant', csrf: 'temporary-scoped-grant'};
    },
    logout: async () => {},
  },
};
vm.runInNewContext(fs.readFileSync(source, 'utf8'), sandbox);
(async () => {
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(get('builder-runtime-url').value, runtime);
  assert.equal(get('builder-connection').hidden, false);
  const editor = new URL(get('builder-open').href);
  assert.equal(editor.pathname, '/workflows/local/test-workflow');
  assert.equal(editor.searchParams.get('serverUrl'), runtime);
  assert.equal(editor.searchParams.get('csrf'), 'temporary-scoped-grant');
  if (secure === 'true') assert.equal(get('workflow-iframe').src, editor.href);
  else {
    assert.equal(get('workflow-iframe').src, undefined);
    assert.equal(get('builder-launch').hidden, false);
  }
  await get('builder-copy-runtime').handlers.click();
  assert.deepEqual(copied, [runtime]);
  assert.match(get('builder-copy-status').textContent, /已复制/);
  sandbox.navigator.clipboard = undefined;
  await get('builder-copy-runtime').handlers.click();
  assert(get('builder-runtime-url').selected);
  assert.match(get('builder-copy-status').textContent, /Ctrl\+C/);
  await get('builder-logout').handlers.click();
  assert.equal(get('builder-runtime-url').value, '');
  assert.equal(get('builder-connection').hidden, true);
  assert.equal(get('builder-open').href, undefined);
  assert.equal(get('workflow-iframe').src, undefined);
  assert.equal(get('builder-copy-status').textContent, '');
  await get('builder-copy-runtime').handlers.click();
  assert.equal(copied.length, 1);
  // Expiration follows the same cleanup, including a fresh connection URL.
  await get('builder-retry').handlers.click();
  assert.equal(get('builder-runtime-url').value, runtime);
  listeners['edge-session-expired']();
  assert.equal(get('builder-runtime-url').value, '');
  assert.equal(get('builder-open').href, undefined);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [
            node,
            "-e",
            script,
            str(ROOT / "inference/edge/static/builder.js"),
            str(ROOT / "inference/edge/static/build.html"),
            str(secure_context).lower(),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
