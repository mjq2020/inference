(() => {
  'use strict';
  const ORIGIN = 'https://app.roboflow.com';
  const frame = document.getElementById('workflow-iframe');
  const status = document.getElementById('builder-status');
  const errorBox = document.getElementById('builder-error');
  const login = document.getElementById('builder-login');
  const launch = document.getElementById('builder-launch');
  const logout = document.getElementById('builder-logout');
  const retry = document.getElementById('builder-retry');
  const connection = document.getElementById('builder-connection');
  const runtimeUrl = document.getElementById('builder-runtime-url');
  const copyStatus = document.getElementById('builder-copy-status');
  let access = null;
  function editorPath() {
    const match = location.pathname.match(/^\/build(?:\/edit\/([\w-]+))?\/?$/);
    if (!match && !['/', '/ui/'].includes(location.pathname)) throw new Error('无效的 Workflow 地址');
    return '/workflows/local' + (match?.[1] ? '/' + encodeURIComponent(match[1]) : '');
  }
  function navigateFrame() {
    const url = new URL(editorPath(), ORIGIN);
    const runtime = location.origin + access.runtime_path;
    runtimeUrl.value = runtime;
    connection.hidden = false;
    url.searchParams.set('serverUrl', runtime);
    url.searchParams.set('csrf', access.csrf);
    document.getElementById('builder-open').href = url.href;
    // HTTPS inside an HTTP LAN parent is not a secure context. The official
    // top-level HTTPS page can request Local Network Access normally.
    if (window.isSecureContext) {
      launch.hidden = true; frame.hidden = false; frame.src = url.href;
    } else {
      frame.hidden = true; frame.removeAttribute('src'); launch.hidden = false;
    }
  }
  function signedOut() {
    access = null; frame.removeAttribute('src'); frame.hidden = true;
    document.getElementById('builder-open').removeAttribute('href');
    connection.hidden = true; runtimeUrl.value = ''; copyStatus.textContent = '';
    launch.hidden = true; login.hidden = false; logout.hidden = true;
    status.textContent = '请登录设备应用';
  }
  async function connect() {
    errorBox.hidden = true; retry.hidden = true; status.textContent = '正在连接设备…';
    try {
      await EdgeUI.refreshSession();
      access = await EdgeUI.request('/ui/builder-session', {});
      if (access.origin !== ORIGIN || !/^\/ui\/runtime\/[\w-]+$/.test(access.runtime_path) || typeof access.csrf !== 'string' || !access.csrf) throw new Error('画布连接信息无效');
      login.hidden = true; logout.hidden = false; retry.hidden = false;
      navigateFrame(); status.textContent = '设备已登录 · 请在画布中设置运行地址';
    } catch (error) {
      if (error.status === 401) signedOut();
      else { status.textContent = '连接未完成'; errorBox.hidden = false; errorBox.textContent = error.message; retry.hidden = false; }
    }
  }
  document.getElementById('builder-login-form').addEventListener('submit', async event => {
    event.preventDefault(); errorBox.hidden = true;
    const input = document.getElementById('builder-password');
    try { await EdgeUI.login(input.value); input.value = ''; await connect(); }
    catch (error) { errorBox.hidden = false; errorBox.textContent = error.status === 401 ? '访问口令不正确，请检查应用配置。' : error.message; }
  });
  document.getElementById('builder-show-password').addEventListener('click', event => {
    const input = document.getElementById('builder-password');
    input.type = input.type === 'password' ? 'text' : 'password';
    event.currentTarget.textContent = input.type === 'password' ? '显示' : '隐藏';
  });
  document.getElementById('builder-copy-runtime').addEventListener('click', async () => {
    if (!access || !runtimeUrl.value) return;
    const value = runtimeUrl.value;
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(value);
      if (access && runtimeUrl.value === value) copyStatus.textContent = '已复制。在运行位置中选择 Other，粘贴后点击 Connect。';
    } catch (_) {
      if (!access || runtimeUrl.value !== value) return;
      // HTTP device pages may not expose the Clipboard API. Keep the address
      // selectable and give a useful manual fallback when copying is blocked.
      runtimeUrl.focus(); runtimeUrl.select();
      let copied = false;
      try { copied = document.execCommand('copy'); } catch (_) {}
      copyStatus.textContent = copied ? '已复制。在运行位置中选择 Other，粘贴后点击 Connect。' : '地址已选中，请按 Ctrl+C（Mac：⌘C）复制。';
    }
  });
  logout.addEventListener('click', async () => { await EdgeUI.logout(); signedOut(); });
  window.addEventListener('message', event => {
    if (event.origin !== ORIGIN || event.source !== frame.contentWindow || !access) return;
    const message = event.data;
    if (!message || typeof message !== 'object' || Array.isArray(message)) return;
    if (message.type === 'setTitle' && typeof message.title === 'string') document.title = message.title.slice(0, 100) + ' · Workflows';
    if (message.type === 'navigate' && typeof message.path === 'string') {
      const path = message.path.replace(/^\//, '');
      if (path !== '' && !/^edit\/[\w-]+$/.test(path)) return;
      history.pushState({}, '', path ? '/build/' + path : '/build');
    }
  });
  window.addEventListener('popstate', () => { if (access) navigateFrame(); });
  window.addEventListener('edge-session-expired', signedOut);
  retry.addEventListener('click', connect);
  connect();
})();
