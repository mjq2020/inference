/* Same-origin session authentication. Credentials are never persisted. */
(() => {
  'use strict';
  let session = null;
  const decode = async response => {
    const text = await response.text();
    let value;
    try { value = text ? JSON.parse(text) : {}; } catch (_) { value = {message: '服务器返回了非 JSON 响应'}; }
    if (!response.ok) {
      const detail = value.message || value.detail || value.error || `HTTP ${response.status}`;
      const error = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      error.status = response.status;
      if (response.status === 401) { session = null; window.dispatchEvent(new Event('edge-session-expired')); }
      throw error;
    }
    return value;
  };
  async function request(path, body, method) {
    if (!path.startsWith('/') || path.startsWith('//')) throw new Error('只允许访问本设备接口');
    const headers = {Accept: 'application/json'};
    if (session?.csrf) headers['X-CSRF'] = session.csrf;
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    return decode(await fetch(path, {method: method || (body === undefined ? 'GET' : 'POST'), headers,
      credentials: 'same-origin', redirect: 'error', cache: 'no-store',
      body: body === undefined ? undefined : JSON.stringify(body)}));
  }
  async function refreshSession() {
    session = await request('/ui/session');
    if (!session.authenticated || !session.csrf) throw new Error('请先登录应用');
    return session;
  }
  async function login(token) {
    session = await request('/ui/login', {token});
    if (!session.authenticated || !session.csrf) throw new Error('登录未成功');
    return session;
  }
  async function logout() { await request('/ui/logout', {}); session = null; }
  function toast(message, error = false) {
    const node = document.getElementById('toast');
    if (!node) return;
    node.textContent = String(message).slice(0, 2000); node.classList.toggle('is-error', error); node.hidden = false;
    clearTimeout(toast.timer); toast.timer = setTimeout(() => { node.hidden = true; }, error ? 9000 : 4500);
  }
  window.EdgeUI = {request, refreshSession, login, logout, toast, get session() {return session;}};
})();
