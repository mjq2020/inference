// Compatibility bookmark: lifecycle and results now belong to the app card.
(() => {
  const target = new URL(location.href);
  target.protocol = 'https:';
  target.port = '';
  target.pathname = '/app-center';
  target.search = '?workflowResults=inference-rv1126b';
  target.hash = '';
  location.replace(target.href);
})();
