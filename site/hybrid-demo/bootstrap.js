// Load the local module graph and surface startup failures in the interface.
(() => {
  let completed = false;
  function showFailure(error) {
    if (completed) return;
    completed = true;
    console.error('Demo startup failed:', error);
    const status = document.getElementById('service-status');
    if (status) { status.textContent = 'Demo could not start'; status.className = 'offline'; }
    const note = document.getElementById('live-service-note');
    if (note) note.textContent = 'The demo could not finish loading. Reload the page to try again.';
    const message = document.getElementById('viewer-message');
    if (message) {
      message.hidden = false; message.classList.add('error');
      const ring = message.querySelector('.loading-ring'); if (ring) ring.hidden = true;
    }
    const text = document.getElementById('viewer-message-text');
    if (text) text.textContent = 'The demo could not start. Reload the page to try again.';
    const reload = document.getElementById('retry-scene');
    if (reload) { reload.hidden = false; reload.textContent = 'Reload page'; reload.addEventListener('click', () => window.location.reload(), { once: true }); }
    const select = document.getElementById('scene-select');
    if (select) { select.replaceChildren(new Option('Scenes unavailable', '')); select.disabled = true; }
  }
  const watchdog = setTimeout(() => showFailure(new Error('Startup exceeded 25 seconds')), 25000);
  import('./demo.js?v=5235cde58e15').then(() => { completed = true; clearTimeout(watchdog); }).catch((error) => { clearTimeout(watchdog); showFailure(error); });
})();
