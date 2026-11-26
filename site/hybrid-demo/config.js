// Public Gradio Space. Set this to '' to use the REST service below instead.
// Never place access tokens or model credentials in frontend configuration.
window.DEMO_GRADIO_SPACE = window.DEMO_GRADIO_SPACE ?? 'NoraQiu/Alignment-Fusion';
// Explicit device contract: cpu or zerogpu. Never falls back between devices.
window.DEMO_INFERENCE_BACKEND = window.DEMO_INFERENCE_BACKEND ?? 'cpu';
window.DEMO_API_BASE = window.DEMO_API_BASE || '';
window.DEMO_ASSET_BASE = new URL('.', document.currentScript.src).href.replace(/\/$/, '');
