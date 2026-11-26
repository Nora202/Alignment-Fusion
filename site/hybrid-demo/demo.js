import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const API_BASE = String(window.DEMO_API_BASE || '').replace(/\/+$/, '');
const ASSET_BASE = String(window.DEMO_ASSET_BASE || '').replace(/\/+$/, '');
const TASKS = ['segmentation', 'grounding', 'qa', 'caption'];
const RUN_LABELS = { grounding: 'Locate object', qa: 'Answer question', caption: 'Generate caption' };
const RUNNING_LABELS = { grounding: 'Locating object…', qa: 'Generating answer…', caption: 'Generating caption…' };
const PLACEHOLDERS = {
  grounding: 'The prediction will appear here and highlight an object in the scene.',
  qa: 'The model’s answer will appear here.',
  caption: 'The caption will appear here alongside its highlighted target.',
};
const $ = (id) => document.getElementById(id);
const state = {
  scenes: [], scene: null, task: 'segmentation', selectedId: null,
  mode: 'rgb', ready: false, busy: false, pendingTask: null,
  sceneVersion: 0, sessions: new Map(), checking: false,
};
let renderer, camera, controls, rgbRoot, instancesRoot, liveRoot;
let pointTargets = [], renderFrame = 0, sceneRadius = 1;
let initialCamera = null;
const scene3d = new THREE.Scene();
const loader = new GLTFLoader();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function apiURL(path) { return `${API_BASE}${path}`; }
function assetURL(path) {
  if (/^https?:\/\//i.test(path)) return path;
  if (ASSET_BASE) return `${ASSET_BASE}/${String(path).replace(/^\/+/, '')}`;
  if (String(path).startsWith('/') && API_BASE) return `${API_BASE}${path}`;
  return new URL(path, document.baseURI).href;
}
async function fetchJSON(url, options = {}, timeout = 12000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { data = null; }
    if (!response.ok) {
      let detail = data?.detail;
      if (Array.isArray(detail)) detail = detail.map((item) => item.msg || '').filter(Boolean).join('; ');
      const error = new Error(typeof detail === 'string' ? detail : `The service could not complete this request (${response.status}).`);
      error.status = response.status;
      throw error;
    }
    if (!data) throw new Error('The service returned an unreadable response.');
    return data;
  } finally { clearTimeout(timer); }
}
function currentSession() {
  const id = state.scene?.id;
  if (!state.sessions.has(id)) state.sessions.set(id, { inputs: {}, results: {} });
  return state.sessions.get(id);
}
function objectsFor(task = state.task) {
  return task === 'segmentation' ? (state.scene?.objects || []) : (state.scene?.live_objects || []);
}
function objectById(id, task = state.task) {
  return objectsFor(task).find((item) => Number(item.id) === Number(id)) || null;
}
function objectTitle(object, task = state.task) {
  return `${object.class_name || 'Object'} · ${task === 'segmentation' ? 'instance' : 'proposal'} ${object.id}`;
}
function formatConfidence(value) {
  return value !== null && value !== undefined && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : null;
}
function updateControls() {
  for (const task of ['grounding', 'qa', 'caption']) {
    const input = $(`${task}-text`);
    const button = document.querySelector(`form[data-task="${task}"] .run-button`);
    const text = input.value.trim();
    const valid = task === 'caption' ? Boolean(text || state.selectedId !== null) : Boolean(text);
    button.disabled = !state.ready || state.busy || !state.scene || !valid;
    button.textContent = state.busy && state.pendingTask === task ? RUNNING_LABELS[task] : RUN_LABELS[task];
    button.setAttribute('aria-busy', String(state.busy && state.pendingTask === task));
    input.disabled = state.busy;
  }
  $('scene-select').disabled = state.busy || !state.scenes.length;
  $('instance-select').disabled = state.busy || !state.scene;
  $('clear-selection').disabled = state.busy || state.selectedId === null;
  $('reconnect').disabled = state.busy || state.checking;
  const selected = state.selectedId !== null ? objectById(state.selectedId) : null;
  $('caption-target').textContent = selected
    ? `Selected target: ${objectTitle(selected)}. A text description will locate a new target.`
    : 'Select an instance or enter a target description.';
  $('caption-target').classList.toggle('selected', Boolean(selected));
}
async function checkService() {
  if (state.checking) return;
  state.checking = true;
  $('service-status').textContent = 'Checking service…';
  $('service-status').className = '';
  updateControls();
  try {
    const health = await fetchJSON(apiURL('/api/health'), {}, 12000);
    state.ready = health.status === 'ready' && health.fusion === true;
    $('service-status').textContent = state.ready
      ? `Ready · ${String(health.device || 'CPU').toUpperCase()}`
      : (health.status === 'loading' || health.status === 'starting' ? 'Model is starting…' : 'Live service unavailable');
    $('service-status').className = state.ready ? 'ready' : 'offline';
    $('live-service-note').textContent = state.ready
      ? `${health.model || 'stage2_best'} · Orthogonal Fusion enabled. Requests run on the live service.`
      : 'Saved scene results remain available. Check the connection again when the live service is ready.';
  } catch {
    state.ready = false;
    $('service-status').textContent = 'Live service unavailable';
    $('service-status').className = 'offline';
    $('live-service-note').textContent = 'Saved scene results remain available. Live text inference is not connected at the moment.';
  } finally {
    state.checking = false;
    $('reconnect').hidden = state.ready;
    updateControls();
  }
}
function setTask(task, focus = false) {
  if (!TASKS.includes(task)) return;
  const changedGroup = (state.task === 'segmentation') !== (task === 'segmentation');
  state.task = task;
  if (changedGroup) state.selectedId = null;
  populateInstanceSelect();
  for (const item of TASKS) {
    const active = item === task;
    $(`tab-${item}`).setAttribute('aria-selected', String(active));
    $(`tab-${item}`).tabIndex = active ? 0 : -1;
    $(`panel-${item}`).hidden = !active;
  }
  const tab = $(`tab-${task}`);
  if (focus) {
    tab.focus();
    tab.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  }
  const result = state.scene && currentSession().results[task];
  if (result && result.object_id !== null && result.object_id !== undefined) selectInstance(result.object_id);
  else selectInstance(state.selectedId);
  updateAppearance();
  updateControls();
}
function renderResult(task) {
  const box = $(`${task}-result`);
  const result = currentSession().results[task];
  box.replaceChildren();
  if (!result) {
    const p = document.createElement('p'); p.className = 'result-placeholder'; p.textContent = PLACEHOLDERS[task]; box.append(p); return;
  }
  if (result.loading) {
    const p = document.createElement('p'); p.className = 'result-placeholder'; p.textContent = `${RUNNING_LABELS[task]} The service may queue requests.`; box.append(p); return;
  }
  if (result.error) {
    const p = document.createElement('p'); p.className = 'result-error'; p.textContent = result.error; box.append(p); return;
  }
  const label = document.createElement('span'); label.className = 'result-label'; label.textContent = task === 'grounding' ? 'Located object' : task === 'qa' ? 'Model answer' : 'Generated caption';
  const p = document.createElement('p');
  const object = result.object_id !== null && result.object_id !== undefined ? objectById(result.object_id, task) : null;
  p.textContent = result.text?.trim() || (object ? objectTitle(object, task) : 'The service returned no text.');
  box.append(label, p);
  if (object || Number.isFinite(result.elapsed_ms)) {
    const timing = document.createElement('div'); timing.className = 'result-timing';
    const parts = [];
    if (object) parts.push(objectTitle(object, task));
    if (Number.isFinite(result.elapsed_ms)) parts.push(`${(result.elapsed_ms / 1000).toFixed(2)} s`);
    timing.textContent = parts.join(' · '); box.append(timing);
  }
}
function renderExamples() {
  for (const task of ['grounding', 'qa', 'caption']) {
    const box = $(`${task}-examples`); box.replaceChildren();
    const examples = state.scene?.examples?.[task] || [];
    for (const text of examples.slice(0, 3)) {
      if (typeof text !== 'string') continue;
      const button = document.createElement('button'); button.type = 'button'; button.className = 'example-chip'; button.textContent = text;
      button.addEventListener('click', () => {
        if (state.busy) return;
        $(`${task}-text`).value = text;
        currentSession().inputs[task] = text;
        updateControls();
      });
      box.append(button);
    }
  }
}
async function runTask(task) {
  if (!state.scene || !state.ready || state.busy) return;
  const text = $(`${task}-text`).value.trim();
  if (task !== 'caption' && !text) return;
  if (task === 'caption' && !text && state.selectedId === null) return;
  const sceneId = state.scene.id;
  const session = currentSession();
  session.inputs[task] = text;
  session.results[task] = { loading: true };
  state.busy = true; state.pendingTask = task;
  renderResult(task); updateControls();
  try {
    const result = await fetchJSON(apiURL('/api/infer'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene_id: sceneId, task, text, object_id: state.selectedId }),
    }, 120000);
    session.results[task] = {
      text: typeof result.text === 'string' ? result.text : '',
      object_id: result.object_id ?? null,
      elapsed_ms: typeof result.elapsed_ms === 'number' ? result.elapsed_ms : null,
    };
    if (state.scene.id === sceneId) {
      renderResult(task);
      if (state.task === task && result.object_id !== null && result.object_id !== undefined) selectInstance(result.object_id);
    }
  } catch (error) {
    const message = error.name === 'AbortError'
      ? 'The request took too long. Please try again when the service is less busy.'
      : (error instanceof TypeError ? 'The inference service could not be reached. Check the connection and try again.' : error.message);
    session.results[task] = { error: message };
    if (state.scene.id === sceneId) renderResult(task);
    if (error instanceof TypeError || error.status === 503) {
      state.ready = false;
      $('service-status').textContent = 'Live service unavailable'; $('service-status').className = 'offline'; $('reconnect').hidden = false;
      $('live-service-note').textContent = 'Saved scene results remain available. Reconnect to use live text inference.';
    }
  } finally {
    state.busy = false; state.pendingTask = null; updateControls();
  }
}
function viewerMessage(message, error = false) {
  $('viewer-message-text').textContent = message;
  $('viewer-message').hidden = !message;
  $('viewer-message').classList.toggle('error', error);
  $('viewer-message').querySelector('.loading-ring').hidden = error;
  $('retry-scene').hidden = !error;
}
function requestRender() {
  if (!renderer || renderFrame) return;
  renderFrame = requestAnimationFrame(() => { renderFrame = 0; renderer.render(scene3d, camera); });
}
function disposeRoot(root) {
  if (!root) return;
  root.traverse((object) => {
    object.geometry?.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) if (material) material.dispose();
  });
}
function preparePoints(root) {
  root.traverse((object) => {
    if (!object.isPoints) return;
    // The supplied point-cloud GLBs carry RGB values, so normalize them for the linear renderer.
    const colorAttribute = object.geometry.getAttribute('color');
    if (colorAttribute) {
      const colors = new Float32Array(colorAttribute.count * 3);
      const color = new THREE.Color();
      for (let i = 0; i < colorAttribute.count; i++) {
        color.setRGB(colorAttribute.getX(i), colorAttribute.getY(i), colorAttribute.getZ(i)).convertSRGBToLinear();
        colors.set([color.r, color.g, color.b], i * 3);
      }
      object.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    const previous = object.material;
    object.material = new THREE.PointsMaterial({ size: 0.018, sizeAttenuation: true, vertexColors: Boolean(colorAttribute), transparent: true, opacity: 1, depthWrite: true, toneMapped: false });
    if (Array.isArray(previous)) previous.forEach((m) => m.dispose()); else previous?.dispose();
    object.frustumCulled = false;
  });
}
function instanceId(object) {
  for (let node = object; node; node = node.parent) {
    const id = node.userData?.instance_index;
    if (id !== undefined && Number.isInteger(Number(id))) return Number(id);
  }
  return -1;
}
function updateAppearance() {
  const isSaved = state.task === 'segmentation';
  const hasSelection = state.selectedId !== null;
  document.querySelector('.view-switch').hidden = !isSaved;
  $('viewer-badge').textContent = isSaved ? 'Stage 1 predictions' : 'Precomputed proposal set · live language inference';
  if (!rgbRoot || !instancesRoot) return;
  rgbRoot.visible = !isSaved || state.mode === 'rgb';
  instancesRoot.visible = isSaved && (state.mode === 'instances' || state.selectedId !== null);
  if (liveRoot) liveRoot.visible = !isSaved && state.selectedId !== null;
  const update = (root, live) => root?.traverse((object) => {
    if (!object.isPoints) return;
    const id = instanceId(object);
    const selected = state.selectedId !== null && id === state.selectedId && live !== isSaved;
    object.visible = live ? selected : (isSaved && (state.mode === 'instances' || selected));
    object.renderOrder = selected ? 2 : 0;
    const material = object.material;
    const vertexColors = !selected;
    if (material.vertexColors !== vertexColors) { material.vertexColors = vertexColors; material.needsUpdate = true; }
    material.color.set(selected ? '#df4b27' : '#ffffff');
    material.opacity = selected ? 0.98 : (hasSelection ? (id < 0 ? 0.16 : 0.24) : 1);
    material.depthTest = !selected;
    material.depthWrite = selected || state.selectedId === null;
    material.size = Math.max(sceneRadius * 0.006, 0.008) * (selected ? 1.12 : 1);
  });
  update(instancesRoot, false); update(liveRoot, true);
  rgbRoot.traverse((object) => {
    if (!object.isPoints) return;
    object.material.size = Math.max(sceneRadius * 0.006, 0.008);
    object.material.opacity = hasSelection ? 0.24 : 1;
    object.material.depthWrite = !hasSelection;
  });
  requestRender();
}
function populateInstanceSelect() {
  const saved = state.task === 'segmentation';
  $('instance-select-label').textContent = saved ? 'Selected instance' : 'Selected target proposal';
  const select = $('instance-select');
  select.replaceChildren(new Option(saved ? 'Select an instance in the scene' : 'Select a target proposal', ''));
  for (const object of objectsFor()) select.add(new Option(objectTitle(object), String(object.id)));
  select.value = state.selectedId === null ? '' : String(state.selectedId);
}
function selectInstance(value) {
  const id = value === null || value === '' || value === undefined ? null : Number(value);
  const object = id !== null ? objectById(id) : null;
  state.selectedId = object ? id : null;
  $('instance-select').value = object ? String(id) : '';
  const confidence = object ? formatConfidence(object.confidence) : null;
  $('instance-meta').textContent = object
    ? `${objectTitle(object)}${confidence ? ` · Confidence ${confidence}` : ''}`
    : (state.task === 'segmentation' ? 'You can also choose an instance from this list.' : 'Choose a proposal here or locate a target using text.');
  updateAppearance(); updateControls();
}
function resetCamera() {
  if (!camera || !initialCamera) return;
  camera.position.copy(initialCamera.position); controls.target.copy(initialCamera.target); controls.update(); requestRender();
}
async function loadScene(scene) {
  const version = ++state.sceneVersion;
  viewerMessage('Loading 3D scene…'); $('reset-camera').disabled = true;
  if (!renderer) { viewerMessage('The 3D viewer could not start. Try a browser with WebGL enabled.', true); return; }
  try {
    const rgbPath = scene.assets?.rgb || `assets/scenes/${scene.id}/rgb.glb`;
    const instancePath = scene.assets?.instances || `assets/scenes/${scene.id}/instances.glb`;
    const livePath = scene.assets?.live_instances || `assets/scenes/${scene.id}/live_instances.glb`;
    const results = await Promise.allSettled([loader.loadAsync(assetURL(rgbPath)), loader.loadAsync(assetURL(instancePath)), loader.loadAsync(assetURL(livePath))]);
    if (results.some((result) => result.status === 'rejected')) {
      for (const result of results) if (result.status === 'fulfilled') disposeRoot(result.value.scene);
      throw new Error('One or more 3D scene files could not be loaded.');
    }
    const [rgb, instances, live] = results.map((result) => result.value);
    if (version !== state.sceneVersion) { disposeRoot(rgb.scene); disposeRoot(instances.scene); disposeRoot(live.scene); return; }
    if (rgbRoot) scene3d.remove(rgbRoot); if (instancesRoot) scene3d.remove(instancesRoot); if (liveRoot) scene3d.remove(liveRoot);
    disposeRoot(rgbRoot); disposeRoot(instancesRoot); disposeRoot(liveRoot);
    rgbRoot = rgb.scene; instancesRoot = instances.scene; liveRoot = live.scene;
    preparePoints(rgbRoot); preparePoints(instancesRoot); preparePoints(liveRoot);
    scene3d.add(rgbRoot, instancesRoot, liveRoot); scene3d.updateMatrixWorld(true);
    pointTargets = [];
    for (const [root, group] of [[instancesRoot, 'segmentation'], [liveRoot, 'live']]) root.traverse((object) => { if (object.isPoints && instanceId(object) >= 0) { object.userData.demoGroup = group; pointTargets.push(object); } });
    const bounds = new THREE.Box3().setFromObject(rgbRoot);
    if (bounds.isEmpty()) throw new Error('The scene does not contain displayable geometry.');
    const size = bounds.getSize(new THREE.Vector3()); const center = bounds.getCenter(new THREE.Vector3());
    sceneRadius = Math.max(size.length() / 2, 0.1);
    const distance = sceneRadius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * Math.max(1, 1 / camera.aspect) * 1.06;
    const direction = new THREE.Vector3(1.0, 1.1, 1.25).normalize();
    initialCamera = { position: center.clone().addScaledVector(direction, distance), target: center.clone() };
    camera.near = sceneRadius / 2000; camera.far = sceneRadius * 100; camera.updateProjectionMatrix();
    controls.minDistance = sceneRadius * 0.12; controls.maxDistance = sceneRadius * 12;
    raycaster.params.Points.threshold = sceneRadius * 0.013;
    resetCamera(); updateAppearance(); viewerMessage(''); $('reset-camera').disabled = false;
  } catch (error) {
    if (version !== state.sceneVersion) return;
    viewerMessage('The 3D scene could not be loaded. Check your connection and reload this scene.', true);
    console.warn('Scene load failed:', error);
  }
}
function chooseScene(id) {
  const scene = state.scenes.find((item) => item.id === id);
  if (!scene) return;
  state.scene = scene; state.selectedId = null;
  $('scene-select').value = scene.id;
  $('saved-scene-name').textContent = scene.name || scene.id;
  $('instance-count').textContent = String(scene.objects.length);
  populateInstanceSelect();
  $('instance-meta').textContent = 'You can also choose an instance from this list.';
  const session = currentSession();
  for (const task of ['grounding', 'qa', 'caption']) { $(`${task}-text`).value = session.inputs[task] || ''; renderResult(task); }
  renderExamples(); updateControls(); updateAppearance(); loadScene(scene);
}
async function loadScenes() {
  try {
    let data;
    try { data = await fetchJSON(apiURL('/api/scenes'), {}, 5000); }
    catch { data = await fetchJSON(assetURL('assets/scenes.json'), {}, 12000); }
    const scenes = Array.isArray(data) ? data : data.scenes;
    if (!Array.isArray(scenes) || !scenes.length) throw new Error('No scene manifest is available.');
    state.scenes = scenes.filter((scene) => typeof scene.id === 'string').map((scene) => ({ ...scene, objects: Array.isArray(scene.objects) ? scene.objects : [], live_objects: Array.isArray(scene.live_objects) ? scene.live_objects : [] }));
    if (!state.scenes.length) throw new Error('The scene manifest is empty.');
    const select = $('scene-select'); select.replaceChildren();
    for (const scene of state.scenes) {
      const name = String(scene.name || '').trim();
      const label = !name ? scene.id : (name.includes(scene.id) ? name : `${name} · ${scene.id}`);
      select.add(new Option(label, scene.id));
    }
    chooseScene(state.scenes[0].id);
  } catch (error) {
    $('scene-select').replaceChildren(new Option('Scenes unavailable', ''));
    $('saved-scene-name').textContent = 'Unavailable'; $('instance-count').textContent = '—';
    viewerMessage('The scene list could not be loaded. Check your connection and reload the page.', true);
    console.warn('Scene manifest unavailable:', error);
  }
}
function initViewer() {
  const host = $('viewer');
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0xf4f7fc, 1);
    renderer.domElement.setAttribute('aria-label', 'Interactive 3D scene. Drag to rotate, scroll to zoom, and use arrow keys to pan. Use the instance list to select an object with the keyboard.'); renderer.domElement.setAttribute('role', 'img'); renderer.domElement.tabIndex = 0;
    host.prepend(renderer.domElement);
    camera = new THREE.PerspectiveCamera(42, 1, 0.001, 1000);
    controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = false; controls.screenSpacePanning = true; controls.maxPolarAngle = Math.PI * 0.94;
    controls.listenToKeyEvents(renderer.domElement);
    controls.addEventListener('change', requestRender);
    const resize = () => {
      const width = Math.max(host.clientWidth, 1); const height = Math.max(host.clientHeight, 1);
      renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix(); requestRender();
    };
    new ResizeObserver(resize).observe(host); resize();
    let down = null;
    renderer.domElement.addEventListener('pointerdown', (event) => { if (event.button === 0) down = { x: event.clientX, y: event.clientY }; });
    renderer.domElement.addEventListener('pointercancel', () => { down = null; });
    renderer.domElement.addEventListener('pointerup', (event) => {
      if (!down || event.button !== 0) return;
      const distance = Math.hypot(event.clientX - down.x, event.clientY - down.y); down = null;
      if (distance > 5 || state.busy || !instancesRoot) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
      scene3d.updateMatrixWorld(true); raycaster.setFromCamera(pointer, camera);
      // Raycast the saved instance geometry even when original RGB is displayed.
      const group = state.task === 'segmentation' ? 'segmentation' : 'live';
      const intersections = raycaster.intersectObjects(pointTargets.filter((object) => object.userData.demoGroup === group), false);
      if (intersections.length) selectInstance(instanceId(intersections[0].object));
    });
  } catch (error) { renderer = null; console.warn('WebGL initialization failed:', error); }
}
for (const tab of document.querySelectorAll('[role="tab"]')) {
  tab.addEventListener('click', () => setTask(tab.dataset.task));
  tab.addEventListener('keydown', (event) => {
    const index = TASKS.indexOf(tab.dataset.task);
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % TASKS.length;
    else if (event.key === 'ArrowLeft') next = (index + TASKS.length - 1) % TASKS.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = TASKS.length - 1;
    else return;
    event.preventDefault(); setTask(TASKS[next], true);
  });
}
for (const task of ['grounding', 'qa', 'caption']) {
  document.querySelector(`form[data-task="${task}"]`).addEventListener('submit', (event) => { event.preventDefault(); runTask(task); });
  $(`${task}-text`).addEventListener('input', () => { if (state.scene) currentSession().inputs[task] = $(`${task}-text`).value; updateControls(); });
}
$('scene-select').addEventListener('change', (event) => chooseScene(event.target.value));
$('instance-select').addEventListener('change', (event) => selectInstance(event.target.value));
$('clear-selection').addEventListener('click', () => selectInstance(null));
$('reset-camera').addEventListener('click', resetCamera);
$('reconnect').addEventListener('click', checkService);
$('retry-scene').addEventListener('click', () => { if (state.scene) loadScene(state.scene); else loadScenes(); });
for (const radio of document.querySelectorAll('input[name="view-mode"]')) radio.addEventListener('change', () => { if (radio.checked) { state.mode = radio.value; updateAppearance(); } });
$('show-instance-colors').addEventListener('click', () => { state.mode = 'instances'; document.querySelector('input[name="view-mode"][value="instances"]').checked = true; updateAppearance(); });
initViewer();
updateControls();
await Promise.allSettled([loadScenes(), checkService()]);
