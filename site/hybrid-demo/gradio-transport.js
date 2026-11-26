/**
 * Public Gradio API transport; importing this file performs no network request.
 *
 * const transport = createGradioTransport('owner/space', { backend: 'cpu' });
 * const options = { signal: AbortSignal.timeout(120000), onStatus: showStatus };
 * const health = await transport.health(options); // Trust health.status, not queue status.
 * const result = await transport.infer({ scene_id, task, text, object_id }, options);
 *
 * CPU: /health; /infer takes positional inputs
 * [scene_id, task, text, object_id] -> one gr.JSON dictionary. -1 means no target.
 * Explicit zerogpu mode uses /health_gpu and /infer_gpu with the same schema.
 * Each call owns its session, so cancelling health cannot reset an inference.
 * The caller owns timeout/retry policy. Cancelling is best effort: a running task
 * that has started may finish on the server after the caller stops waiting.
 *
 * Client bundle is the official npm artifact, pinned and loaded on first use.
 * The vendored browser bundle retains its original license file and provenance.
 * https://www.gradio.app/docs/js/js-client
 * https://www.gradio.app/guides/getting-started-with-the-js-client
 * https://huggingface.co/docs/hub/en/spaces-zerogpu
 * ZeroGPU browser calls use anonymous GPU quota. No owner token belongs here.
 */
export const GRADIO_CLIENT_VERSION = '2.8.0';
export const GRADIO_CLIENT_URL = new URL(`./vendor/gradio-client-${GRADIO_CLIENT_VERSION}/index.min.js`, import.meta.url).href;
const EXPECTED_CHECKPOINT_SHA256 = 'a374b6bd8cc04f13fbbf7c1a7e62d02ea57598750184ae7d1cddfa9c687f0c7d';

let clientModule;

export class GradioTransportError extends Error {
  constructor(message, code, detail = '', cause) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'GradioTransportError';
    this.code = code;
    this.detail = detail;
  }
}

function backendSpec(backend) {
  if (backend === 'cpu') return { device: 'cpu', label: 'CPU', gpuRequired: false, health: '/health', infer: '/infer' };
  if (backend === 'zerogpu') return { device: 'cuda', label: 'ZeroGPU', gpuRequired: true, health: '/health_gpu', infer: '/infer_gpu' };
  throw new TypeError('Inference backend must be explicitly configured as cpu or zerogpu.');
}

export function validateGradioHealth(result, backend = 'zerogpu') {
  const spec = backendSpec(backend);
  if (result?.status !== 'ready' || result.fusion !== true || result.device !== spec.device
    || result.backend !== backend || result.gpu_execution_required !== spec.gpuRequired || result.model !== 'stage2_best'
    || result.strict_online_load !== true || result.fusion_parameters !== 16
    || result.checkpoint_sha256 !== EXPECTED_CHECKPOINT_SHA256
    || (backend === 'cpu' && result.cuda_initialized !== false)) {
    throw new GradioTransportError(`The service has not verified the required ${spec.label} model.`, 'invalid_response');
  }
  return result;
}

export function validateGradioInference(result, backend = 'zerogpu') {
  const spec = backendSpec(backend);
  if (result?.device !== spec.device || result.fusion !== true || result.model !== 'stage2_best'
    || typeof result.text !== 'string' || !Number.isFinite(result.elapsed_ms) || result.elapsed_ms < 0
    || !(result.object_id === null || (Number.isInteger(result.object_id) && result.object_id >= -1))) {
    throw new GradioTransportError(`The service did not return a verified ${spec.label} inference result.`, 'invalid_response');
  }
  return { ...result, object_id: result.object_id === -1 ? null : result.object_id };
}

// Preserve the explicit ZeroGPU validation API for existing integrations.
export function validateZeroGPUHealth(result) { return validateGradioHealth(result, 'zerogpu'); }
export function validateZeroGPUInference(result) { return validateGradioInference(result, 'zerogpu'); }

function loadClient() {
  if (!clientModule) {
    clientModule = import(GRADIO_CLIENT_URL).catch((cause) => {
      clientModule = undefined;
      throw new GradioTransportError('The demo client could not load. Check your connection and retry.', 'client_load', messageOf(cause), cause);
    });
  }
  return clientModule;
}

function messageOf(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(messageOf).filter(Boolean).join('; ');
  return value?.message || value?.msg || value?.detail || '';
}

function abortError(signal) {
  if (signal.reason?.name === 'TimeoutError' || signal.reason?.name === 'AbortError') return signal.reason;
  const error = new Error(messageOf(signal.reason) || 'Request cancelled.');
  error.name = 'AbortError';
  return error;
}

function throwIfAborted(signal) {
  if (signal?.aborted) throw abortError(signal);
}

function abortable(promise, signal) {
  if (!signal) return promise;
  return new Promise((resolve, reject) => {
    const cleanup = () => signal.removeEventListener('abort', onAbort);
    const onAbort = () => { cleanup(); reject(abortError(signal)); };
    signal.addEventListener('abort', onAbort, { once: true });
    Promise.resolve(promise).then(
      (value) => { cleanup(); resolve(value); },
      (error) => { cleanup(); reject(error); },
    );
    if (signal.aborted) onAbort();
  });
}

function notify(callback, status) {
  if (typeof callback !== 'function') return;
  try { callback(status); }
  catch (error) { console.warn('Demo status callback failed:', error); }
}

function queueStatus(event) {
  let message = messageOf(event.message);
  if (!message) {
    if (event.stage === 'pending') {
      message = event.queue ? 'Queued for inference' : 'Preparing request';
      if (Number.isInteger(event.position) && event.position >= 0) message += ` · position ${event.position + 1}`;
      if (Number.isFinite(event.eta) && event.eta > 0) message += ` · estimated ${Math.ceil(event.eta)}s`;
      message += '.';
    } else if (event.stage === 'generating' || event.stage === 'streaming') message = 'Running inference…';
    else if (event.stage === 'complete') message = 'Request completed.';
    else if (event.stage === 'error') message = 'The demo service could not complete this request.';
  }
  return {
    source: 'queue', stage: event.stage, message,
    position: event.position ?? null, size: event.size ?? null, eta: event.eta ?? null,
    raw: event,
  };
}

function readableError(cause, backend) {
  if (cause?.name === 'AbortError' || cause?.name === 'TimeoutError' || cause instanceof GradioTransportError) return cause;
  const detail = String(messageOf(cause) || 'The demo service did not respond.');
  if (/quota|rate.?limit|too many requests|exceeded.*gpu|gpu.*exceeded/i.test(detail)) {
    if (backend === 'cpu') {
      return new GradioTransportError('The demo service reported a usage limit. Please try again later.', 'usage_limit', detail, cause);
    }
    return new GradioTransportError(`Hugging Face usage limit reached. ${detail} Open the Space to check access and quota.`, 'quota', detail, cause);
  }
  if (/\b40[134]\b|unauthori|forbidden|private|authentication|sign.?in|log.?in/i.test(detail)) {
    return new GradioTransportError('The Space could not be accessed. Open it on Hugging Face to check its availability and access settings.', 'access', detail, cause);
  }
  if (/fetch|network|connection|load failed|load.*config|could not resolve/i.test(detail)) {
    return new GradioTransportError(`The demo service could not be reached. ${detail}`, 'connection', detail, cause);
  }
  return new GradioTransportError(detail, 'service', detail, cause);
}

function unwrapDictionary(data) {
  const value = Array.isArray(data) && data.length === 1 ? data[0] : null;
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new GradioTransportError('The demo returned an invalid JSON result.', 'invalid_response');
  }
  return value;
}

function cancelSubmission(job) {
  if (!job) return;
  const cancel = () => Promise.resolve().then(() => job.cancel()).catch(() => {});
  void cancel();
  // Queue join may still be in flight when a caller aborts. Retry cancellation
  // once the SDK knows its event ID, without delaying the caller's timeout.
  if (typeof job.wait_for_id === 'function') {
    void Promise.resolve().then(() => job.wait_for_id()).then((id) => id ? cancel() : undefined).catch(() => {});
  }
}

/** onStatus receives {source, stage, message, raw, position?, size?, eta?}. */
export function createGradioTransport(spaceId, { backend = 'zerogpu' } = {}) {
  if (typeof spaceId !== 'string' || !/^[\w.-]+\/[\w.-]+$/.test(spaceId)) {
    throw new TypeError('A Hugging Face Space ID in owner/space format is required.');
  }
  const spec = backendSpec(backend);
  const spaceURL = `https://huggingface.co/spaces/${spaceId}`;

  async function call(endpoint, inputs, { signal, onStatus } = {}) {
    let client, job, completed = false, active = true;
    const publish = (status) => { if (active) notify(onStatus, status); };
    try {
      throwIfAborted(signal);
      publish({ source: 'transport', stage: 'connecting', message: 'Connecting to the demo service…', raw: null });
      const { Client } = await abortable(loadClient(), signal);
      throwIfAborted(signal);
      let rejectSpace;
      const spaceFailure = new Promise((_, reject) => { rejectSpace = reject; });
      const connection = Client.connect(spaceId, {
        events: ['data', 'status'],
        status_callback: (status) => {
          publish({ source: 'space', stage: status.status, message: messageOf(status.message) || 'Starting the demo service…', raw: status });
          if (status.load_status === 'error') rejectSpace(new Error(messageOf(status.message) || `Space unavailable (${status.detail || status.status}).`));
        },
      });
      // Client.connect has no AbortSignal option. If it finishes after timeout,
      // close its session; the timed-out call must never submit an inference.
      connection.then((connected) => { if (!active) connected.close(); }, () => {});
      client = await abortable(Promise.race([connection, spaceFailure]), signal);
      throwIfAborted(signal);
      job = client.submit(endpoint, inputs);
      let data, hasData = false, terminalComplete = false;
      while (true) {
        const event = await abortable(job.next(), signal);
        if (event.done) break;
        if (event.value.type === 'status') {
          publish(queueStatus(event.value));
          if (event.value.stage === 'error' || event.value.success === false) {
            throw new Error(messageOf(event.value.message) || 'The demo service could not complete this request.');
          }
          if (event.value.stage === 'complete') terminalComplete = true;
        } else if (event.value.type === 'data') {
          data = event.value.data; hasData = true;
        }
        // The SDK's non-queued path emits data + complete without closing its
        // iterator. Complete marks the final payload; do not await another next.
        if (terminalComplete && hasData) break;
      }
      throwIfAborted(signal);
      const result = unwrapDictionary(data);
      completed = true;
      return result;
    } catch (cause) {
      const error = readableError(cause, backend);
      error.spaceURL = spaceURL;
      const message = error.name === 'TimeoutError' ? 'The request timed out. The service may still be starting or busy; please retry later.' : error.message;
      publish({ source: 'transport', stage: 'error', message, raw: error });
      throw error;
    } finally {
      active = false;
      if (job && !completed) cancelSubmission(job);
      if (job) void Promise.resolve().then(() => job.return()).catch(() => {});
      client?.close();
    }
  }

  return Object.freeze({
    spaceId, spaceURL, backend,
    async health(options) { return validateGradioHealth(await call(spec.health, [], options), backend); },
    async infer(payload, options) {
      const { scene_id, task, text = '', object_id = null } = payload || {};
      const target = object_id === null || object_id === undefined ? -1 : object_id;
      if (typeof scene_id !== 'string' || !scene_id.trim() || !['grounding', 'qa', 'caption'].includes(task)
        || typeof text !== 'string' || !Number.isInteger(target) || target < -1
        || (task !== 'caption' && !text.trim()) || (task === 'caption' && !text.trim() && target === -1)) {
        throw new GradioTransportError('Choose a scene and provide text or a valid caption target.', 'invalid_input');
      }
      const result = await call(spec.infer, [scene_id, task, text, target], options);
      return validateGradioInference(result, backend);
    },
  });
}
