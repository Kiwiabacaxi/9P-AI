// Web Worker that runs the Go WASM in a separate thread.
// Communication: main thread sends commands via postMessage,
// worker responds with results/progress via postMessage back.

importScripts('wasm_exec.js');

let ready = false;

const go = new Go();
WebAssembly.instantiateStreaming(fetch('main.wasm'), go.importObject).then(result => {
  go.run(result.instance);
  // The Go main() fires a 'wasmReady' event on `document`,
  // but there's no DOM in a Worker. We patched the Go bridge
  // to also set a global flag. Let's just check if functions exist.
  ready = true;
  postMessage({ type: 'ready' });
});

// Dispatch commands from main thread
onmessage = function(e) {
  const { id, cmd, args } = e.data;
  if (!ready) {
    postMessage({ id, error: 'WASM not ready' });
    return;
  }
  try {
    switch (cmd) {
      // ── Sync functions: call and return result ──
      case 'status':           reply(id, wasmStatus()); break;
      case 'hebbPortas':       reply(id, wasmHebbPortas()); break;
      case 'hebbTrain':        reply(id, wasmHebbTrain(args[0])); break;
      case 'percPortasLista':  reply(id, wasmPercPortasLista()); break;
      case 'percPortasTrain':  reply(id, wasmPercPortasTrain(args[0])); break;
      case 'percLetrasTrain':  reply(id, wasmPercLetrasTrain()); break;
      case 'percLetrasDataset':reply(id, wasmPercLetrasDataset()); break;
      case 'mlpResult':        reply(id, wasmMlpResult()); break;
      case 'letrasResult':     reply(id, wasmLetrasResult()); break;
      case 'letrasClassify':   reply(id, wasmLetrasClassify(args[0])); break;
      case 'letrasDataset':    reply(id, wasmLetrasDataset()); break;
      case 'madResult':        reply(id, wasmMadResult()); break;
      case 'madClassify':      reply(id, wasmMadClassify(args[0])); break;
      case 'madDataset':       reply(id, wasmMadDataset()); break;
      case 'imgregTarget':     reply(id, wasmImgregTarget(args[0])); break;
      case 'imgregReset':      reply(id, wasmImgregReset()); break;
      case 'igorReset':        reply(id, wasmIgorReset()); break;
      case 'imbReset':         reply(id, wasmImbReset()); break;

      // ── Promise-based (heavy sync): run in Go goroutine, reply when done ──
      case 'mlpTrain': {
        const p = wasmMlpTrain();
        if (p && typeof p.then === 'function') {
          p.then(r => reply(id, r));
        } else {
          reply(id, p);
        }
        break;
      }

      // ── Streaming functions: call with callback, forward progress ──
      case 'letrasTrain':
        wasmLetrasTrain((stepJSON) => {
          postMessage({ id, type: 'step', data: stepJSON });
        });
        break;
      case 'madTrain':
        wasmMadTrain((stepJSON) => {
          postMessage({ id, type: 'step', data: stepJSON });
        });
        break;
      case 'imgregTrain':
        wasmImgregTrain(args[0], (stepJSON) => {
          postMessage({ id, type: 'step', data: stepJSON });
        });
        break;
      case 'igorTrain':
        wasmIgorTrain(args[0], (stepJSON) => {
          postMessage({ id, type: 'step', data: stepJSON });
        });
        break;
      case 'imbTrain':
        wasmImbTrain(args[0], (stepJSON) => {
          postMessage({ id, type: 'step', data: stepJSON });
        });
        break;

      default:
        postMessage({ id, error: 'unknown command: ' + cmd });
    }
  } catch (err) {
    postMessage({ id, error: err.message || String(err) });
  }
};

function reply(id, data) {
  postMessage({ id, type: 'result', data });
}
