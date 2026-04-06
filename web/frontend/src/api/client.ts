const API = '/api';

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(API + path);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(API + path, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

export function apiSSE(path: string, handlers: {
  onMessage?: (data: unknown, event?: string) => void;
  onDone?: (data: unknown) => void;
  onError?: () => void;
  body?: unknown; // if provided, uses POST + fetch instead of EventSource GET
}): () => void {
  // If body is provided, use fetch-based SSE (POST)
  if (handlers.body != null) {
    const ctrl = new AbortController();
    (async () => {
      try {
        const res = await fetch(API + path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(handlers.body),
          signal: ctrl.signal,
        });
        if (!res.ok || !res.body) { handlers.onError?.(); return; }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          // Parse SSE events from buffer
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          let currentEvent = '';
          let currentData = '';
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              currentData = line.slice(6);
            } else if (line === '' && currentData) {
              const parsed = JSON.parse(currentData);
              if (currentEvent === 'done') {
                handlers.onDone?.(parsed);
              } else {
                handlers.onMessage?.(parsed, currentEvent);
              }
              currentEvent = '';
              currentData = '';
            }
          }
        }
      } catch (e) {
        if ((e as Error).name !== 'AbortError') handlers.onError?.();
      }
    })();
    return () => ctrl.abort();
  }

  // Default: GET-based EventSource
  const es = new EventSource(API + path);
  let completed = false;

  if (handlers.onMessage) {
    es.onmessage = (e) => handlers.onMessage!(JSON.parse(e.data));
  }
  if (handlers.onDone) {
    es.addEventListener('done', (e: MessageEvent) => {
      completed = true;
      handlers.onDone!(JSON.parse(e.data));
      es.close();
    });
  }
  es.onerror = () => {
    es.close();
    if (!completed) handlers.onError?.();
  };
  return () => { completed = true; es.close(); };
}
