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
  onMessage?: (data: unknown) => void;
  onDone?: (data: unknown) => void;
  onError?: () => void;
}): () => void {
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
    // Only call onError if stream didn't complete successfully
    if (!completed) handlers.onError?.();
  };
  return () => { completed = true; es.close(); };
}
