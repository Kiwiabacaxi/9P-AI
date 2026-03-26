import { useState, createContext, useContext, useCallback, type ReactNode } from 'react';

interface ToastCtx {
  show: (msg: string, duration?: number) => void;
}

const Ctx = createContext<ToastCtx>({ show: () => {} });

export function useToast() { return useContext(Ctx); }

export function ToastProvider({ children }: { children: ReactNode }) {
  const [msg, setMsg] = useState('');
  const [visible, setVisible] = useState(false);

  const show = useCallback((m: string, dur = 2500) => {
    setMsg(m);
    setVisible(true);
    setTimeout(() => setVisible(false), dur);
  }, []);

  return (
    <Ctx.Provider value={{ show }}>
      {children}
      <div className={`toast${visible ? ' show' : ''}`}>{msg}</div>
    </Ctx.Provider>
  );
}
