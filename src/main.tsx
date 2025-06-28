import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import { StripeProvider } from './components/StripeProvider.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <StripeProvider>
      <App />
    </StripeProvider>
  </StrictMode>
);
