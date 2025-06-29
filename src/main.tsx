import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import { StripeProvider } from './components/StripeProvider.tsx';
import './index.css';

// Remove StrictMode in production for better performance
const root = createRoot(document.getElementById('root')!);

root.render(
  <React.StrictMode>
    <StripeProvider>
      <App />
    </StripeProvider>
  </React.StrictMode>
);
