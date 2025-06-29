import React from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from 'react-query';
import App from './App.tsx';
import { StripeProvider } from './components/StripeProvider.tsx';
import './index.css';

// Create a client for React Query with caching configuration
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      refetchOnMount: false,
      refetchOnReconnect: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 30 * 60 * 1000, // 30 minutes
    },
  },
});

const root = createRoot(document.getElementById('root')!);

root.render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <StripeProvider>
        <App />
      </StripeProvider>
    </QueryClientProvider>
  </React.StrictMode>
);

// Preload critical components
if ('requestIdleCallback' in window) {
  window.requestIdleCallback(() => {
    import('./components/LandingPage');
    import('./components/auth/LoginPage');
    import('./components/auth/SignupPage');
  });
}
