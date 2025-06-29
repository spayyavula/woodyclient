import React, { createContext, useContext, useEffect, useState } from 'react';
import { loadStripe, Stripe } from '@stripe/stripe-js';
import { getCachedData, setCache, CACHE_EXPIRATION, createCacheKey } from '../utils/cacheUtils';

interface StripeContextType {
  stripe: Stripe | null;
  isLoading: boolean;
  error: string | null;
}

const StripeContext = createContext<StripeContextType>({
  stripe: null,
  isLoading: true,
  error: null,
});

export const useStripe = () => {
  const context = useContext(StripeContext);
  if (!context) {
    throw new Error('useStripe must be used within a StripeProvider');
  }
  return context;
};

interface StripeProviderProps {
  children: React.ReactNode;
}

export const StripeProvider: React.FC<StripeProviderProps> = ({ children }) => {
  const [stripe, setStripe] = useState<Stripe | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initializeStripe = async () => {
      try {
        setIsLoading(true);
        const stripePublishableKey = import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY;
        
        if (!stripePublishableKey) {
          if (import.meta.env.DEV) {
            console.warn('Stripe publishable key is not configured. Stripe features will be disabled.');
          }
          setError('Stripe not configured');
          setIsLoading(false);
          return;
        }

        // Try to get cached Stripe instance ID
        const cacheKey = createCacheKey('stripe', 'instance-id');
        const cachedStripeId = await getCachedData<string>(
          cacheKey,
          async () => '',
          CACHE_EXPIRATION.VERY_LONG
        );
        
        // Load Stripe with the cached ID if available
        const stripeInstance = await loadStripe(stripePublishableKey, {
          stripeAccount: cachedStripeId || undefined
        });
        
        if (!stripeInstance) {
          throw new Error('Failed to load Stripe');
        }

        // Cache the Stripe instance ID for future use
        if (stripeInstance._id) {
          await setCache(cacheKey, stripeInstance._id, CACHE_EXPIRATION.VERY_LONG);
        }

        setStripe(stripeInstance);
      } catch (err: any) {
        console.error('Error initializing Stripe:', err);
        setError(err.message || 'Failed to initialize Stripe');
      } finally {
        setIsLoading(false);
      }
    };

    initializeStripe();
  }, []);

  const value: StripeContextType = {
    stripe,
    isLoading,
    error,
  };

  return (
    <StripeContext.Provider value={value}>
      {children}
    </StripeContext.Provider>
  );
};