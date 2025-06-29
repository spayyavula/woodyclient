import { useState, useEffect } from 'react';
import { supabase, cachedQuery, createSupabaseQueryKey } from '../lib/supabase';
import { getProductByPriceId } from '../stripe-config';
import { CACHE_EXPIRATION } from '../utils/cacheUtils';

interface Subscription {
  subscription_status: string;
  price_id: string | null;
  current_period_end: number | null;
  cancel_at_period_end: boolean;
  payment_method_brand: string | null;
  payment_method_last4: string | null;
}

interface UseSubscriptionReturn {
  subscription: Subscription | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  hasActiveSubscription: boolean;
  currentPlan: string;
}

export const useSubscription = (): UseSubscriptionReturn => {
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSubscription = async () => {
    try {
      setLoading(true);
      setError(null);

      // Check if user is authenticated first
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        // User not authenticated, set to null and return
        setSubscription(null);
        return;
      }

      // Create a cache key based on the user ID
      const cacheKey = createSupabaseQueryKey('stripe_user_subscriptions', 'select', { user_id: session.user.id });
      
      // Use cached query with a longer expiration for subscription data
      const { data, error: fetchError } = await cachedQuery(
        () => supabase.from('stripe_user_subscriptions').select('*').maybeSingle(),
        cacheKey,
        CACHE_EXPIRATION.LONG
      );

      if (fetchError) {
        // If it's an auth error, just set subscription to null
        if (fetchError.code === '401' || fetchError.message.includes('authorization')) {
          setSubscription(null);
          return;
        }
        throw fetchError;
      }

      setSubscription(data);
    } catch (err: any) {
      console.error('Error fetching subscription:', err);
      // Don't show auth errors to user, just set subscription to null
      if (err.code === '401' || err.message.includes('authorization')) {
        setSubscription(null);
      } else {
        setError(err.message || 'Failed to fetch subscription');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSubscription();
  }, []);

  const hasActiveSubscription = subscription?.subscription_status === 'active';

  const getCurrentPlan = (): string => {
    if (!subscription || !subscription.price_id) {
      return 'Free Plan';
    }

    const product = getProductByPriceId(subscription.price_id);
    return product?.name || 'Unknown Plan';
  };

  return {
    subscription,
    loading,
    error,
    refetch: fetchSubscription,
    hasActiveSubscription,
    currentPlan: getCurrentPlan(),
  };
};