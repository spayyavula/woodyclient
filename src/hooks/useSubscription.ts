import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { getProductByPriceId } from '../stripe-config';

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

      // Check if we're in demo mode
      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
      if (!supabaseUrl || supabaseUrl === 'your_supabase_project_url_here' || supabaseUrl === 'https://demo.supabase.co') {
        // Return mock data for demo mode
        setSubscription(null);
        setLoading(false);
        return;
      }

      const { data, error: fetchError } = await supabase
        .from('stripe_user_subscriptions')
        .select('*')
        .maybeSingle();

      if (fetchError) {
        throw fetchError;
      }

      setSubscription(data);
    } catch (err: any) {
      console.error('Error fetching subscription:', err);
      // Don't show error in demo mode
      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
      if (supabaseUrl && supabaseUrl !== 'your_supabase_project_url_here' && supabaseUrl !== 'https://demo.supabase.co') {
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