import { useState, useEffect, useCallback } from 'react';
import { useUserSubscription } from './useOptimizedQuery';

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
  const [user, setUser] = useState<{ id: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Get user session
  useEffect(() => {
    const getUser = async () => {
      const { data } = await supabase.auth.getSession();
      setUser(data.session?.user || null);
    };
    
    getUser();
  }, []);
  
  // Use the optimized query hook
  const { 
    data: subscriptionData,
    isLoading: subscriptionLoading,
    error: subscriptionError,
    refetch
  } = useUserSubscription(user?.id);
  
  // Map the subscription data to the expected format
  const subscription: Subscription | null = subscriptionData ? {
    subscription_status: subscriptionData.subscription_status,
    price_id: subscriptionData.price_id,
    current_period_end: subscriptionData.current_period_end,
    cancel_at_period_end: subscriptionData.cancel_at_period_end,
    payment_method_brand: subscriptionData.payment_method_brand,
    payment_method_last4: subscriptionData.payment_method_last4
  } : null;

  // Update loading and error states
  useEffect(() => {
    setLoading(subscriptionLoading);
    setError(subscriptionError ? String(subscriptionError) : null);
  }, [subscriptionLoading, subscriptionError]);

  // Memoize the refetch function
  const fetchSubscription = useCallback(() => {
    return refetch();
  }, [refetch]);

  const hasActiveSubscription = subscription?.subscription_status === 'active';

  const getCurrentPlan = (): string => {
    if (!subscriptionData) {
      return 'Free Plan';
    }
    
    return subscriptionData.product_name || 'Unknown Plan';
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