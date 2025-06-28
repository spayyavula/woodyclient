import React from 'react';
import { Crown, Calendar, CreditCard, AlertTriangle, CheckCircle } from 'lucide-react';
import { useSubscription } from '../hooks/useSubscription';

interface SubscriptionStatusProps {
  className?: string;
}

const SubscriptionStatus: React.FC<SubscriptionStatusProps> = ({ className = '' }) => {
  const { subscription, loading, error, hasActiveSubscription, currentPlan } = useSubscription();

  if (loading) {
    return (
      <div className={`animate-pulse ${className}`}>
        <div className="h-4 bg-gray-700 rounded w-24"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`flex items-center space-x-2 text-red-400 ${className}`}>
        <AlertTriangle className="w-4 h-4" />
        <span className="text-sm">Error loading subscription</span>
      </div>
    );
  }

  const getStatusIcon = () => {
    if (hasActiveSubscription) {
      return <CheckCircle className="w-4 h-4 text-green-400" />;
    }
    return <Crown className="w-4 h-4 text-gray-400" />;
  };

  const getStatusColor = () => {
    if (hasActiveSubscription) {
      return 'text-green-400';
    }
    return 'text-gray-400';
  };

  const formatDate = (timestamp: number | null) => {
    if (!timestamp) return null;
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      {getStatusIcon()}
      <div className="flex flex-col">
        <span className={`text-sm font-medium ${getStatusColor()}`}>
          {currentPlan}
        </span>
        {subscription && subscription.current_period_end && hasActiveSubscription && (
          <span className="text-xs text-gray-500">
            Renews {formatDate(subscription.current_period_end)}
          </span>
        )}
        {subscription && subscription.cancel_at_period_end && (
          <span className="text-xs text-yellow-400">
            Cancels {formatDate(subscription.current_period_end)}
          </span>
        )}
      </div>
    </div>
  );
};

export default SubscriptionStatus;