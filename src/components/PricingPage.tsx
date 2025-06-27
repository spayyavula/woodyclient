import React, { useState } from 'react';
import { stripeProducts, StripeProduct } from '../stripe-config';
import { supabase } from '../lib/supabase';
import { Check, Crown, Coffee, Heart, Users, Zap } from 'lucide-react';

interface PricingPageProps {
  onClose: () => void;
}

const PricingPage: React.FC<PricingPageProps> = ({ onClose }) => {
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState('');

  const handlePurchase = async (product: StripeProduct) => {
    setLoading(product.priceId);
    setError('');

    try {
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        setError('Please sign in to make a purchase');
        setLoading(null);
        return;
      }

      const response = await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/stripe-checkout`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          price_id: product.priceId,
          mode: product.mode,
          success_url: `${window.location.origin}/success`,
          cancel_url: `${window.location.origin}/pricing`,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to create checkout session');
      }

      if (data.url) {
        window.location.href = data.url;
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(null);
    }
  };

  const getProductIcon = (name: string) => {
    switch (name) {
      case 'Enterprise Plan':
        return <Crown className="w-8 h-8 text-yellow-400" />;
      case 'Premium Plan':
        return <Zap className="w-8 h-8 text-blue-400" />;
      case 'Sponsor Us':
        return <Users className="w-8 h-8 text-purple-400" />;
      case 'Support Us':
        return <Heart className="w-8 h-8 text-red-400" />;
      case 'Buy me coffee':
        return <Coffee className="w-8 h-8 text-orange-400" />;
      default:
        return <Zap className="w-8 h-8 text-gray-400" />;
    }
  };

  const getProductFeatures = (name: string) => {
    switch (name) {
      case 'Enterprise Plan':
        return [
          'Unlimited projects',
          'Priority support',
          'Custom whitelisting',
          'Advanced collaboration',
          'Enterprise security',
          'Custom integrations'
        ];
      case 'Premium Plan':
        return [
          'Advanced features',
          'Priority builds',
          'Extended storage',
          'Premium templates',
          'Advanced debugging'
        ];
      case 'Sponsor Us':
        return [
          'Support our team',
          'Help improve the platform',
          'Recognition in credits',
          'Early access to features'
        ];
      case 'Support Us':
        return [
          'Support development',
          'Help maintain servers',
          'Community recognition'
        ];
      case 'Buy me coffee':
        return [
          'Show appreciation',
          'Support the developer',
          'Fuel more coding sessions'
        ];
      default:
        return [];
    }
  };

  const subscriptionProducts = stripeProducts.filter(p => p.mode === 'subscription');
  const oneTimeProducts = stripeProducts.filter(p => p.mode === 'payment');

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-gray-800 rounded-lg max-w-6xl w-full p-6 border border-gray-700 my-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-white mb-2">Choose Your Plan</h2>
            <p className="text-gray-400">Unlock the full potential of Rust Cloud IDE</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-xl"
          >
            ✕
          </button>
        </div>

        {error && (
          <div className="bg-red-900/20 border border-red-500 text-red-300 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {/* Subscription Plans */}
        {subscriptionProducts.length > 0 && (
          <div className="mb-12">
            <h3 className="text-xl font-semibold text-white mb-6">Subscription Plans</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {subscriptionProducts.map((product) => (
                <div
                  key={product.priceId}
                  className="bg-gray-700 rounded-lg p-6 border border-gray-600 hover:border-orange-500 transition-colors relative"
                >
                  {product.name === 'Enterprise Plan' && (
                    <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                      <span className="bg-gradient-to-r from-yellow-400 to-orange-500 text-black px-3 py-1 rounded-full text-xs font-bold">
                        MOST POPULAR
                      </span>
                    </div>
                  )}
                  
                  <div className="text-center mb-6">
                    <div className="flex justify-center mb-4">
                      {getProductIcon(product.name)}
                    </div>
                    <h4 className="text-xl font-semibold text-white mb-2">{product.name}</h4>
                    <p className="text-gray-400 mb-4">{product.description}</p>
                    <div className="text-3xl font-bold text-white">
                      ${product.price}
                      <span className="text-lg text-gray-400">/month</span>
                    </div>
                  </div>

                  <ul className="space-y-3 mb-6">
                    {getProductFeatures(product.name).map((feature, index) => (
                      <li key={index} className="flex items-center space-x-2">
                        <Check className="w-4 h-4 text-green-400 flex-shrink-0" />
                        <span className="text-gray-300">{feature}</span>
                      </li>
                    ))}
                  </ul>

                  <button
                    onClick={() => handlePurchase(product)}
                    disabled={loading === product.priceId}
                    className="w-full py-3 px-4 bg-orange-600 hover:bg-orange-700 disabled:bg-orange-800 text-white font-medium rounded-lg transition-colors flex items-center justify-center"
                  >
                    {loading === product.priceId ? (
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      'Subscribe Now'
                    )}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* One-time Purchases */}
        {oneTimeProducts.length > 0 && (
          <div>
            <h3 className="text-xl font-semibold text-white mb-6">Support & One-time Purchases</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {oneTimeProducts.map((product) => (
                <div
                  key={product.priceId}
                  className="bg-gray-700 rounded-lg p-6 border border-gray-600 hover:border-blue-500 transition-colors"
                >
                  <div className="text-center mb-4">
                    <div className="flex justify-center mb-3">
                      {getProductIcon(product.name)}
                    </div>
                    <h4 className="text-lg font-semibold text-white mb-2">{product.name}</h4>
                    <p className="text-gray-400 text-sm mb-3">{product.description}</p>
                    <div className="text-2xl font-bold text-white">
                      ${product.price}
                    </div>
                  </div>

                  <ul className="space-y-2 mb-4">
                    {getProductFeatures(product.name).map((feature, index) => (
                      <li key={index} className="flex items-center space-x-2">
                        <Check className="w-3 h-3 text-green-400 flex-shrink-0" />
                        <span className="text-gray-300 text-sm">{feature}</span>
                      </li>
                    ))}
                  </ul>

                  <button
                    onClick={() => handlePurchase(product)}
                    disabled={loading === product.priceId}
                    className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white font-medium rounded-lg transition-colors flex items-center justify-center"
                  >
                    {loading === product.priceId ? (
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      'Purchase'
                    )}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="mt-8 text-center">
          <p className="text-gray-400 text-sm">
            All payments are processed securely through Stripe. Cancel anytime for subscriptions.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PricingPage;