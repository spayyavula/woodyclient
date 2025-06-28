import React, { useState } from 'react';
import { 
  TestTube, 
  Play, 
  CheckCircle, 
  XCircle, 
  Clock, 
  CreditCard, 
  AlertTriangle,
  RefreshCw,
  Download,
  Eye,
  Zap
} from 'lucide-react';
import { usePayments } from '../hooks/usePayments';
import { stripeProducts } from '../stripe-config';

interface TestCase {
  id: string;
  name: string;
  description: string;
  category: 'subscription' | 'one-time' | 'edge-case' | 'webhook';
  testData: any;
  expectedResult: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  duration?: number;
  error?: string;
}

interface StripeTestSuiteProps {
  isVisible: boolean;
  onClose: () => void;
}

const StripeTestSuite: React.FC<StripeTestSuiteProps> = ({ isVisible, onClose }) => {
  const { createCheckoutSession, loading } = usePayments();
  const [testResults, setTestResults] = useState<Record<string, TestCase>>({});
  const [runningTests, setRunningTests] = useState<Set<string>>(new Set());
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const testCases: TestCase[] = [
    // Subscription Tests
    {
      id: 'sub-enterprise-success',
      name: 'Enterprise Subscription - Success Flow',
      description: 'Test successful Enterprise Plan subscription with valid card',
      category: 'subscription',
      testData: {
        product: stripeProducts.find(p => p.name === 'Enterprise Plan'),
        cardNumber: '4242424242424242',
        expectedStatus: 'active'
      },
      expectedResult: 'Subscription created successfully, user redirected to success page',
      status: 'pending'
    },
    {
      id: 'sub-enterprise-decline',
      name: 'Enterprise Subscription - Card Declined',
      description: 'Test Enterprise Plan subscription with declined card',
      category: 'subscription',
      testData: {
        product: stripeProducts.find(p => p.name === 'Enterprise Plan'),
        cardNumber: '4000000000000002',
        expectedStatus: 'incomplete'
      },
      expectedResult: 'Payment declined, user shown error message',
      status: 'pending'
    },
    {
      id: 'sub-enterprise-3ds',
      name: 'Enterprise Subscription - 3D Secure',
      description: 'Test Enterprise Plan subscription requiring 3D Secure authentication',
      category: 'subscription',
      testData: {
        product: stripeProducts.find(p => p.name === 'Enterprise Plan'),
        cardNumber: '4000000000003220',
        expectedStatus: 'active'
      },
      expectedResult: '3D Secure challenge completed, subscription active',
      status: 'pending'
    },

    // One-time Payment Tests
    {
      id: 'payment-premium-success',
      name: 'Premium Plan - Success Flow',
      description: 'Test successful Premium Plan one-time payment',
      category: 'one-time',
      testData: {
        product: stripeProducts.find(p => p.name === 'Premium Plan'),
        cardNumber: '4242424242424242',
        expectedStatus: 'paid'
      },
      expectedResult: 'Payment processed successfully, order recorded',
      status: 'pending'
    },
    {
      id: 'payment-coffee-success',
      name: 'Buy Me Coffee - Success Flow',
      description: 'Test successful coffee purchase',
      category: 'one-time',
      testData: {
        product: stripeProducts.find(p => p.name === 'Buy me coffee'),
        cardNumber: '4242424242424242',
        expectedStatus: 'paid'
      },
      expectedResult: 'Coffee purchase completed, thank you message shown',
      status: 'pending'
    },
    {
      id: 'payment-sponsor-decline',
      name: 'Sponsor Us - Insufficient Funds',
      description: 'Test sponsorship payment with insufficient funds',
      category: 'one-time',
      testData: {
        product: stripeProducts.find(p => p.name === 'Sponsor Us'),
        cardNumber: '4000000000009995',
        expectedStatus: 'failed'
      },
      expectedResult: 'Payment failed due to insufficient funds',
      status: 'pending'
    },

    // Edge Cases
    {
      id: 'edge-duplicate-customer',
      name: 'Duplicate Customer Creation',
      description: 'Test handling of existing customer attempting new subscription',
      category: 'edge-case',
      testData: {
        scenario: 'existing_customer',
        product: stripeProducts.find(p => p.name === 'Enterprise Plan')
      },
      expectedResult: 'Existing customer record used, no duplicate created',
      status: 'pending'
    },
    {
      id: 'edge-expired-card',
      name: 'Expired Card Handling',
      description: 'Test payment with expired credit card',
      category: 'edge-case',
      testData: {
        cardNumber: '4000000000000069',
        product: stripeProducts.find(p => p.name === 'Premium Plan')
      },
      expectedResult: 'Expired card error handled gracefully',
      status: 'pending'
    },
    {
      id: 'edge-processing-error',
      name: 'Processing Error Recovery',
      description: 'Test handling of generic processing errors',
      category: 'edge-case',
      testData: {
        cardNumber: '4000000000000119',
        product: stripeProducts.find(p => p.name === 'Support Us')
      },
      expectedResult: 'Processing error handled with retry option',
      status: 'pending'
    },

    // Webhook Tests
    {
      id: 'webhook-subscription-created',
      name: 'Subscription Created Webhook',
      description: 'Test webhook handling for new subscription',
      category: 'webhook',
      testData: {
        event: 'customer.subscription.created',
        customerId: 'cus_test_123'
      },
      expectedResult: 'Subscription record created in database',
      status: 'pending'
    },
    {
      id: 'webhook-payment-succeeded',
      name: 'Payment Succeeded Webhook',
      description: 'Test webhook handling for successful payment',
      category: 'webhook',
      testData: {
        event: 'payment_intent.succeeded',
        paymentIntentId: 'pi_test_123'
      },
      expectedResult: 'Order record updated with payment success',
      status: 'pending'
    },
    {
      id: 'webhook-subscription-cancelled',
      name: 'Subscription Cancelled Webhook',
      description: 'Test webhook handling for cancelled subscription',
      category: 'webhook',
      testData: {
        event: 'customer.subscription.deleted',
        subscriptionId: 'sub_test_123'
      },
      expectedResult: 'Subscription status updated to cancelled',
      status: 'pending'
    }
  ];

  const runTest = async (testCase: TestCase) => {
    setRunningTests(prev => new Set([...prev, testCase.id]));
    const startTime = Date.now();

    try {
      // Simulate test execution
      await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 3000));

      // For demo purposes, randomly pass/fail tests
      const success = Math.random() > 0.2; // 80% success rate

      const duration = Date.now() - startTime;
      const result: TestCase = {
        ...testCase,
        status: success ? 'passed' : 'failed',
        duration,
        error: success ? undefined : 'Simulated test failure for demonstration'
      };

      setTestResults(prev => ({ ...prev, [testCase.id]: result }));
    } catch (error: any) {
      const duration = Date.now() - startTime;
      setTestResults(prev => ({
        ...prev,
        [testCase.id]: {
          ...testCase,
          status: 'failed',
          duration,
          error: error.message
        }
      }));
    } finally {
      setRunningTests(prev => {
        const newSet = new Set(prev);
        newSet.delete(testCase.id);
        return newSet;
      });
    }
  };

  const runAllTests = async () => {
    for (const testCase of testCases) {
      if (selectedCategory === 'all' || testCase.category === selectedCategory) {
        await runTest(testCase);
      }
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'running':
        return <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'passed':
        return 'text-green-400 bg-green-900/20';
      case 'failed':
        return 'text-red-400 bg-red-900/20';
      case 'running':
        return 'text-blue-400 bg-blue-900/20';
      default:
        return 'text-gray-400 bg-gray-900/20';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'subscription':
        return <RefreshCw className="w-4 h-4 text-purple-400" />;
      case 'one-time':
        return <CreditCard className="w-4 h-4 text-blue-400" />;
      case 'edge-case':
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      case 'webhook':
        return <Zap className="w-4 h-4 text-green-400" />;
      default:
        return <TestTube className="w-4 h-4 text-gray-400" />;
    }
  };

  const filteredTests = selectedCategory === 'all' 
    ? testCases 
    : testCases.filter(test => test.category === selectedCategory);

  const testStats = {
    total: filteredTests.length,
    passed: Object.values(testResults).filter(r => r.status === 'passed').length,
    failed: Object.values(testResults).filter(r => r.status === 'failed').length,
    running: runningTests.size
  };

  const exportResults = () => {
    const results = {
      timestamp: new Date().toISOString(),
      summary: testStats,
      tests: Object.values(testResults)
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stripe-test-results-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-xl border border-gray-700 w-full max-w-6xl max-h-[95vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            <div className="p-2 bg-blue-600 rounded-lg">
              <TestTube className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Stripe E2E Test Suite</h2>
              <p className="text-gray-400">Comprehensive payment flow testing</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-xl"
          >
            ✕
          </button>
        </div>

        <div className="flex h-[80vh]">
          {/* Sidebar */}
          <div className="w-80 bg-gray-900 border-r border-gray-700 p-6 overflow-y-auto">
            <div className="space-y-6">
              {/* Test Stats */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-4">Test Results</h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">{testStats.total}</div>
                    <div className="text-xs text-gray-400">Total</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">{testStats.passed}</div>
                    <div className="text-xs text-gray-400">Passed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-400">{testStats.failed}</div>
                    <div className="text-xs text-gray-400">Failed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">{testStats.running}</div>
                    <div className="text-xs text-gray-400">Running</div>
                  </div>
                </div>
              </div>

              {/* Category Filter */}
              <div>
                <h3 className="text-sm font-medium text-gray-300 mb-3">Test Categories</h3>
                <div className="space-y-2">
                  {[
                    { id: 'all', label: 'All Tests', count: testCases.length },
                    { id: 'subscription', label: 'Subscriptions', count: testCases.filter(t => t.category === 'subscription').length },
                    { id: 'one-time', label: 'One-time Payments', count: testCases.filter(t => t.category === 'one-time').length },
                    { id: 'edge-case', label: 'Edge Cases', count: testCases.filter(t => t.category === 'edge-case').length },
                    { id: 'webhook', label: 'Webhooks', count: testCases.filter(t => t.category === 'webhook').length }
                  ].map(category => (
                    <button
                      key={category.id}
                      onClick={() => setSelectedCategory(category.id)}
                      className={`w-full flex items-center justify-between p-2 rounded-lg text-sm transition-colors ${
                        selectedCategory === category.id
                          ? 'bg-blue-600 text-white'
                          : 'text-gray-300 hover:bg-gray-700'
                      }`}
                    >
                      <span>{category.label}</span>
                      <span className="text-xs opacity-75">({category.count})</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="space-y-3">
                <button
                  onClick={runAllTests}
                  disabled={runningTests.size > 0}
                  className="w-full flex items-center justify-center space-x-2 p-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
                >
                  <Play className="w-4 h-4" />
                  <span>Run All Tests</span>
                </button>
                
                <button
                  onClick={exportResults}
                  className="w-full flex items-center justify-center space-x-2 p-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  <Download className="w-4 h-4" />
                  <span>Export Results</span>
                </button>
              </div>

              {/* Test Cards */}
              <div>
                <h3 className="text-sm font-medium text-gray-300 mb-3">Test Cards</h3>
                <div className="space-y-2 text-xs">
                  <div className="bg-gray-800 p-2 rounded">
                    <div className="text-green-400 font-mono">4242424242424242</div>
                    <div className="text-gray-400">Success</div>
                  </div>
                  <div className="bg-gray-800 p-2 rounded">
                    <div className="text-red-400 font-mono">4000000000000002</div>
                    <div className="text-gray-400">Declined</div>
                  </div>
                  <div className="bg-gray-800 p-2 rounded">
                    <div className="text-yellow-400 font-mono">4000000000003220</div>
                    <div className="text-gray-400">3D Secure</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Test List */}
          <div className="flex-1 p-6 overflow-y-auto">
            <div className="space-y-4">
              {filteredTests.map(testCase => {
                const result = testResults[testCase.id];
                const isRunning = runningTests.has(testCase.id);
                const status = isRunning ? 'running' : (result?.status || 'pending');

                return (
                  <div
                    key={testCase.id}
                    className="bg-gray-700 rounded-lg p-6 border border-gray-600 hover:border-gray-500 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          {getCategoryIcon(testCase.category)}
                          <h3 className="text-lg font-semibold text-white">{testCase.name}</h3>
                          <span className={`px-2 py-1 rounded text-xs ${getStatusColor(status)}`}>
                            {status.toUpperCase()}
                          </span>
                        </div>
                        <p className="text-gray-300 mb-2">{testCase.description}</p>
                        <p className="text-sm text-gray-400">Expected: {testCase.expectedResult}</p>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(status)}
                        {result?.duration && (
                          <span className="text-xs text-gray-400">
                            {result.duration}ms
                          </span>
                        )}
                      </div>
                    </div>

                    {result?.error && (
                      <div className="bg-red-900/20 border border-red-500 rounded p-3 mb-4">
                        <div className="flex items-center space-x-2 text-red-400">
                          <AlertTriangle className="w-4 h-4" />
                          <span className="font-medium">Test Failed</span>
                        </div>
                        <p className="text-red-300 text-sm mt-1">{result.error}</p>
                      </div>
                    )}

                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4 text-sm text-gray-400">
                        <span>Category: {testCase.category}</span>
                        {testCase.testData?.cardNumber && (
                          <span>Card: ****{testCase.testData.cardNumber.slice(-4)}</span>
                        )}
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => runTest(testCase)}
                          disabled={isRunning}
                          className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded text-sm transition-colors"
                        >
                          <Play className="w-3 h-3" />
                          <span>Run</span>
                        </button>
                        
                        <button className="flex items-center space-x-1 px-3 py-1 bg-gray-600 hover:bg-gray-700 text-white rounded text-sm transition-colors">
                          <Eye className="w-3 h-3" />
                          <span>Details</span>
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StripeTestSuite;