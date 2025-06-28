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
  Zap,
  Shield,
  Database,
  Settings
} from 'lucide-react';
import { useStripeTests } from '../hooks/useStripeTests';

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
  const { runTest, runAllTests, testResults, isRunning, clearResults } = useStripeTests();
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const testCases: TestCase[] = [
    // Client-side Integration Tests
    {
      id: 'db-connection',
      name: 'Database Connection',
      description: 'Test Supabase database connectivity using anonymous key',
      category: 'integration',
      testData: {
        endpoint: 'stripe_customers',
        method: 'SELECT'
      },
      expectedResult: 'Database connection established successfully',
      status: 'pending'
    },
    {
      id: 'auth-flow',
      name: 'Authentication Flow',
      description: 'Test authentication simulation and user session handling',
      category: 'integration',
      testData: {
        userEmail: 'test@example.com',
        sessionType: 'mock'
      },
      expectedResult: 'Authentication flow works correctly',
      status: 'pending'
    },
    {
      id: 'stripe-config',
      name: 'Stripe Configuration',
      description: 'Validate Stripe product configuration and pricing data',
      category: 'configuration',
      testData: {
        products: 'all',
        validation: 'complete'
      },
      expectedResult: 'All Stripe products properly configured',
      status: 'pending'
    },

    {
      id: 'checkout-endpoint',
      name: 'Checkout Endpoint',
      description: 'Test Stripe checkout endpoint availability and response',
      category: 'endpoint',
      testData: {
        endpoint: '/functions/v1/stripe-checkout',
        method: 'POST'
      },
      expectedResult: 'Checkout endpoint accessible and responding',
      status: 'pending'
    },
    {
      id: 'webhook-endpoint',
      name: 'Webhook Endpoint',
      description: 'Test Stripe webhook endpoint availability and signature validation',
      category: 'endpoint',
      testData: {
        endpoint: '/functions/v1/stripe-webhook',
        method: 'POST'
      },
      expectedResult: 'Webhook endpoint accessible with proper validation',
      status: 'pending'
    },
    {
      id: 'data-validation',
      name: 'Data Validation',
      description: 'Test database constraints and data validation rules',
      category: 'security',
      testData: {
        table: 'stripe_subscriptions',
        constraint: 'status_enum'
      },
      expectedResult: 'Data validation rules enforced correctly',
      status: 'pending'
    },

    {
      id: 'env-config',
      name: 'Environment Configuration',
      description: 'Validate all required environment variables are properly set',
      category: 'configuration',
      testData: {
        variables: ['VITE_SUPABASE_URL', 'VITE_SUPABASE_ANON_KEY', 'VITE_STRIPE_PUBLISHABLE_KEY']
      },
      expectedResult: 'All environment variables configured correctly',
      status: 'pending'
    },
    {
      id: 'client-integration',
      name: 'Client Integration',
      description: 'Test client-side hooks and components integration',
      category: 'integration',
      testData: {
        hooks: ['usePayments', 'useSubscription'],
        components: 'all'
      },
      expectedResult: 'Client-side integration working correctly',
      status: 'pending'
    }
  ];

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
      case 'integration':
        return <Zap className="w-4 h-4 text-blue-400" />;
      case 'configuration':
        return <Settings className="w-4 h-4 text-purple-400" />;
      case 'endpoint':
        return <Zap className="w-4 h-4 text-green-400" />;
      case 'security':
        return <Shield className="w-4 h-4 text-red-400" />;
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
    running: Object.values(testResults).filter(r => r.status === 'running').length
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
                    { id: 'integration', label: 'Integration', count: testCases.filter(t => t.category === 'integration').length },
                    { id: 'configuration', label: 'Configuration', count: testCases.filter(t => t.category === 'configuration').length },
                    { id: 'endpoint', label: 'Endpoints', count: testCases.filter(t => t.category === 'endpoint').length },
                    { id: 'security', label: 'Security', count: testCases.filter(t => t.category === 'security').length }
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
                  disabled={isRunning}
                  className="w-full flex items-center justify-center space-x-2 p-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:opacity-50 text-white rounded-lg transition-colors"
                >
                  {isRunning ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Running Tests...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Run All Tests</span>
                    </>
                  )}
                </button>
                
                <button
                  onClick={clearResults}
                  className="w-full flex items-center justify-center space-x-2 p-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Clear Results</span>
                </button>
              </div>

              {/* Security Notice */}
              <div>
                <h3 className="text-sm font-medium text-gray-300 mb-3">Security Notice</h3>
                <div className="bg-green-900/20 border border-green-500 rounded p-3">
                  <div className="flex items-center space-x-2 text-green-400 mb-2">
                    <Shield className="w-4 h-4" />
                    <span className="font-medium text-xs">Client-side Testing</span>
                  </div>
                  <p className="text-xs text-green-300">
                    These tests use only the anonymous Supabase key for maximum security. 
                    No service role key required.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Test List */}
          <div className="flex-1 p-6 overflow-y-auto">
            <div className="space-y-4">
              {filteredTests.map(testCase => {
                const result = testResults[testCase.id];
                const status = result?.status || 'pending';
                const isTestRunning = status === 'running';

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
                          onClick={() => runTest(testCase.id)}
                          disabled={isTestRunning || isRunning}
                          className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:opacity-50 text-white rounded text-sm transition-colors"
                        >
                          {isTestRunning ? (
                            <>
                              <RefreshCw className="w-3 h-3 animate-spin" />
                              <span>Running</span>
                            </>
                          ) : (
                            <>
                              <Play className="w-3 h-3" />
                              <span>Run</span>
                            </>
                          )}
                        </button>
                        
                        {result && (
                          <button className="flex items-center space-x-1 px-3 py-1 bg-gray-600 hover:bg-gray-700 text-white rounded text-sm transition-colors">
                            <Eye className="w-3 h-3" />
                            <span>Details</span>
                          </button>
                        )}
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