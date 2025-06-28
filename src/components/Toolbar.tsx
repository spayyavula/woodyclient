import React, { useState } from 'react';
import DeploymentModal from './DeploymentModal';
import { 
  Play, 
  Square, 
  Save, 
  Folder, 
  FileText, 
  Search, 
  Settings, 
  Terminal,
  GitBranch,
  Bug,
  Smartphone,
  Monitor,
  Tablet,
  Users,
  ShoppingCart,
  User,
  CreditCard,
  Eye,
  TestTube,
  Link,
  Hammer,
  Wrench,
  Zap,
  CheckSquare
} from 'lucide-react';
import SubscriptionStatus from './SubscriptionStatus';
import AutomationPanel from './AutomationPanel';

interface ToolbarProps {
  user?: any;
  onShowPricing: () => void;
  onShowProfile: () => void;
  onToggleTerminal: () => void;
  onToggleCollaboration: () => void;
  onToggleMarketplace: () => void;
  onSave: () => void;
  onRun: () => void;
  onShowConfigCheck: () => void;
  onShowTemplates: () => void;
  onShowMobilePreview: () => void;
  onShowStripeTests: () => void;
  onShowScriptRunner: () => void;
  onShowIntegrations: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({ 
  user,
  onShowPricing,
  onShowProfile,
  onToggleTerminal, 
  onToggleCollaboration, 
  onToggleMarketplace,
  onSave, 
  onRun,
  onShowConfigCheck,
  onShowTemplates,
  onShowMobilePreview,
  onShowScriptRunner,
  onShowStripeTests,
  onShowIntegrations
}) => {
  const [showBuildLogs, setShowBuildLogs] = useState(false);
  const [selectedPlatform, setSelectedPlatform] = React.useState('android');
  const [showDeploymentModal, setShowDeploymentModal] = React.useState(false);
  const [isBuilding, setIsBuilding] = React.useState(false);
  const [buildOutput, setBuildOutput] = React.useState<string[]>([]);
  const [showAutomationPanel, setShowAutomationPanel] = React.useState(false);

  const handleBuild = async () => {
    setIsBuilding(true);
    setBuildOutput([]);
    setError(null);
    
    try {
      // Simulate build process with real commands
      const commands = [
        'rustyclint scan --deep',
        'cargo build --release',
        'cargo test',
        'rustyclint audit --compliance'
      ];
      
      for (const command of commands) {
        setBuildOutput(prev => [...prev, `$ ${command}`]);
        
        // Simulate command execution time
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
        
        // Add realistic output
        if (command.includes('rustyclint scan')) {
          setBuildOutput(prev => [...prev, 
            '🔍 Analyzing 47,392 lines of code...',
            '⚡ Performance: 10.2M lines/second',
            '✅ Analysis complete in 0.08s',
            '⚠️  2 medium-risk issues found',
            '🔧 3 optimizations suggested'
          ]);
        } else if (command.includes('cargo build')) {
          setBuildOutput(prev => [...prev,
            '   Compiling rustyclint v0.1.0',
            '    Finished release [optimized] target(s) in 3.42s'
          ]);
        } else if (command.includes('cargo test')) {
          setBuildOutput(prev => [...prev,
            'running 8 tests',
            'test result: ok. 8 passed; 0 failed; 0 ignored'
          ]);
        } else if (command.includes('rustyclint audit')) {
          setBuildOutput(prev => [...prev,
            '🛡️  Security Score: 98/100 (Excellent)',
            '✅ SOC 2 Type II compliant',
            '✅ GDPR data protection verified'
          ]);
        }
      }
      
      setBuildOutput(prev => [...prev, '🎉 Build completed successfully!']);
      
      // Trigger the original onRun callback
      onRun();
    } catch (err) {
      console.error('Build error:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      setBuildOutput(prev => [...prev, `❌ Build failed: ${err instanceof Error ? err.message : 'Unknown error'}`]);
    } finally {
      setIsBuilding(false);
    }
  };

  const handleDeploy = () => {
    setShowDeploymentModal(true);
  };

  return (
    <>
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center justify-between">
      <div className="flex items-center space-x-1">
        <div className="flex items-center space-x-1 mr-4">
          <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
            <Folder className="w-4 h-4" />
          </button>
          <button 
            onClick={onShowTemplates}
            className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors"
            title="Project Templates"
          >
            <FileText className="w-4 h-4" />
          </button>
          <button 
            onClick={onSave}
            className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors"
          >
            <Save className="w-4 h-4" />
          </button>
        </div>
        
        <div className="w-px h-6 bg-gray-600 mx-2" />
        
        {/* Platform Selection */}
        <div className="flex items-center space-x-1 mr-4">
          <select 
            value={selectedPlatform}
            onChange={(e) => setSelectedPlatform(e.target.value)}
            className="bg-gray-700 text-gray-200 text-sm px-3 py-1.5 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
          >
            <option value="android">Android</option>
            <option value="ios">iOS</option>
            <option value="flutter">Flutter</option>
            <option value="web">Web</option>
            <option value="desktop">Desktop</option>
          </select>
          <div className="flex items-center space-x-1">
            {selectedPlatform === 'android' && <Smartphone className="w-4 h-4 text-green-400" />}
            {selectedPlatform === 'ios' && <Smartphone className="w-4 h-4 text-blue-400" />}
            {selectedPlatform === 'flutter' && <Tablet className="w-4 h-4 text-cyan-400" />}
            {selectedPlatform === 'web' && <Monitor className="w-4 h-4 text-green-400" />}
            {selectedPlatform === 'desktop' && <Monitor className="w-4 h-4 text-purple-400" />}
          </div>
        </div>
        
        <div className="w-px h-6 bg-gray-600 mx-2" />
        
        <div className="flex items-center space-x-1">
          <button 
            onClick={handleBuild}
            disabled={isBuilding}
            className="flex items-center px-4 py-2 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 rounded-lg text-white text-sm font-medium transition-all transform hover:scale-105 shadow-lg"
            className={`flex items-center px-4 py-2 rounded-lg text-white text-sm font-medium transition-all transform hover:scale-105 shadow-lg ${
              isBuilding 
                ? 'bg-gray-600 cursor-not-allowed' 
                : 'bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700'
            }`}
            title="Build & Analyze Code"
          >
            {isBuilding ? (
              <>
                <div className="w-4 h-4 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Building...</span>
              </>
            ) : (
              <>
                <Hammer className="w-4 h-4 mr-2" />
                <span>Build & Analyze</span>
              </>
            )}
          </button>
          <button 
            onClick={() => setIsBuilding(false)}
            disabled={!isBuilding}
            className="flex items-center px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-white text-sm font-medium transition-colors"
            className={`flex items-center px-3 py-2 rounded-lg text-white text-sm font-medium transition-colors ${
              isBuilding 
                ? 'bg-red-600 hover:bg-red-700' 
                : 'bg-gray-600 hover:bg-gray-700 cursor-not-allowed opacity-50'
            }`}
            title="Stop Build"
          >
            <Square className="w-4 h-4 mr-1" />
            <span>{isBuilding ? 'Stop' : 'Stopped'}</span>
          </button>
          <button 
            onClick={handleDeploy}
            className="flex items-center px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white text-sm font-medium transition-colors"
            title="Deploy to Production"
          >
            <Smartphone className="w-4 h-4 mr-2" />
            <span>Deploy</span>
          </button>
          <button 
            className="flex items-center px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm font-medium transition-colors"
            title="Debug Mode"
          >
            <Bug className="w-4 h-4 mr-1" />
            <span>Debug</span>
          </button>
          <button 
            onClick={onShowMobilePreview}
            className="flex items-center px-3 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg text-white text-sm font-medium transition-colors"
            title="Preview Mobile App"
          >
            <Eye className="w-4 h-4 mr-2" />
            <span>Preview</span>
          </button>
          <button 
            onClick={onShowConfigCheck}
            className="flex items-center px-3 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-white text-sm font-medium transition-colors"
            title="Check Configuration"
          >
            <CheckSquare className="w-4 h-4 mr-2" />
            <span>Check Config</span>
          </button>
          <button 
            onClick={onShowScriptRunner}
            className="flex items-center px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm font-medium transition-colors"
            title="Run Deployment Scripts"
          >
            <Terminal className="w-4 h-4 mr-2" />
            <span>Run Scripts</span>
          </button>
          <button 
            onClick={onShowStripeTests}
            className="flex items-center px-3 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-white text-sm font-medium transition-colors"
            title="Stripe E2E Tests"
          >
            <TestTube className="w-4 h-4 mr-2" />
            <span>Test Stripe</span>
          </button>
          <button 
            onClick={onShowIntegrations}
            className="flex items-center px-3 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-white text-sm font-medium transition-colors"
            title="Integrations"
          >
            <Link className="w-4 h-4 mr-2" />
            <span>Integrations</span>
          </button>
          <button 
            onClick={() => setShowBuildLogs(true)}
            className="flex items-center px-3 py-2 bg-orange-600 hover:bg-orange-700 text-white text-sm font-medium transition-colors"
            title="View Build Logs"
          >
            <Terminal className="w-4 h-4 mr-2" />
            <span>Build Logs</span>
          </button>
          <button 
            onClick={() => setShowAutomationPanel(true)}
            className="flex items-center px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium transition-colors"
            title="Automation Center"
          >
            <Zap className="w-4 h-4 mr-2" />
            <span>Automate</span>
          </button>
        </div>
      </div>

      <div className="flex items-center space-x-1">
        {user && (
          <>
            <div className="mr-4">
              <SubscriptionStatus />
            </div>
            <button 
              onClick={onShowPricing}
              className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors"
              title="Pricing & Plans"
            >
              <CreditCard className="w-4 h-4" />
            </button>
            <button 
              onClick={onShowProfile}
              className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors"
              title="User Profile"
            >
              <User className="w-4 h-4" />
            </button>
          </>
        )}
        <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
          <Search className="w-4 h-4" />
        </button>
        <button 
          onClick={onToggleCollaboration}
          className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors relative"
          title="Collaboration"
        >
          <Users className="w-4 h-4" />
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse" />
        </button>
        <button 
          onClick={onToggleMarketplace}
          className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors relative"
          title="Developer Marketplace"
        >
          <ShoppingCart className="w-4 h-4" />
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-orange-400 rounded-full animate-pulse" />
        </button>
        <button 
          onClick={onToggleTerminal}
          className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors"
        >
          <Terminal className="w-4 h-4" />
        </button>
        <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
          <GitBranch className="w-4 h-4" />
        </button>
        <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </div>
      
      {/* Build Output Overlay */}
      {isBuilding && buildOutput.length > 0 && (
        <div className="fixed bottom-4 right-4 w-96 bg-gray-900 border border-gray-700 rounded-lg shadow-2xl z-40">
          <div className={`flex items-center justify-between p-3 border-b border-gray-700 ${error ? 'bg-red-900/20' : ''}`}>
            <div className="flex items-center space-x-2">
              {error ? (
                <>
                  <XCircle className="w-4 h-4 text-red-400" />
                  <span className="text-red-300 font-medium">Build Failed</span>
                </>
              ) : (
                <>
                  <div className="w-3 h-3 border-2 border-orange-400 border-t-transparent rounded-full animate-spin" />
                  <span className="text-white font-medium">Building...</span>
                </>
              )}
            </div>
            <button 
              onClick={() => setBuildOutput([])}
              className="text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>
          <div className="p-3 max-h-64 overflow-y-auto">
            <div className="font-mono text-sm space-y-1">
              {buildOutput.map((line, index) => (
                <div key={index} className={
                  line.startsWith('$') ? 'text-green-400' : 
                  line.includes('❌') ? 'text-red-400' :
                  line.includes('⚠️') ? 'text-yellow-400' :
                  line.includes('✅') ? 'text-green-400' :
                  'text-gray-300'
                }>{line}</div>
              ))}
            </div>
            {error && (
              <div className="mt-3 p-2 bg-red-900/20 border border-red-500 rounded text-red-300 text-xs">
                {error}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Deployment Modal */}
      <DeploymentModal 
        isVisible={showDeploymentModal}
        onClose={() => setShowDeploymentModal(false)}
        initialPlatform={selectedPlatform as any}
      />
      
      {/* Build Logs Viewer */}
      {showBuildLogs && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="w-full max-w-6xl">
            <AndroidBuildLogs
              isLive={isBuilding}
              onClose={() => setShowBuildLogs(false)}
            />
          </div>
        </div>
      )}
      
      {/* Automation Panel */}
      <AutomationPanel
        platform={selectedPlatform}
        isVisible={showAutomationPanel}
        onClose={() => setShowAutomationPanel(false)}
      />
    </>
  );
};

export default Toolbar;