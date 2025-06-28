import React, { useState, useEffect } from 'react';
import { 
  X, 
  Play, 
  Square, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Smartphone,
  Monitor,
  Tablet,
  Settings,
  Terminal,
  ExternalLink,
  Download,
  Upload,
  Loader2,
  Zap,
  Shield,
  Package,
  TrendingUp,
  Globe,
  Cpu,
  HardDrive,
  Wifi,
  RefreshCw
} from 'lucide-react';
import { useDeployment, DeploymentConfig, DeploymentStep } from '../hooks/useDeployment';
import DeploymentAssistant from './DeploymentAssistant';
import { useDeploymentAutomation } from '../hooks/useDeploymentAutomation';

interface DeploymentModalProps {
  isVisible: boolean;
  onClose: () => void;
  initialPlatform?: 'ios' | 'android' | 'flutter' | 'desktop' | 'web';
}

const DeploymentModal: React.FC<DeploymentModalProps> = ({ 
  isVisible, 
  onClose, 
  initialPlatform = 'ios' 
}) => {
  const { 
    isDeploying, 
    currentStep, 
    steps, 
    deploymentConfig, 
    startDeployment, 
    stopDeployment,
    getStepProgress,
    getOverallProgress
  } = useDeployment();

  const [config, setConfig] = useState<DeploymentConfig>({
    platform: initialPlatform,
    buildType: 'release',
    webTarget: 'spa',
    desktopTarget: 'all',
    androidTarget: 'aab'
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isConfiguring, setIsConfiguring] = useState(false);
  const [showPreDeployCheck, setShowPreDeployCheck] = useState(false);
  const [realTimeMetrics, setRealTimeMetrics] = useState({
    buildSpeed: 0,
    memoryUsage: 0,
    cpuUsage: 0,
    networkSpeed: 0
  });
  const [deploymentError, setDeploymentError] = useState<Error | null>(null);
  const [showAssistant, setShowAssistant] = useState(true);
  const [assistantCurrentStep, setAssistantCurrentStep] = useState('');
  
  const { runAutomation, automationSteps, isRunning: automationRunning } = useDeploymentAutomation();

  // Simulate real-time metrics during deployment
  useEffect(() => {
    if (isDeploying) {
      const interval = setInterval(() => {
        setRealTimeMetrics({
          buildSpeed: 50 + Math.random() * 100, // MB/s
          memoryUsage: 30 + Math.random() * 40, // %
          cpuUsage: 20 + Math.random() * 60, // %
          networkSpeed: 10 + Math.random() * 50 // Mbps
        });
      }, 1000);
      
      return () => clearInterval(interval);
    }
  }, [isDeploying]);

  const handleStartDeployment = () => {
    setIsConfiguring(true);
    setDeploymentError(null);
    
    // Show configuration animation
    try {
      setTimeout(() => {
        setIsConfiguring(false);
        setShowPreDeployCheck(true);
      }, 2000);
    } catch (error) {
      setDeploymentError(error instanceof Error ? error : new Error('An unknown error occurred'));
      setIsConfiguring(false);
    }
  };

  const handleConfirmDeployment = () => {
    try {
      setShowPreDeployCheck(false);
      setAssistantCurrentStep('rust-build');
      startDeployment(config);
    } catch (error) {
      setDeploymentError(error instanceof Error ? error : new Error('An unknown error occurred'));
      console.error('Deployment error:', error);
    }
  };

  // Update assistant step based on current deployment step
  useEffect(() => {
    if (steps.length > 0 && currentStep < steps.length) {
      const step = steps[currentStep];
      setAssistantCurrentStep(step.id);
      
      if (step.status === 'failed') {
        setDeploymentError(step.output || 'Deployment step failed');
      } else {
        setDeploymentError(null);
      }
    }
  }, [currentStep, steps]);

  const handleRunAutomation = async (action: string) => {
    await runAutomation(action, config.platform);
  };

  const getStepIcon = (step: DeploymentStep) => {
    switch (step.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400 animate-pulse" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-400" />;
      case 'running':
        return <div className="w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStepStatusColor = (step: DeploymentStep, index: number) => {
    if (index === currentStep && isDeploying) {
      return 'border-blue-500 bg-gradient-to-r from-blue-600/10 to-purple-600/10 shadow-lg';
    }
    switch (step.status) {
      case 'completed':
        return 'border-green-500 bg-gradient-to-r from-green-600/10 to-emerald-600/10';
      case 'failed':
        return 'border-red-500 bg-gradient-to-r from-red-600/10 to-pink-600/10';
      default:
        return 'border-gray-600 bg-gray-700/50';
    }
  };

  const getPlatformIcon = (platform: string) => {
    switch (platform) {
      case 'ios':
        return <Smartphone className="w-5 h-5 text-blue-400" />;
      case 'android':
        return <Smartphone className="w-5 h-5 text-green-400" />;
      case 'flutter':
        return <Tablet className="w-5 h-5 text-cyan-400" />;
      case 'desktop':
        return <Monitor className="w-5 h-5 text-purple-400" />;
      case 'web':
        return <Monitor className="w-5 h-5 text-green-400" />;
      default:
        return <Smartphone className="w-5 h-5 text-gray-400" />;
    }
  };

  const getPlatformColor = (platform: string) => {
    switch (platform) {
      case 'ios':
        return 'from-blue-500 to-cyan-500';
      case 'android':
        return 'from-green-500 to-emerald-500';
      case 'flutter':
        return 'from-cyan-500 to-blue-500';
      case 'desktop':
        return 'from-purple-500 to-pink-500';
      case 'web':
        return 'from-green-500 to-teal-500';
      default:
        return 'from-gray-500 to-slate-500';
    }
  };

  const getEstimatedTime = (platform: string) => {
    switch (platform) {
      case 'ios':
        return '8-12 minutes';
      case 'android':
        return '5-8 minutes';
      case 'web':
        return '2-4 minutes';
      case 'desktop':
        return '6-10 minutes';
      default:
        return '5-8 minutes';
    }
  };

  const getDeploymentGuide = (platform: string) => {
    switch (platform) {
      case 'ios':
        return {
          title: 'iOS Deployment Guide',
          requirements: [
            'Xcode 15+ installed',
            'iOS Developer Account',
            'Valid code signing certificate',
            'Provisioning profile configured',
            'App Store Connect access'
          ],
          steps: [
            'Build Rust code for iOS targets (device + simulator)',
            'Create universal binary with lipo',
            'Build iOS app with Xcode',
            'Create archive for distribution',
            'Export IPA with proper entitlements',
            'Upload to App Store Connect',
            'Submit for App Store review'
          ]
        };
      case 'android':
        return {
          title: 'Android Deployment Guide',
          requirements: [
            'Android Studio installed',
            'Android SDK 34+ configured',
            'NDK for native code compilation',
            'Google Play Developer Account',
            'Signing keystore configured',
            'Google Play Console access',
            'Fastlane configured (optional)'
          ],
          steps: [
            'Build Rust code for all Android architectures (ARM64, ARM, x86, x86_64)',
            'Copy native libraries to Android project',
            'Build APK or AAB with Gradle',
            'Sign with release keystore using SHA-256',
            'Align APK (if using APK format)',
            'Upload to Google Play Console',
            'Submit for Play Store review and rollout'
          ]
        };
      case 'flutter':
        return {
          title: 'Flutter Deployment Guide',
          requirements: [
            'Flutter SDK installed',
            'flutter_rust_bridge configured',
            'Platform-specific requirements (iOS/Android)',
            'Dart dependencies updated'
          ],
          steps: [
            'Generate Rust-Flutter bridge code',
            'Install Flutter dependencies',
            'Build for iOS platform',
            'Build for Android platform',
            'Deploy to iOS App Store',
            'Deploy to Google Play Store'
          ]
        };
      case 'web':
        return {
          title: 'Web Deployment Guide',
          requirements: [
            'wasm-pack installed',
            'Node.js and npm/yarn',
            'Web bundler (Vite/Webpack)',
            'Deployment target (Vercel/Netlify/AWS)',
            'CDN configuration'
          ],
          steps: [
            'Compile Rust to WebAssembly',
            'Install web dependencies',
            'Build web application (SPA/PWA/SSR)',
            'Optimize WebAssembly binaries',
            'Deploy to production hosting',
            'Configure CDN and caching policies'
          ]
        };
      default:
        return {
          title: 'Desktop Deployment Guide', 
          requirements: [
            'Rust toolchain installed',
            'Tauri CLI or Electron',
            'Platform-specific build tools',
            'Code signing certificates',
            'Distribution channels configured'
          ],
          steps: [
            'Build Rust application',
            'Build desktop UI framework',
            'Bundle with Tauri/Electron',
            'Code sign binaries for security',
            'Create platform-specific installers',
            'Upload to release channels (GitHub/Steam/etc.)'
          ]
        };
    }
  };

  if (!isVisible) return null;

  const guide = getDeploymentGuide(config.platform);

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-xl border border-gray-700 w-full max-w-6xl max-h-[95vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            <div className={`p-3 bg-gradient-to-br ${getPlatformColor(config.platform)} rounded-xl shadow-lg`}>
              {getPlatformIcon(config.platform)}
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Deploy to {config.platform.toUpperCase()}</h2>
              <p className="text-gray-400">Build and deploy your application to production</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-xl"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Real-time Metrics Bar (only during deployment) */}
        {isDeploying && (
          <div className="bg-gray-900 border-b border-gray-700 px-6 py-3">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-blue-400" />
                  <span className="text-gray-300">Build: {realTimeMetrics.buildSpeed.toFixed(1)} MB/s</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Cpu className="w-4 h-4 text-green-400" />
                  <span className="text-gray-300">CPU: {realTimeMetrics.cpuUsage.toFixed(0)}%</span>
                </div>
                <div className="flex items-center space-x-2">
                  <HardDrive className="w-4 h-4 text-purple-400" />
                  <span className="text-gray-300">Memory: {realTimeMetrics.memoryUsage.toFixed(0)}%</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Wifi className="w-4 h-4 text-orange-400" />
                  <span className="text-gray-300">Network: {realTimeMetrics.networkSpeed.toFixed(1)} Mbps</span>
                </div>
              </div>
              <div className="text-gray-400">
                Step {currentStep + 1} of {steps.length} • {getOverallProgress().toFixed(0)}% Complete
              </div>
            </div>
          </div>
        )}

        <div className="flex h-[80vh]">
          {/* Configuration Panel */}
          <div className="w-80 bg-gray-900 border-r border-gray-700 p-6 overflow-y-auto">
            <h3 className="text-lg font-semibold text-white mb-4">Deployment Configuration</h3>
            
            {/* Platform Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">Platform</label>
              <div className="grid grid-cols-2 gap-2 mb-3">
                {['ios', 'android', 'flutter'].map(platform => (
                  <button
                    key={platform}
                    onClick={() => setConfig(prev => ({ ...prev, platform: platform as any }))}
                    className={`flex items-center space-x-2 p-3 rounded-lg border transition-all transform hover:scale-105 ${
                      config.platform === platform
                        ? `border-blue-500 bg-gradient-to-r ${getPlatformColor(platform)}/20 text-white shadow-lg`
                        : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                    }`}
                  >
                    {getPlatformIcon(platform)}
                    <span className="text-sm font-medium">{platform.toUpperCase()}</span>
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                {['web', 'desktop'].map(platform => (
                  <button
                    key={platform}
                    onClick={() => setConfig(prev => ({ ...prev, platform: platform as any }))}
                    className={`flex items-center space-x-2 p-3 rounded-lg border transition-all transform hover:scale-105 ${
                      config.platform === platform
                        ? `border-blue-500 bg-gradient-to-r ${getPlatformColor(platform)}/20 text-white shadow-lg`
                        : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                    }`}
                  >
                    {getPlatformIcon(platform)}
                    <span className="text-sm font-medium">{platform.toUpperCase()}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Platform Info */}
            <div className="mb-6 p-4 bg-gray-800 rounded-lg border border-gray-600">
              <div className="flex items-center space-x-2 mb-2">
                {getPlatformIcon(config.platform)}
                <span className="font-medium text-white">{config.platform.toUpperCase()} Deployment</span>
              </div>
              <div className="text-sm text-gray-400 space-y-1">
                <div>Estimated time: {getEstimatedTime(config.platform)}</div>
                <div>Build type: {config.buildType}</div>
                {config.platform === 'android' && <div>Target: {config.androidTarget?.toUpperCase()}</div>}
                {config.platform === 'web' && <div>Target: {config.webTarget?.toUpperCase()}</div>}
                {config.platform === 'desktop' && <div>Target: {config.desktopTarget}</div>}
              </div>
            </div>

            {/* Platform-specific Options */}
            {config.platform === 'android' && (
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">Android Target</label>
                <div className="flex space-x-2">
                  {[
                    { key: 'aab', label: 'AAB (Recommended)', desc: 'App Bundle for Play Store' },
                    { key: 'apk', label: 'APK', desc: 'Direct installation' }
                  ].map(target => (
                    <button
                      key={target.key}
                      onClick={() => setConfig(prev => ({ ...prev, androidTarget: target.key as any }))}
                      className={`flex-1 p-3 rounded-lg border text-left transition-all transform hover:scale-105 ${
                        config.androidTarget === target.key
                          ? 'border-green-500 bg-green-600/20 text-green-300'
                          : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                      }`}
                    >
                      <div className="font-medium text-sm">{target.label}</div>
                      <div className="text-xs opacity-75">{target.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {config.platform === 'web' && (
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">Web Target</label>
                <div className="space-y-2">
                  {[
                    { key: 'spa', label: 'SPA', desc: 'Single Page Application' },
                    { key: 'pwa', label: 'PWA', desc: 'Progressive Web App' },
                    { key: 'ssr', label: 'SSR', desc: 'Server-Side Rendered' }
                  ].map(target => (
                    <button
                      key={target.key}
                      onClick={() => setConfig(prev => ({ ...prev, webTarget: target.key as any }))}
                      className={`w-full p-3 rounded-lg border text-left transition-all transform hover:scale-105 ${
                        config.webTarget === target.key
                          ? 'border-green-500 bg-green-600/20 text-green-300'
                          : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                      }`}
                    >
                      <div className="font-medium text-sm">{target.label}</div>
                      <div className="text-xs opacity-75">{target.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {config.platform === 'desktop' && (
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">Desktop Target</label>
                <div className="space-y-2">
                  {[
                    { key: 'all', label: 'All Platforms', desc: 'Windows, macOS, Linux' },
                    { key: 'windows', label: 'Windows', desc: 'Windows 10/11' },
                    { key: 'macos', label: 'macOS', desc: 'macOS 10.15+' },
                    { key: 'linux', label: 'Linux', desc: 'Ubuntu, Debian, etc.' }
                  ].map(target => (
                    <button
                      key={target.key}
                      onClick={() => setConfig(prev => ({ ...prev, desktopTarget: target.key as any }))}
                      className={`w-full p-3 rounded-lg border text-left transition-all transform hover:scale-105 ${
                        config.desktopTarget === target.key
                          ? 'border-purple-500 bg-purple-600/20 text-purple-300'
                          : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                      }`}
                    >
                      <div className="font-medium text-sm">{target.label}</div>
                      <div className="text-xs opacity-75">{target.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Build Type */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">Build Type</label>
              <div className="flex space-x-2">
                {['debug', 'release'].map(type => (
                  <button
                    key={type}
                    onClick={() => setConfig(prev => ({ ...prev, buildType: type as any }))}
                    className={`flex-1 p-2 rounded-lg text-sm font-medium transition-all transform hover:scale-105 ${
                      config.buildType === type
                        ? 'bg-orange-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Advanced Settings */}
            <div className="mb-6">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors"
              >
                <Settings className="w-4 h-4" />
                <span>Advanced Settings</span>
              </button>
              
              {showAdvanced && (
                <div className="mt-4 space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Target</label>
                    <input
                      type="text"
                      value={config.target || ''}
                      onChange={(e) => setConfig(prev => ({ ...prev, target: e.target.value }))}
                      placeholder="e.g., aarch64-apple-ios"
                      className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Output Path</label>
                    <input
                      type="text"
                      value={config.outputPath || ''}
                      onChange={(e) => setConfig(prev => ({ ...prev, outputPath: e.target.value }))}
                      placeholder="e.g., build/"
                      className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Requirements */}
            <div className="mb-6">
              <h4 className="text-sm font-medium text-gray-300 mb-3">Requirements</h4>
              <div className="space-y-2">
                {guide.requirements.map((req, index) => (
                  <div key={index} className="flex items-center space-x-2 text-sm">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                    <span className="text-gray-300">{req}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="space-y-3">
              {!isDeploying && !isConfiguring && !showPreDeployCheck ? (
                <div className="space-y-3">
                  <button
                  onClick={handleStartDeployment}
                  className={`w-full flex items-center justify-center space-x-2 py-3 px-4 bg-gradient-to-r ${getPlatformColor(config.platform)} hover:shadow-xl text-white rounded-lg font-medium transition-all transform hover:scale-105 shadow-lg`}
                >
                  <Play className="w-4 h-4" />
                  <span>Start Deployment</span>
                </button>
                  
                  {/* Quick Stats */}
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-gray-800 p-2 rounded text-center">
                      <div className="text-white font-medium">{getEstimatedTime(config.platform)}</div>
                      <div className="text-gray-400">Est. Time</div>
                    </div>
                    <div className="bg-gray-800 p-2 rounded text-center">
                      <div className="text-white font-medium">{guide.steps.length}</div>
                      <div className="text-gray-400">Steps</div>
                    </div>
                  </div>
                </div>
              ) : isConfiguring ? (
                <div className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-blue-600 text-white rounded-lg font-medium">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Preparing Deployment...</span>
                </div>
              ) : showPreDeployCheck ? (
                <div className="space-y-3">
                  <div className="bg-gradient-to-r from-orange-900/20 to-red-900/20 border border-orange-500/30 rounded-lg p-4 shadow-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertCircle className="w-5 h-5 text-orange-400" />
                      <span className="font-medium text-orange-300">Ready to Deploy</span>
                    </div>
                    <p className="text-sm text-orange-200 mb-3">
                      {config.platform === 'android' 
                        ? `Building ${config.androidTarget?.toUpperCase()} for Google Play Store. This will create a production-ready build.`
                        : `Building for ${config.platform.toUpperCase()} production. This process may take several minutes.`
                      }
                    </p>
                    <div className="flex space-x-2">
                      <button
                        onClick={handleConfirmDeployment}
                        className={`flex-1 flex items-center justify-center space-x-2 py-2 px-3 bg-gradient-to-r ${getPlatformColor(config.platform)} hover:shadow-lg text-white rounded-lg text-sm font-medium transition-all transform hover:scale-105`}
                      >
                        <Zap className="w-4 h-4" />
                        <span>Deploy Now</span>
                      </button>
                      <button
                        onClick={() => setShowPreDeployCheck(false)}
                        className="flex-1 py-2 px-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg text-sm font-medium transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <button
                  onClick={stopDeployment}
                  className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 hover:shadow-xl text-white rounded-lg font-medium transition-all transform hover:scale-105 shadow-lg"
                >
                  <Square className="w-4 h-4" />
                  <span>Stop Deployment</span>
                </button>
              )}
              
              <button className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors">
                <ExternalLink className="w-4 h-4" />
                <span>View Documentation</span>
              </button>
              
              {/* Visual Status Indicators */}
              {(isConfiguring || isDeploying) && (
                <div className={`mt-4 p-3 bg-gradient-to-r ${getPlatformColor(config.platform)}/20 border border-blue-500/30 rounded-lg shadow-lg`}>
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                    <span className="text-blue-300 text-sm font-medium">
                      {isConfiguring ? 'Configuring Build Environment' : 'Deployment In Progress'}
                    </span>
                  </div>
                  <div className="text-xs text-blue-200">
                    {isConfiguring 
                      ? 'Setting up build tools and dependencies...'
                      : `Building ${config.platform.toUpperCase()} application...`
                    }
                  </div>
                  
                  {/* Progress indicator */}
                  {isDeploying && (
                    <div className="mt-2">
                      <div className="w-full bg-gray-700 rounded-full h-1">
                        <div 
                          className="bg-blue-400 h-1 rounded-full transition-all duration-300"
                          style={{ width: `${getOverallProgress()}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Deployment Progress */}
          <div className="flex-1 p-6 overflow-y-auto">
            {steps.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">{config.platform === 'ios' ? '📱' : config.platform === 'android' ? '🤖' : '💻'}</div>
                <h3 className="text-xl font-semibold text-white mb-2">{guide.title}</h3>
                <p className="text-gray-400 mb-8">Configure your deployment settings and click "Start Deployment" to begin</p>
                
                <div className="max-w-2xl mx-auto text-left">
                  <h4 className="text-lg font-semibold text-white mb-4">Deployment Process</h4>
                  <div className="space-y-3">
                    {guide.steps.map((step, index) => (
                      <div key={index} className="flex items-start space-x-3">
                        <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-medium flex-shrink-0 mt-0.5">
                          {index + 1}
                        </div>
                        <span className="text-gray-300">{step}</span>
                      </div>
                    ))}
                  </div>
                  
                  {/* Platform-specific info */}
                  {config.platform === 'android' && (
                    <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                      <h5 className="font-medium text-green-300 mb-2">🤖 Android Deployment Tips</h5>
                      <ul className="text-sm text-green-200 space-y-1">
                        <li>• AAB format is recommended for Play Store (smaller downloads)</li>
                        <li>• Ensure all architectures are built for maximum compatibility</li>
                        <li>• Use SHA-256 signing for enhanced security</li>
                        <li>• Test on internal track before production release</li>
                        <li>• Build process typically takes 3-5 minutes for Android</li>
                        <li>• Upload to Play Console happens automatically after build</li>
                      </ul>
                    </div>
                  )}
                  
                  {config.platform === 'web' && (
                    <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                      <h5 className="font-medium text-green-300 mb-2">🌐 Web Deployment Tips</h5>
                      <ul className="text-sm text-green-200 space-y-1">
                        <li>• WebAssembly provides near-native performance</li>
                        <li>• PWA enables offline functionality and app-like experience</li>
                        <li>• CDN configuration is crucial for global performance</li>
                        <li>• Consider SSR for better SEO and initial load times</li>
                      </ul>
                    </div>
                  )}
                  
                  {config.platform === 'desktop' && (
                    <div className="mt-6 p-4 bg-purple-900/20 border border-purple-500/30 rounded-lg">
                      <h5 className="font-medium text-purple-300 mb-2">💻 Desktop Deployment Tips</h5>
                      <ul className="text-sm text-purple-200 space-y-1">
                        <li>• Tauri provides smaller bundle sizes than Electron</li>
                        <li>• Code signing is required for macOS and recommended for Windows</li>
                        <li>• Auto-updater can be configured for seamless updates</li>
                        <li>• Consider distribution via GitHub Releases, Steam, or app stores</li>
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Deployment Progress</h3>
                  <div className="text-sm text-gray-400">
                    Step {currentStep + 1} of {steps.length} • {getOverallProgress().toFixed(0)}% Complete
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-gray-700 rounded-full h-3 shadow-inner">
                  <div 
                    className={`bg-gradient-to-r ${getPlatformColor(config.platform)} h-3 rounded-full transition-all duration-500 shadow-lg`}
                    style={{ width: `${getOverallProgress()}%` }}
                  />
                </div>

                {/* Steps */}
                <div className="space-y-4">
                  {steps.map((step, index) => (
                    <div 
                      key={step.id} 
                      className={`p-4 rounded-lg border transition-all duration-300 ${getStepStatusColor(step, index)}`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-3">
                          {getStepIcon(step)}
                          <span className="font-medium text-white">{step.name}</span>
                          {index === currentStep && isDeploying && (
                            <div className="flex items-center space-x-2">
                              <div className="flex items-center space-x-1">
                                <div className="w-1 h-1 bg-blue-400 rounded-full animate-ping"></div>
                                <div className="w-1 h-1 bg-blue-400 rounded-full animate-ping" style={{ animationDelay: '0.2s' }}></div>
                                <div className="w-1 h-1 bg-blue-400 rounded-full animate-ping" style={{ animationDelay: '0.4s' }}></div>
                              </div>
                              <span className="text-xs text-blue-300 font-medium">
                                {getStepProgress(index).toFixed(0)}%
                              </span>
                            </div>
                          )}
                        </div>
                        <div className="flex items-center space-x-3">
                          {step.metadata?.estimatedTime && index >= currentStep && (
                            <span className="text-xs text-gray-500">
                              ~{step.metadata.estimatedTime}
                            </span>
                          )}
                          {step.duration && (
                            <span className="text-sm text-gray-400">{(step.duration / 1000).toFixed(1)}s</span>
                          )}
                        </div>
                      </div>
                      
                      {/* Progress indicator for current step */}
                      {index === currentStep && isDeploying && (
                        <div className="mb-3 space-y-2">
                          <div className="w-full bg-gray-700 rounded-full h-2 shadow-inner">
                            <div 
                              className={`bg-gradient-to-r ${getPlatformColor(config.platform)} h-2 rounded-full transition-all duration-300 shadow-sm`}
                              style={{ width: `${getStepProgress(index)}%` }}
                            />
                          </div>
                          {step.substeps && step.currentSubstep !== undefined && (
                            <div className="text-xs text-gray-400">
                              {step.substeps[step.currentSubstep] || 'Processing...'}
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Step metadata */}
                      {step.metadata && (index === currentStep || step.status === 'completed') && (
                        <div className="mb-3 grid grid-cols-2 gap-2 text-xs">
                          {step.metadata.outputSize && (
                            <div className="bg-gray-800 p-2 rounded">
                              <div className="text-gray-400">Output Size</div>
                              <div className="text-white font-medium">{step.metadata.outputSize}</div>
                            </div>
                          )}
                          {step.metadata.estimatedTime && (
                            <div className="bg-gray-800 p-2 rounded">
                              <div className="text-gray-400">Est. Time</div>
                              <div className="text-white font-medium">{step.metadata.estimatedTime}</div>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {step.command && (
                        <div className="mb-2">
                          <div className="flex items-center space-x-2 text-sm text-gray-400 mb-1">
                            <Terminal className="w-3 h-3" />
                            <span>Command:</span>
                          </div>
                          <code className="block text-sm bg-gray-800 p-2 rounded font-mono text-gray-300 overflow-x-auto">
                            {step.command}
                          </code>
                        </div>
                      )}
                      
                      {step.output && (
                        <div>
                          <div className="flex items-center space-x-2 text-sm text-gray-400 mb-1">
                            <Terminal className="w-3 h-3" />
                            <span>Output:</span>
                          </div>
                          <pre className="text-sm bg-gray-800 p-3 rounded font-mono text-gray-300 overflow-x-auto whitespace-pre-wrap">
                            {step.output}
                          </pre>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Completion Actions */}
                {!isDeploying && steps.every(step => step.status === 'completed') && (
                  <div className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 border border-green-500 rounded-lg p-6 text-center shadow-xl">
                    <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">Deployment Successful!</h3>
                    <p className="text-gray-300 mb-6">
                      Your {config.platform === 'android' ? config.androidTarget?.toUpperCase() : 'app'} has been successfully deployed to {config.platform.toUpperCase()}
                      {config.platform === 'android' && ' Play Console'}
                    </p>
                    
                    {/* Platform-specific success info */}
                    {config.platform === 'android' && (
                      <div className="bg-gradient-to-r from-green-800/30 to-emerald-800/30 rounded-lg p-4 mb-6 border border-green-600/30">
                        <div className="flex items-center justify-center space-x-2 mb-2">
                          <Package className="w-5 h-5 text-green-400" />
                          <span className="font-medium text-green-300">Android Build Complete</span>
                        </div>
                        <div className="text-sm text-green-200 space-y-1">
                          <div>✓ {config.androidTarget?.toUpperCase()} signed with release keystore</div>
                          <div>✓ Uploaded to Google Play Console</div>
                          <div>✓ Available on internal testing track</div>
                          <div className="text-yellow-200 mt-2">
                            ⏳ Review process: 1-3 days for production release
                          </div>
                        </div>
                      </div>
                    )}

                    {config.platform === 'ios' && (
                      <div className="bg-gradient-to-r from-blue-800/30 to-cyan-800/30 rounded-lg p-4 mb-6 border border-blue-600/30">
                        <div className="flex items-center justify-center space-x-2 mb-2">
                          <Smartphone className="w-5 h-5 text-blue-400" />
                          <span className="font-medium text-blue-300">iOS Build Complete</span>
                        </div>
                        <div className="text-sm text-blue-200 space-y-1">
                          <div>✓ IPA signed with distribution certificate</div>
                          <div>✓ Uploaded to App Store Connect</div>
                          <div>✓ Processing for App Store review</div>
                          <div className="text-yellow-200 mt-2">
                            ⏳ Review process: 24-48 hours for App Store approval
                          </div>
                        </div>
                      </div>
                    )}

                    {config.platform === 'web' && (
                      <div className="bg-gradient-to-r from-green-800/30 to-teal-800/30 rounded-lg p-4 mb-6 border border-green-600/30">
                        <div className="flex items-center justify-center space-x-2 mb-2">
                          <Globe className="w-5 h-5 text-green-400" />
                          <span className="font-medium text-green-300">Web Deployment Complete</span>
                        </div>
                        <div className="text-sm text-green-200 space-y-1">
                          <div>✓ {config.webTarget?.toUpperCase()} built and optimized</div>
                          <div>✓ WebAssembly optimized for performance</div>
                          <div>✓ Deployed to global CDN</div>
                          <div className="text-blue-200 mt-2">
                            🌐 Live in 50+ edge locations worldwide
                          </div>
                        </div>
                      </div>
                    )}

                    {config.platform === 'desktop' && (
                      <div className="bg-gradient-to-r from-purple-800/30 to-pink-800/30 rounded-lg p-4 mb-6 border border-purple-600/30">
                        <div className="flex items-center justify-center space-x-2 mb-2">
                          <Monitor className="w-5 h-5 text-purple-400" />
                          <span className="font-medium text-purple-300">Desktop Build Complete</span>
                        </div>
                        <div className="text-sm text-purple-200 space-y-1">
                          <div>✓ Native binaries created for {config.desktopTarget}</div>
                          <div>✓ Installers generated and signed</div>
                          <div>✓ Ready for distribution</div>
                          <div className="text-blue-200 mt-2">
                            📦 Available for download and installation
                          </div>
                        </div>
                      </div>
                    )}
                    
                    <div className="flex justify-center space-x-4">
                      {config.platform === 'web' ? (
                        <>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:shadow-lg text-white rounded-lg transition-all transform hover:scale-105">
                            <ExternalLink className="w-4 h-4" />
                            <span>View Live Site</span>
                          </button>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:shadow-lg text-white rounded-lg transition-all transform hover:scale-105">
                            <Download className="w-4 h-4" />
                            <span>Download Assets</span>
                          </button>
                        </>
                      ) : config.platform === 'desktop' ? (
                        <>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:shadow-lg text-white rounded-lg transition-all transform hover:scale-105">
                            <Download className="w-4 h-4" />
                            <span>Download Installers</span>
                          </button>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:shadow-lg text-white rounded-lg transition-all transform hover:scale-105">
                            <ExternalLink className="w-4 h-4" />
                            <span>View Releases</span>
                          </button>
                        </>
                      ) : (
                        <>
                          <button className={`flex items-center space-x-2 px-4 py-2 bg-gradient-to-r ${getPlatformColor(config.platform)} hover:shadow-lg text-white rounded-lg transition-all transform hover:scale-105`}>
                            <Download className="w-4 h-4" />
                            <span>Download {config.platform === 'android' ? config.androidTarget?.toUpperCase() : 'Build'}</span>
                          </button>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:shadow-lg text-white rounded-lg transition-all transform hover:scale-105">
                            <Upload className="w-4 h-4" />
                            <span>View in {config.platform === 'android' ? 'Play Console' : 'App Store'}</span>
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Failed deployment feedback */}
                {!isDeploying && steps.some(step => step.status === 'failed') && (
                  <div className="bg-gradient-to-r from-red-900/20 to-pink-900/20 border border-red-500 rounded-lg p-6 text-center shadow-xl">
                    <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">Deployment Failed</h3>
                    <p className="text-gray-300 mb-6">
                      The deployment process encountered an error. Check the logs above for details.
                    </p>
                    <div className="flex justify-center space-x-4">
                      <button 
                        onClick={() => handleStartDeployment()}
                        className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-orange-600 to-red-600 hover:shadow-lg text-white rounded-lg transition-all transform hover:scale-105"
                      >
                        <RefreshCw className="w-4 h-4" />
                        <span>Retry Deployment</span>
                      </button>
                      <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-gray-600 to-gray-700 hover:shadow-lg text-white rounded-lg transition-all transform hover:scale-105">
                        <ExternalLink className="w-4 h-4" />
                        <span>Get Help</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Deployment Assistant */}
      <DeploymentAssistant
        currentStep={assistantCurrentStep}
        platform={config.platform}
        isVisible={showAssistant}
        onClose={() => setShowAssistant(false)}
        deploymentStatus={
          isDeploying ? 'deploying' :
          isConfiguring ? 'configuring' :
          steps.some(s => s.status === 'failed') ? 'error' :
          steps.every(s => s.status === 'completed') && steps.length > 0 ? 'success' :
          'idle'
        }
        currentError={deploymentError}
      />
    </div>
  );
};

export default DeploymentModal;