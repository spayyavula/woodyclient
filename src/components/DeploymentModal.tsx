import React, { useState } from 'react';
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
  Upload
} from 'lucide-react';
import { useDeployment, DeploymentConfig, DeploymentStep } from '../hooks/useDeployment';

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
    stopDeployment 
  } = useDeployment();

  const [config, setConfig] = useState<DeploymentConfig>({
    platform: initialPlatform,
    buildType: 'release',
    webTarget: 'spa',
    desktopTarget: 'all',
    androidTarget: 'aab'
  });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleStartDeployment = () => {
    startDeployment(config);
  };

  const getStepIcon = (step: DeploymentStep) => {
    switch (step.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-400" />;
      case 'running':
        return <div className="w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
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
            <div className="p-2 bg-blue-600 rounded-lg">
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
                    className={`flex items-center space-x-2 p-3 rounded-lg border transition-colors ${
                      config.platform === platform
                        ? 'border-blue-500 bg-blue-600/20 text-blue-300'
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
                    className={`flex items-center space-x-2 p-3 rounded-lg border transition-colors ${
                      config.platform === platform
                        ? 'border-blue-500 bg-blue-600/20 text-blue-300'
                        : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                    }`}
                  >
                    {getPlatformIcon(platform)}
                    <span className="text-sm font-medium">{platform.toUpperCase()}</span>
                  </button>
                ))}
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
                      className={`flex-1 p-3 rounded-lg border text-left transition-colors ${
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
                      className={`w-full p-3 rounded-lg border text-left transition-colors ${
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
                      className={`w-full p-3 rounded-lg border text-left transition-colors ${
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
                    className={`flex-1 p-2 rounded-lg text-sm font-medium transition-colors ${
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
              {!isDeploying ? (
                <button
                  onClick={handleStartDeployment}
                  className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors"
                >
                  <Play className="w-4 h-4" />
                  <span>Start Deployment</span>
                </button>
              ) : (
                <button
                  onClick={stopDeployment}
                  className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors"
                >
                  <Square className="w-4 h-4" />
                  <span>Stop Deployment</span>
                </button>
              )}
              
              <button className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors">
                <ExternalLink className="w-4 h-4" />
                <span>View Documentation</span>
              </button>
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
                    Step {currentStep + 1} of {steps.length}
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                  />
                </div>

                {/* Steps */}
                <div className="space-y-4">
                  {steps.map((step, index) => (
                    <div 
                      key={step.id} 
                      className={`p-4 rounded-lg border transition-colors ${
                        index === currentStep && isDeploying
                          ? 'border-blue-500 bg-blue-600/10'
                          : step.status === 'completed'
                          ? 'border-green-500 bg-green-600/10'
                          : step.status === 'failed'
                          ? 'border-red-500 bg-red-600/10'
                          : 'border-gray-600 bg-gray-700/50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-3">
                          {getStepIcon(step)}
                          <span className="font-medium text-white">{step.name}</span>
                        </div>
                        {step.duration && (
                          <span className="text-sm text-gray-400">{(step.duration / 1000).toFixed(1)}s</span>
                        )}
                      </div>
                      
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
                  <div className="bg-green-900/20 border border-green-500 rounded-lg p-6 text-center">
                    <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">Deployment Successful!</h3>
                    <p className="text-gray-300 mb-6">Your app has been successfully deployed to {config.platform.toUpperCase()}</p>
                    <div className="flex justify-center space-x-4">
                      {config.platform === 'web' ? (
                        <>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                            <ExternalLink className="w-4 h-4" />
                            <span>View Live Site</span>
                          </button>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors">
                            <Download className="w-4 h-4" />
                            <span>Download Assets</span>
                          </button>
                        </>
                      ) : config.platform === 'desktop' ? (
                        <>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                            <Download className="w-4 h-4" />
                            <span>Download Installers</span>
                          </button>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors">
                            <ExternalLink className="w-4 h-4" />
                            <span>View Releases</span>
                          </button>
                        </>
                      ) : (
                        <>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                            <Download className="w-4 h-4" />
                            <span>Download Build</span>
                          </button>
                          <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors">
                            <Upload className="w-4 h-4" />
                            <span>View in Store</span>
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeploymentModal;