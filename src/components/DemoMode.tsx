import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Zap, Code, Terminal, Eye } from 'lucide-react';

interface DemoModeProps {
  isActive: boolean;
  onToggle: () => void;
}

const DemoMode: React.FC<DemoModeProps> = ({ isActive, onToggle }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  const demoSteps = [
    {
      title: "Welcome to Rust Cloud IDE",
      description: "Let's take a tour of the powerful features available in this development environment.",
      action: "intro",
      duration: 3000,
    },
    {
      title: "File Explorer",
      description: "Browse through your project structure. Click on any file to open it in the editor.",
      action: "highlight-explorer",
      duration: 4000,
    },
    {
      title: "Code Editor",
      description: "Full-featured code editor with syntax highlighting for Rust, Dart, Kotlin, and more.",
      action: "highlight-editor",
      duration: 4000,
    },
    {
      title: "Multi-Platform Support",
      description: "Build for Android, iOS, Flutter, React Native, and desktop from a single codebase.",
      action: "highlight-platforms",
      duration: 4000,
    },
    {
      title: "Integrated Terminal",
      description: "Run cargo commands, build your project, and deploy to mobile platforms.",
      action: "show-terminal",
      duration: 4000,
    },
    {
      title: "Real-time Compilation",
      description: "See your code compile in real-time with instant feedback and error reporting.",
      action: "demo-compilation",
      duration: 5000,
    },
    {
      title: "Project Templates",
      description: "Start with pre-built templates for games, IoT, blockchain, and more.",
      action: "show-templates",
      duration: 4000,
    },
    {
      title: "Ready to Code!",
      description: "You're all set! Choose a template or start coding from scratch.",
      action: "complete",
      duration: 3000,
    },
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isPlaying && isActive) {
      interval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + (100 / (demoSteps[currentStep].duration / 100));
          
          if (newProgress >= 100) {
            if (currentStep < demoSteps.length - 1) {
              setCurrentStep(prev => prev + 1);
              return 0;
            } else {
              setIsPlaying(false);
              return 100;
            }
          }
          
          return newProgress;
        });
      }, 100);
    }
    
    return () => clearInterval(interval);
  }, [isPlaying, isActive, currentStep, demoSteps]);

  const handlePlay = () => {
    setIsPlaying(true);
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setProgress(0);
  };

  const handleStepClick = (stepIndex: number) => {
    setCurrentStep(stepIndex);
    setProgress(0);
    setIsPlaying(false);
  };

  if (!isActive) {
    return (
      <button
        onClick={onToggle}
        className="fixed bottom-6 right-6 bg-orange-600 hover:bg-orange-700 text-white p-4 rounded-full shadow-lg transition-colors z-50"
        title="Start Demo"
      >
        <Play className="w-6 h-6" />
      </button>
    );
  }

  const currentStepData = demoSteps[currentStep];

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-lg max-w-2xl w-full p-6 border border-gray-700">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-orange-600 rounded-lg">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Interactive Demo</h2>
              <p className="text-gray-400 text-sm">Step {currentStep + 1} of {demoSteps.length}</p>
            </div>
          </div>
          <button
            onClick={onToggle}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm text-gray-400 mb-2">
            <span>Progress</span>
            <span>{Math.round(((currentStep + progress / 100) / demoSteps.length) * 100)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-orange-600 h-2 rounded-full transition-all duration-100"
              style={{ 
                width: `${((currentStep + progress / 100) / demoSteps.length) * 100}%` 
              }}
            />
          </div>
        </div>

        {/* Current Step */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-white mb-2">
            {currentStepData.title}
          </h3>
          <p className="text-gray-300 leading-relaxed">
            {currentStepData.description}
          </p>
        </div>

        {/* Step Progress */}
        <div className="mb-6">
          <div className="w-full bg-gray-700 rounded-full h-1">
            <div 
              className="bg-orange-400 h-1 rounded-full transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {!isPlaying ? (
              <button
                onClick={handlePlay}
                className="flex items-center space-x-2 bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded-lg transition-colors"
              >
                <Play className="w-4 h-4" />
                <span>Play</span>
              </button>
            ) : (
              <button
                onClick={handlePause}
                className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
              >
                <Pause className="w-4 h-4" />
                <span>Pause</span>
              </button>
            )}
            
            <button
              onClick={handleReset}
              className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>
          </div>

          <div className="flex items-center space-x-1">
            {demoSteps.map((_, index) => (
              <button
                key={index}
                onClick={() => handleStepClick(index)}
                className={`w-3 h-3 rounded-full transition-colors ${
                  index === currentStep
                    ? 'bg-orange-600'
                    : index < currentStep
                    ? 'bg-orange-400'
                    : 'bg-gray-600'
                }`}
              />
            ))}
          </div>
        </div>

        {/* Demo Actions */}
        {currentStepData.action === 'show-templates' && (
          <div className="mt-6 p-4 bg-gray-700 rounded-lg">
            <div className="flex items-center space-x-2 mb-3">
              <Code className="w-5 h-5 text-orange-400" />
              <span className="font-medium text-white">Available Templates</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-300">🎮 Mobile Game Engine</div>
              <div className="text-gray-300">🏠 IoT Dashboard</div>
              <div className="text-gray-300">💰 Crypto Wallet</div>
              <div className="text-gray-300">🌐 Web Scraper</div>
              <div className="text-gray-300">💬 Chat Application</div>
              <div className="text-gray-300">📱 React Native App</div>
            </div>
          </div>
        )}

        {currentStepData.action === 'demo-compilation' && (
          <div className="mt-6 p-4 bg-gray-900 rounded-lg font-mono text-sm">
            <div className="flex items-center space-x-2 mb-2">
              <Terminal className="w-4 h-4 text-green-400" />
              <span className="text-green-400">$ cargo build --release</span>
            </div>
            <div className="text-gray-300 space-y-1">
              <div>   Compiling mobile-rust-app v0.1.0</div>
              <div className="text-green-400">    Finished release [optimized] target(s) in 2.34s</div>
            </div>
          </div>
        )}

        {currentStepData.action === 'complete' && (
          <div className="mt-6 text-center">
            <div className="text-4xl mb-2">🎉</div>
            <p className="text-gray-300">
              Ready to start building amazing mobile applications with Rust!
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DemoMode;