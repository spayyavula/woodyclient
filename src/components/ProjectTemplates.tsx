import React, { useState } from 'react';
import { 
  Code, 
  Smartphone, 
  Zap, 
  Shield, 
  Users, 
  Star, 
  ArrowRight, 
  Play, 
  CheckCircle, 
  Github, 
  Twitter, 
  Mail,
  Eye,
  EyeOff,
  Loader2,
  Rocket,
  Globe,
  Database,
  Cloud,
  Terminal,
  Cpu,
  Lock,
  Heart,
  Award,
  TrendingUp,
  Coffee,
  Monitor,
  AlertTriangle,
  GitBranch,
  Workflow,
  X
} from 'lucide-react';

interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
  tags: string[];
  icon: React.ReactNode;
  estimatedTime: string;
  files: Record<string, { content: string; language: string }>;
  features: string[];
  useCase: string;
  techStack: string[];
}

interface ProjectTemplatesProps {
  isVisible: boolean;
  onClose: () => void;
  onSelectTemplate: (template: Template) => void;
}

const ProjectTemplates: React.FC<ProjectTemplatesProps> = ({ isVisible, onClose, onSelectTemplate }) => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);

  const categories = [
    { id: 'all', name: 'All Templates', icon: <Code className="w-4 h-4" /> },
    { id: 'mobile', name: 'Mobile Apps', icon: <Smartphone className="w-4 h-4" /> },
    { id: 'web', name: 'Web Apps', icon: <Globe className="w-4 h-4" /> },
    { id: 'ai', name: 'AI/ML', icon: <Cpu className="w-4 h-4" /> },
    { id: 'security', name: 'Security', icon: <Shield className="w-4 h-4" /> },
    { id: 'deployment', name: 'Deployment', icon: <Rocket className="w-4 h-4" /> }
  ];

  const templates: Template[] = [
    {
      id: 'android-hello-world',
      name: 'Android Hello World',
      description: 'A simple Android Hello World app with complete deployment pipeline to Google Play Store',
      category: 'deployment',
      difficulty: 'Beginner',
      tags: ['Android', 'Kotlin', 'Deployment', 'Google Play'],
      icon: <Smartphone className="w-8 h-8 text-green-400" />,
      estimatedTime: '30 minutes',
      files: {
        'MainActivity.kt': {
          content: `package com.rustyclint.helloworld

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Find the TextView and set a dynamic message
        val messageTextView = findViewById<TextView>(R.id.messageTextView)
        messageTextView.text = "Hello World from rustyclint!"
    }
}`,
          language: 'kotlin'
        },
        'activity_main.xml': {
          content: `<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout 
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:background="#1F2937"
    tools:context=".MainActivity">

    <TextView
        android:id="@+id/titleTextView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Hello World!"
        android:textColor="#FFFFFF"
        android:textSize="32sp"
        android:textStyle="bold"
        app:layout_constraintBottom_toTopOf="@+id/messageTextView"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_chainStyle="packed" />

    <TextView
        android:id="@+id/messageTextView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginTop="16dp"
        android:text="Welcome to your first Android app!"
        android:textColor="#E5E7EB"
        android:textSize="18sp"
        app:layout_constraintBottom_toTopOf="@+id/versionTextView"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@+id/titleTextView" />

    <TextView
        android:id="@+id/versionTextView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginTop="32dp"
        android:text="Version 1.0.0"
        android:textColor="#9CA3AF"
        android:textSize="14sp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@+id/messageTextView" />

</androidx.constraintlayout.widget.ConstraintLayout>`,
          language: 'xml'
        },
        'build.gradle': {
          content: `plugins {
    id 'com.android.application'
    id 'kotlin-android'
}

// Load keystore properties
def keystorePropertiesFile = rootProject.file("keystore.properties")
def keystoreProperties = new Properties()
if (keystorePropertiesFile.exists()) {
    keystoreProperties.load(new FileInputStream(keystorePropertiesFile))
}

android {
    namespace 'com.rustyclint.helloworld'
    compileSdk 34

    defaultConfig {
        applicationId "com.rustyclint.helloworld"
        minSdk 21
        targetSdk 34
        versionCode 1
        versionName "1.0.0"

        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        release {
            keyAlias keystoreProperties['keyAlias'] ?: System.getenv("KEY_ALIAS") ?: "release-key"
            keyPassword keystoreProperties['keyPassword'] ?: System.getenv("KEY_PASSWORD")
            storeFile keystoreProperties['storeFile'] ? file(keystoreProperties['storeFile']) : null
            storePassword keystoreProperties['storePassword'] ?: System.getenv("KEYSTORE_PASSWORD")
        }
    }

    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }

    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = '1.8'
    }

    buildFeatures {
        viewBinding true
    }
}

dependencies {
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.11.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
}`,
          language: 'gradle'
        },
        'keystore.properties': {
          content: `storePassword=android123
keyPassword=android123
keyAlias=release-key
storeFile=keystore/release.keystore`,
          language: 'properties'
        },
        'deploy-android.sh': {
          content: `#!/bin/bash

# Android Deployment Script
set -e

echo "🤖 Starting Android AAB deployment..."

# Check if keystore exists
if [ ! -f "android/keystore/release.keystore" ]; then
    echo "❌ Error: Keystore not found. Run generate-android-keystore.sh first"
    exit 1
fi

# Load environment variables if .env.android exists
if [ -f ".env.android" ]; then
    export $(cat .env.android | xargs)
    echo "✅ Loaded environment variables from .env.android"
fi

# Build Rust code for Android
echo "📦 Building Rust code for Android targets..."
rustup target add aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android

cargo build --target aarch64-linux-android --release
cargo build --target armv7-linux-androideabi --release  
cargo build --target i686-linux-android --release
cargo build --target x86_64-linux-android --release

# Copy native libraries
echo "📋 Copying native libraries..."
mkdir -p android/app/src/main/jniLibs/{arm64-v8a,armeabi-v7a,x86,x86_64}

cp target/aarch64-linux-android/release/librustyclint.so android/app/src/main/jniLibs/arm64-v8a/
cp target/armv7-linux-androideabi/release/librustyclint.so android/app/src/main/jniLibs/armeabi-v7a/
cp target/i686-linux-android/release/librustyclint.so android/app/src/main/jniLibs/x86/
cp target/x86_64-linux-android/release/librustyclint.so android/app/src/main/jniLibs/x86_64/

# Build signed AAB
echo "🔨 Building signed AAB..."
cd android
./gradlew bundleRelease

# Check if AAB was created
AAB_PATH="app/build/outputs/bundle/release/app-release.aab"
if [ -f "$AAB_PATH" ]; then
    echo "✅ AAB created successfully: $AAB_PATH"
    echo "📊 AAB size: $(du -h "$AAB_PATH" | cut -f1)"
else
    echo "❌ Error: AAB not found at $AAB_PATH"
    exit 1
fi

# Optional: Upload to Play Store using Fastlane
if command -v fastlane &> /dev/null; then
    echo "🚀 Uploading to Google Play Store..."
    fastlane deploy
else
    echo "⚠️  Fastlane not found. Install with: gem install fastlane"
    echo "📱 Manual upload required to Google Play Console"
fi

echo "🎉 Android deployment completed!"`,
          language: 'bash'
        }
      },
      features: [
        'Complete Android app with Kotlin',
        'Automated keystore generation',
        'Secure signing configuration',
        'Google Play Store deployment pipeline',
        'CI/CD integration with GitHub Actions',
        'Step-by-step deployment guide'
      ],
      useCase: 'Perfect for developers who want to learn the complete Android deployment process to Google Play Store.',
      techStack: ['Android', 'Kotlin', 'Gradle', 'GitHub Actions', 'Fastlane']
    },
    {
      id: 'android-deployment-pipeline',
      name: 'Android Deployment Pipeline',
      description: 'Complete deployment pipeline for Android apps with visual progress tracking and automation',
      category: 'deployment',
      difficulty: 'Intermediate',
      tags: ['Android', 'CI/CD', 'Deployment', 'Automation'],
      icon: <Rocket className="w-8 h-8 text-purple-400" />,
      estimatedTime: '45 minutes',
      files: {
        'DeploymentAssistant.tsx': {
          content: `import React, { useState, useEffect } from 'react';
import { 
  HelpCircle, 
  CheckCircle, 
  AlertTriangle, 
  Info, 
  ExternalLink, 
  Copy, 
  Eye, 
  EyeOff,
  Download,
  Upload,
  Settings,
  Key,
  Shield,
  Zap,
  Terminal,
  FileText,
  Clock,
  Smartphone,
  Monitor
} from 'lucide-react';

interface DeploymentAssistantProps {
  currentStep: string;
  platform: string;
  isVisible: boolean;
  onClose: () => void;
  deploymentStatus: 'idle' | 'configuring' | 'building' | 'deploying' | 'success' | 'error';
  currentError?: string;
}

interface HelpContent {
  title: string;
  description: string;
  steps: string[];
  tips: string[];
  troubleshooting: { issue: string; solution: string }[];
  requirements: string[];
  estimatedTime: string;
  automation?: {
    available: boolean;
    description: string;
    action: string;
  };
}

const DeploymentAssistant: React.FC<DeploymentAssistantProps> = ({
  currentStep,
  platform,
  isVisible,
  onClose,
  deploymentStatus,
  currentError
}) => {
  const [activeTab, setActiveTab] = useState<'help' | 'automation' | 'troubleshooting' | 'security'>('help');
  const [showSecrets, setShowSecrets] = useState(false);
  const [automationProgress, setAutomationProgress] = useState(0);

  const getHelpContent = (): HelpContent => {
    const stepKey = \`\${platform}-\${currentStep}\`;
    
    const helpDatabase: Record<string, HelpContent> = {
      'android-rust-build': {
        title: 'Building Rust Code for Android',
        description: 'Compiling Rust code for all Android architectures (ARM64, ARM, x86, x86_64)',
        steps: [
          'Install Android NDK and configure environment',
          'Add Android targets to Rust toolchain',
          'Set up cross-compilation environment',
          'Build for each target architecture',
          'Copy libraries to Android project'
        ],
        tips: [
          'Use cargo-ndk for easier cross-compilation',
          'Enable LTO for smaller binary sizes',
          'Use strip to reduce library size',
          'Test on different architectures'
        ],
        troubleshooting: [
          {
            issue: 'NDK not found error',
            solution: 'Set ANDROID_NDK_ROOT environment variable to NDK path'
          },
          {
            issue: 'Linker errors',
            solution: 'Ensure correct target triple and NDK version compatibility'
          },
          {
            issue: 'Large binary size',
            solution: 'Enable LTO and use release profile optimizations'
          }
        ],
        requirements: [
          'Android NDK r25c or later',
          'Rust toolchain with Android targets',
          'Properly configured environment variables'
        ],
        estimatedTime: '3-5 minutes',
        automation: {
          available: true,
          description: 'Automatically install NDK, add targets, and build for all architectures',
          action: 'auto-build-rust'
        }
      },
      'android-gradle-build': {
        title: 'Building Android AAB/APK',
        description: 'Creating signed Android App Bundle or APK for Google Play Store',
        steps: [
          'Configure signing in build.gradle',
          'Set up keystore properties',
          'Run Gradle build task',
          'Verify signed output',
          'Check bundle/APK integrity'
        ],
        tips: [
          'Use AAB format for Play Store (smaller downloads)',
          'Enable R8 code shrinking for smaller size',
          'Test on multiple devices before release',
          'Use build cache to speed up builds'
        ],
        troubleshooting: [
          {
            issue: 'Keystore not found',
            solution: 'Ensure keystore path is correct in keystore.properties'
          },
          {
            issue: 'Signing failed',
            solution: 'Check keystore and key passwords are correct'
          },
          {
            issue: 'Build timeout',
            solution: 'Increase Gradle daemon heap size and enable parallel builds'
          }
        ],
        requirements: [
          'Valid Android signing keystore',
          'Configured keystore.properties',
          'Android SDK and build tools'
        ],
        estimatedTime: '2-4 minutes',
        automation: {
          available: true,
          description: 'Automatically configure signing and build signed AAB',
          action: 'auto-build-android'
        }
      },
      'android-upload-playstore': {
        title: 'Uploading to Google Play Store',
        description: 'Publishing your app to Google Play Console for distribution',
        steps: [
          'Set up Google Play Console API access',
          'Configure service account credentials',
          'Upload AAB to Play Console',
          'Set release track (internal/alpha/beta/production)',
          'Submit for review'
        ],
        tips: [
          'Start with internal testing track',
          'Use staged rollouts for production',
          'Prepare store listing before upload',
          'Test thoroughly on internal track first'
        ],
        troubleshooting: [
          {
            issue: 'API access denied',
            solution: 'Ensure service account has proper permissions in Play Console'
          },
          {
            issue: 'Upload failed',
            solution: 'Check AAB is properly signed and meets Play Store requirements'
          },
          {
            issue: 'Version code conflict',
            solution: 'Increment version code in build.gradle'
          }
        ],
        requirements: [
          'Google Play Console account',
          'Service account with API access',
          'Signed AAB file',
          'App listing configured'
        ],
        estimatedTime: '5-10 minutes',
        automation: {
          available: true,
          description: 'Automatically upload to Play Console using Fastlane',
          action: 'auto-upload-playstore'
        }
      }
    };

    return helpDatabase[stepKey] || {
      title: 'Deployment Step',
      description: 'General deployment guidance',
      steps: ['Follow the deployment process'],
      tips: ['Check logs for detailed information'],
      troubleshooting: [],
      requirements: [],
      estimatedTime: 'Variable',
      automation: { available: false, description: '', action: '' }
    };
  };

  const handleAutomation = async (action: string) => {
    setAutomationProgress(0);
    
    // Simulate automation progress
    const interval = setInterval(() => {
      setAutomationProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 10;
      });
    }, 500);

    // Here you would trigger the actual automation
    console.log(\`Triggering automation: \${action}\`);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const helpContent = getHelpContent();

  if (!isVisible) return null;

  return (
    <div className="fixed right-4 top-20 bottom-4 w-96 bg-gray-800 border border-gray-700 rounded-lg shadow-2xl z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <HelpCircle className="w-5 h-5 text-blue-400" />
          <h3 className="font-semibold text-white">Deployment Assistant</h3>
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors"
        >
          ✕
        </button>
      </div>

      {/* Status Indicator */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            {platform === 'android' && <Smartphone className="w-4 h-4 text-green-400" />}
            {platform === 'ios' && <Smartphone className="w-4 h-4 text-blue-400" />}
            {platform === 'web' && <Monitor className="w-4 h-4 text-purple-400" />}
            <span className="text-sm font-medium text-gray-300">{platform.toUpperCase()}</span>
          </div>
          <div className="flex-1">
            <div className={\`text-xs px-2 py-1 rounded \${
              deploymentStatus === 'success' ? 'bg-green-900/30 text-green-300' :
              deploymentStatus === 'error' ? 'bg-red-900/30 text-red-300' :
              deploymentStatus === 'building' || deploymentStatus === 'deploying' ? 'bg-blue-900/30 text-blue-300' :
              'bg-gray-900/30 text-gray-300'
            }\`}>
              {deploymentStatus.toUpperCase()}
            </div>
          </div>
        </div>
        {currentError && (
          <div className="mt-2 p-2 bg-red-900/20 border border-red-500/30 rounded text-red-300 text-xs">
            {currentError}
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-700">
        {[
          { id: 'help', label: 'Help', icon: Info },
          { id: 'automation', label: 'Auto', icon: Zap },
          { id: 'troubleshooting', label: 'Debug', icon: AlertTriangle },
          { id: 'security', label: 'Security', icon: Shield }
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id as any)}
            className={\`flex-1 flex items-center justify-center space-x-1 px-3 py-2 text-xs transition-colors \${
              activeTab === id 
                ? 'bg-blue-600 text-white' 
                : 'text-gray-300 hover:text-white hover:bg-gray-700'
            }\`}
          >
            <Icon className="w-3 h-3" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'help' && (
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-white mb-2">{helpContent.title}</h4>
              <p className="text-gray-300 text-sm mb-3">{helpContent.description}</p>
              
              <div className="flex items-center space-x-2 text-xs text-gray-400 mb-3">
                <Clock className="w-3 h-3" />
                <span>Est. time: {helpContent.estimatedTime}</span>
              </div>
            </div>

            <div>
              <h5 className="font-medium text-white mb-2">Steps:</h5>
              <ol className="space-y-1">
                {helpContent.steps.map((step, index) => (
                  <li key={index} className="flex items-start space-x-2 text-sm text-gray-300">
                    <span className="w-5 h-5 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs flex-shrink-0 mt-0.5">
                      {index + 1}
                    </span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            </div>

            <div>
              <h5 className="font-medium text-white mb-2">Requirements:</h5>
              <ul className="space-y-1">
                {helpContent.requirements.map((req, index) => (
                  <li key={index} className="flex items-center space-x-2 text-sm text-gray-300">
                    <CheckCircle className="w-3 h-3 text-green-400 flex-shrink-0" />
                    <span>{req}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h5 className="font-medium text-white mb-2">Pro Tips:</h5>
              <ul className="space-y-1">
                {helpContent.tips.map((tip, index) => (
                  <li key={index} className="flex items-start space-x-2 text-sm text-gray-300">
                    <Zap className="w-3 h-3 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'automation' && (
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-white mb-2">Automation Available</h4>
              {helpContent.automation?.available ? (
                <div className="space-y-3">
                  <p className="text-gray-300 text-sm">{helpContent.automation.description}</p>
                  
                  {automationProgress > 0 && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs text-gray-400">
                        <span>Progress</span>
                        <span>{automationProgress}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: \`\${automationProgress}%\` }}
                        />
                      </div>
                    </div>
                  )}
                  
                  <button
                    onClick={() => handleAutomation(helpContent.automation!.action)}
                    disabled={automationProgress > 0 && automationProgress < 100}
                    className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
                  >
                    <Zap className="w-4 h-4" />
                    <span>{automationProgress > 0 && automationProgress < 100 ? 'Running...' : 'Run Automation'}</span>
                  </button>
                </div>
              ) : (
                <p className="text-gray-400 text-sm">No automation available for this step.</p>
              )}
            </div>

            <div>
              <h5 className="font-medium text-white mb-2">Quick Actions:</h5>
              <div className="space-y-2">
                <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                  <Download className="w-3 h-3" />
                  <span>Download Build Scripts</span>
                </button>
                <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                  <FileText className="w-3 h-3" />
                  <span>Generate Config Files</span>
                </button>
                <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                  <Terminal className="w-3 h-3" />
                  <span>Open Terminal Commands</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'troubleshooting' && (
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-white mb-2">Common Issues</h4>
              {helpContent.troubleshooting.length > 0 ? (
                <div className="space-y-3">
                  {helpContent.troubleshooting.map((item, index) => (
                    <div key={index} className="bg-gray-700 rounded-lg p-3">
                      <div className="flex items-start space-x-2 mb-2">
                        <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                        <h6 className="font-medium text-white text-sm">{item.issue}</h6>
                      </div>
                      <p className="text-gray-300 text-sm ml-6">{item.solution}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-sm">No known issues for this step.</p>
              )}
            </div>

            <div>
              <h5 className="font-medium text-white mb-2">Debug Tools:</h5>
              <div className="space-y-2">
                <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                  <Terminal className="w-3 h-3" />
                  <span>View Build Logs</span>
                </button>
                <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                  <Settings className="w-3 h-3" />
                  <span>Check Configuration</span>
                </button>
                <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                  <ExternalLink className="w-3 h-3" />
                  <span>Open Documentation</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'security' && (
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-white mb-2">Security Checklist</h4>
              <div className="space-y-2">
                {[
                  'Keystore properly secured',
                  'Environment variables configured',
                  'Secrets not in version control',
                  'Code signing certificates valid',
                  'API keys properly scoped'
                ].map((item, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-gray-300 text-sm">{item}</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h5 className="font-medium text-white mb-2">Sensitive Data:</h5>
              <div className="space-y-2">
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-white">Keystore Password</span>
                    <button
                      onClick={() => setShowSecrets(!showSecrets)}
                      className="text-gray-400 hover:text-white"
                    >
                      {showSecrets ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <div className="flex items-center space-x-2">
                    <code className="flex-1 text-xs bg-gray-800 p-2 rounded text-gray-300">
                      {showSecrets ? 'your_keystore_password' : '••••••••••••••••'}
                    </code>
                    <button
                      onClick={() => copyToClipboard('your_keystore_password')}
                      className="text-gray-400 hover:text-white"
                    >
                      <Copy className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h5 className="font-medium text-white mb-2">Security Actions:</h5>
              <div className="space-y-2">
                <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                  <Key className="w-3 h-3" />
                  <span>Rotate Signing Keys</span>
                </button>
                <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                  <Shield className="w-3 h-3" />
                  <span>Audit Permissions</span>
                </button>
                <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                  <Upload className="w-3 h-3" />
                  <span>Backup Keystore</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DeploymentAssistant;`,
          language: 'typescript'
        },
        'AutomationPanel.tsx': {
          content: `import React, { useState } from 'react';
import { 
  Zap, 
  Play, 
  Pause, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Terminal,
  Download,
  Settings,
  RefreshCw,
  AlertTriangle
} from 'lucide-react';
import { useDeploymentAutomation } from '../hooks/useDeploymentAutomation';

interface AutomationPanelProps {
  platform: string;
  isVisible: boolean;
  onClose: () => void;
}

const AutomationPanel: React.FC<AutomationPanelProps> = ({ platform, isVisible, onClose }) => {
  const { runAutomation, automationSteps, isRunning, progress } = useDeploymentAutomation();
  const [selectedAutomation, setSelectedAutomation] = useState<string>('');

  const automationOptions = [
    {
      id: 'full-deployment',
      name: 'Full Deployment Pipeline',
      description: 'Complete automated deployment from build to store upload',
      estimatedTime: '15-20 minutes',
      steps: ['Build Rust', 'Build App', 'Sign & Package', 'Upload to Store'],
      icon: <Zap className="w-5 h-5 text-orange-400" />
    },
    {
      id: 'build-only',
      name: 'Build & Package Only',
      description: 'Build and package app without uploading to store',
      estimatedTime: '8-12 minutes',
      steps: ['Build Rust', 'Build App', 'Sign & Package'],
      icon: <Settings className="w-5 h-5 text-blue-400" />
    },
    {
      id: 'rust-build',
      name: 'Rust Build Only',
      description: 'Build Rust code for target platform',
      estimatedTime: '3-5 minutes',
      steps: ['Setup Environment', 'Add Targets', 'Build Libraries'],
      icon: <Terminal className="w-5 h-5 text-green-400" />
    }
  ];

  const handleRunAutomation = async (automationId: string) => {
    setSelectedAutomation(automationId);
    await runAutomation(automationId, platform);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'running':
        return <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  if (!isVisible) return null;

  return (
    <div className="fixed left-4 top-20 bottom-4 w-80 bg-gray-800 border border-gray-700 rounded-lg shadow-2xl z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <Zap className="w-5 h-5 text-orange-400" />
          <h3 className="font-semibold text-white">Automation Center</h3>
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors"
        >
          ✕
        </button>
      </div>

      {/* Platform Info */}
      <div className="p-4 border-b border-gray-700">
        <div className="text-sm text-gray-400 mb-1">Target Platform</div>
        <div className="text-lg font-semibold text-white capitalize">{platform}</div>
      </div>

      {/* Automation Options */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-4">
          <h4 className="font-medium text-white">Available Automations</h4>
          
          {automationOptions.map((option) => (
            <div
              key={option.id}
              className={\`p-4 rounded-lg border transition-all cursor-pointer \${
                selectedAutomation === option.id
                  ? 'border-orange-500 bg-orange-900/20'
                  : 'border-gray-600 bg-gray-700 hover:border-gray-500'
              }\`}
              onClick={() => setSelectedAutomation(option.id)}
            >
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 mt-1">
                  {option.icon}
                </div>
                <div className="flex-1">
                  <h5 className="font-medium text-white mb-1">{option.name}</h5>
                  <p className="text-gray-300 text-sm mb-2">{option.description}</p>
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>Est. time: {option.estimatedTime}</span>
                    <span>{option.steps.length} steps</span>
                  </div>
                </div>
              </div>
              
              {selectedAutomation === option.id && (
                <div className="mt-3 pt-3 border-t border-gray-600">
                  <div className="space-y-1">
                    {option.steps.map((step, index) => (
                      <div key={index} className="flex items-center space-x-2 text-sm text-gray-300">
                        <div className="w-4 h-4 bg-gray-600 rounded-full flex items-center justify-center text-xs text-white">
                          {index + 1}
                        </div>
                        <span>{step}</span>
                      </div>
                    ))}
                  </div>
                  
                  <button
                    onClick={() => handleRunAutomation(option.id)}
                    disabled={isRunning}
                    className="w-full mt-3 flex items-center justify-center space-x-2 py-2 px-4 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
                  >
                    {isRunning ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Running...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        <span>Run Automation</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Progress Section */}
        {isRunning && (
          <div className="mt-6 p-4 bg-gray-700 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <h5 className="font-medium text-white">Automation Progress</h5>
              <span className="text-sm text-gray-400">{Math.round(progress)}%</span>
            </div>
            
            <div className="w-full bg-gray-600 rounded-full h-2 mb-4">
              <div 
                className="bg-orange-600 h-2 rounded-full transition-all duration-300"
                style={{ width: \`\${progress}%\` }}
              />
            </div>
            
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {automationSteps.map((step) => (
                <div key={step.id} className="flex items-center space-x-2 text-sm">
                  {getStatusIcon(step.status)}
                  <span className={\`\${
                    step.status === 'completed' ? 'text-green-300' :
                    step.status === 'failed' ? 'text-red-300' :
                    step.status === 'running' ? 'text-blue-300' :
                    'text-gray-400'
                  }\`}>
                    {step.name}
                  </span>
                  {step.duration && (
                    <span className="text-xs text-gray-500">
                      ({(step.duration / 1000).toFixed(1)}s)
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="mt-6">
          <h5 className="font-medium text-white mb-3">Quick Actions</h5>
          <div className="space-y-2">
            <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
              <Download className="w-3 h-3" />
              <span>Download Build Scripts</span>
            </button>
            <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
              <Settings className="w-3 h-3" />
              <span>Configure Environment</span>
            </button>
            <button className="w-full flex items-center space-x-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
              <Terminal className="w-3 h-3" />
              <span>Open Terminal</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AutomationPanel;`,
          language: 'typescript'
        },
        'useDeploymentAutomation.ts': {
          content: `import { useState, useCallback } from 'react';

interface AutomationStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  command?: string;
  output?: string;
  duration?: number;
}

interface UseDeploymentAutomationReturn {
  runAutomation: (action: string, platform: string) => Promise<void>;
  automationSteps: AutomationStep[];
  isRunning: boolean;
  progress: number;
}

export const useDeploymentAutomation = (): UseDeploymentAutomationReturn => {
  const [automationSteps, setAutomationSteps] = useState<AutomationStep[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const updateStep = (stepId: string, updates: Partial<AutomationStep>) => {
    setAutomationSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, ...updates } : step
    ));
  };

  const executeCommand = async (command: string, stepId: string): Promise<string> => {
    updateStep(stepId, { status: 'running', command });
    
    // Simulate command execution
    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 3000));
    
    // Mock command outputs based on the command
    let output = '';
    
    if (command.includes('rustup target add')) {
      output = \`info: downloading component 'rust-std' for 'aarch64-linux-android'
info: installing component 'rust-std' for 'aarch64-linux-android'
info: downloading component 'rust-std' for 'armv7-linux-androideabi'
info: installing component 'rust-std' for 'armv7-linux-androideabi'\`;
    } else if (command.includes('cargo build')) {
      output = \`   Compiling rustyclint v0.1.0
    Finished release [optimized] target(s) in 45.67s\`;
    } else if (command.includes('gradlew')) {
      output = \`> Task :app:bundleRelease
BUILD SUCCESSFUL in 2m 34s
1 actionable task: 1 executed\`;
    } else if (command.includes('wasm-pack')) {
      output = \`[INFO]: Checking for the Wasm target...
[INFO]: Compiling to Wasm...
[INFO]: Optimizing wasm binaries with wasm-opt...
[INFO]: Done in 23.45s\`;
    } else if (command.includes('xcodebuild')) {
      output = \`** BUILD SUCCEEDED **
Archive: /path/to/App.xcarchive\`;
    }
    
    updateStep(stepId, { status: 'completed', output });
    return output;
  };

  const getAutomationSteps = (action: string, platform: string): AutomationStep[] => {
    switch (action) {
      case 'auto-build-rust':
        if (platform === 'android') {
          return [
            { id: 'install-ndk', name: 'Install Android NDK', status: 'pending' },
            { id: 'add-targets', name: 'Add Android Targets', status: 'pending' },
            { id: 'build-aarch64', name: 'Build ARM64', status: 'pending' },
            { id: 'build-armv7', name: 'Build ARMv7', status: 'pending' },
            { id: 'build-x86', name: 'Build x86', status: 'pending' },
            { id: 'build-x86_64', name: 'Build x86_64', status: 'pending' },
            { id: 'copy-libs', name: 'Copy Libraries', status: 'pending' }
          ];
        }
        break;
      
      case 'auto-build-android':
        return [
          { id: 'check-keystore', name: 'Verify Keystore', status: 'pending' },
          { id: 'configure-signing', name: 'Configure Signing', status: 'pending' },
          { id: 'gradle-clean', name: 'Clean Project', status: 'pending' },
          { id: 'gradle-build', name: 'Build AAB', status: 'pending' },
          { id: 'verify-signature', name: 'Verify Signature', status: 'pending' }
        ];
      
      case 'auto-build-wasm':
        return [
          { id: 'install-wasm-pack', name: 'Install wasm-pack', status: 'pending' },
          { id: 'add-wasm-target', name: 'Add WASM Target', status: 'pending' },
          { id: 'build-wasm', name: 'Build WebAssembly', status: 'pending' },
          { id: 'optimize-wasm', name: 'Optimize Binary', status: 'pending' },
          { id: 'generate-bindings', name: 'Generate JS Bindings', status: 'pending' }
        ];
      
      case 'auto-build-ios':
        return [
          { id: 'check-certificates', name: 'Check Certificates', status: 'pending' },
          { id: 'configure-signing', name: 'Configure Code Signing', status: 'pending' },
          { id: 'build-rust-ios', name: 'Build Rust for iOS', status: 'pending' },
          { id: 'xcode-build', name: 'Build with Xcode', status: 'pending' },
          { id: 'create-archive', name: 'Create Archive', status: 'pending' },
          { id: 'export-ipa', name: 'Export IPA', status: 'pending' }
        ];
      
      case 'auto-upload-playstore':
        return [
          { id: 'check-credentials', name: 'Check API Credentials', status: 'pending' },
          { id: 'validate-aab', name: 'Validate AAB', status: 'pending' },
          { id: 'upload-aab', name: 'Upload to Play Console', status: 'pending' },
          { id: 'set-track', name: 'Set Release Track', status: 'pending' },
          { id: 'submit-review', name: 'Submit for Review', status: 'pending' }
        ];
      
      default:
        return [];
    }
    
    return [];
  };

  const runAutomation = useCallback(async (action: string, platform: string) => {
    setIsRunning(true);
    setProgress(0);
    
    const steps = getAutomationSteps(action, platform);
    setAutomationSteps(steps);
    
    try {
      for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        setProgress((i / steps.length) * 100);
        
        // Execute the automation step
        const command = getCommandForStep(step.id, platform);
        await executeCommand(command, step.id);
        
        // Small delay between steps
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      
      setProgress(100);
    } catch (error) {
      console.error('Automation failed:', error);
      // Mark current step as failed
      const currentStepIndex = automationSteps.findIndex(s => s.status === 'running');
      if (currentStepIndex >= 0) {
        updateStep(automationSteps[currentStepIndex].id, { 
          status: 'failed', 
          output: \`Error: \${error}\` 
        });
      }
    } finally {
      setIsRunning(false);
    }
  }, [automationSteps]);

  const getCommandForStep = (stepId: string, platform: string): string => {
    const commands: Record<string, string> = {
      'install-ndk': 'echo "Installing Android NDK..."',
      'add-targets': 'rustup target add aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android',
      'build-aarch64': 'cargo build --target aarch64-linux-android --release',
      'build-armv7': 'cargo build --target armv7-linux-androideabi --release',
      'build-x86': 'cargo build --target i686-linux-android --release',
      'build-x86_64': 'cargo build --target x86_64-linux-android --release',
      'copy-libs': 'echo "Copying native libraries to Android project..."',
      'check-keystore': 'echo "Verifying Android keystore..."',
      'configure-signing': 'echo "Configuring Android signing..."',
      'gradle-clean': 'cd android && ./gradlew clean',
      'gradle-build': 'cd android && ./gradlew bundleRelease',
      'verify-signature': 'echo "Verifying AAB signature..."',
      'install-wasm-pack': 'cargo install wasm-pack',
      'add-wasm-target': 'rustup target add wasm32-unknown-unknown',
      'build-wasm': 'wasm-pack build --target web --release',
      'optimize-wasm': 'wasm-opt -Oz --enable-mutable-globals pkg/rustyclint_bg.wasm -o pkg/rustyclint_bg.wasm',
      'generate-bindings': 'echo "Generating JavaScript bindings..."',
      'check-certificates': 'echo "Checking iOS certificates..."',
      'build-rust-ios': 'cargo build --target aarch64-apple-ios --release',
      'xcode-build': 'xcodebuild -project ios/App.xcodeproj -scheme App -configuration Release build',
      'create-archive': 'xcodebuild -project ios/App.xcodeproj -scheme App -configuration Release archive',
      'export-ipa': 'xcodebuild -exportArchive -archivePath build/App.xcarchive -exportPath build/',
      'check-credentials': 'echo "Checking Google Play API credentials..."',
      'validate-aab': 'echo "Validating AAB file..."',
      'upload-aab': 'fastlane supply --aab android/app/build/outputs/bundle/release/app-release.aab',
      'set-track': 'echo "Setting release track to internal..."',
      'submit-review': 'echo "Submitting for review..."'
    };
    
    return commands[stepId] || \`echo "Executing \${stepId}..."\`;
  };

  return {
    runAutomation,
    automationSteps,
    isRunning,
    progress
  };
};`,
          language: 'typescript'
        },
        'android-automation.sh': {
          content: `#!/bin/bash

# Android Deployment Automation Script
# Provides intelligent automation for Android deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Progress tracking
TOTAL_STEPS=0
CURRENT_STEP=0

update_progress() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local percentage=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    echo "PROGRESS:$percentage"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [ ! -d "android" ]; then
        log_error "Android directory not found. Run this from your project root."
        exit 1
    fi
    
    # Check for Rust
    if ! command -v cargo &> /dev/null; then
        log_error "Rust/Cargo not found. Please install Rust."
        exit 1
    fi
    
    # Check for Java
    if ! command -v java &> /dev/null; then
        log_error "Java not found. Please install Java JDK."
        exit 1
    fi
    
    # Check for Android SDK
    if [ -z "$ANDROID_HOME" ] && [ -z "$ANDROID_SDK_ROOT" ]; then
        log_warning "ANDROID_HOME or ANDROID_SDK_ROOT not set. Attempting to detect..."
        
        # Common Android SDK locations
        POSSIBLE_PATHS=(
            "$HOME/Android/Sdk"
            "$HOME/Library/Android/sdk"
            "/usr/local/android-sdk"
            "/opt/android-sdk"
        )
        
        for path in "${POSSIBLE_PATHS[@]}"; do
            if [ -d "$path" ]; then
                export ANDROID_HOME="$path"
                export ANDROID_SDK_ROOT="$path"
                log_success "Found Android SDK at: $path"
                break
            fi
        done
        
        if [ -z "$ANDROID_HOME" ]; then
            log_error "Android SDK not found. Please install Android SDK."
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Setup Android NDK
setup_ndk() {
    log_info "Setting up Android NDK..."
    
    # Check if NDK is already available
    if [ -n "$ANDROID_NDK_ROOT" ] && [ -d "$ANDROID_NDK_ROOT" ]; then
        log_success "NDK already configured at: $ANDROID_NDK_ROOT"
        return
    fi
    
    # Try to find NDK in SDK directory
    if [ -d "$ANDROID_HOME/ndk" ]; then
        # Find the latest NDK version
        NDK_VERSION=$(ls "$ANDROID_HOME/ndk" | sort -V | tail -n 1)
        if [ -n "$NDK_VERSION" ]; then
            export ANDROID_NDK_ROOT="$ANDROID_HOME/ndk/$NDK_VERSION"
            log_success "Found NDK version: $NDK_VERSION"
        fi
    fi
    
    if [ -z "$ANDROID_NDK_ROOT" ]; then
        log_warning "NDK not found. Attempting to install..."
        
        # Install NDK using sdkmanager
        if command -v sdkmanager &> /dev/null; then
            sdkmanager "ndk;25.2.9519653"
            export ANDROID_NDK_ROOT="$ANDROID_HOME/ndk/25.2.9519653"
        else
            log_error "NDK not found and sdkmanager not available. Please install NDK manually."
            exit 1
        fi
    fi
    
    update_progress
}

# Add Rust targets for Android
add_rust_targets() {
    log_info "Adding Rust targets for Android..."
    
    local targets=(
        "aarch64-linux-android"
        "armv7-linux-androideabi"
        "i686-linux-android"
        "x86_64-linux-android"
    )
    
    for target in "${targets[@]}"; do
        log_info "Adding target: $target"
        rustup target add "$target"
    done
    
    log_success "All Android targets added"
    update_progress
}

# Build Rust code for Android
build_rust_android() {
    log_info "Building Rust code for Android architectures..."
    
    # Set up environment for cross-compilation
    export CC_aarch64_linux_android="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang"
    export CC_armv7_linux_androideabi="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi21-clang"
    export CC_i686_linux_android="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android21-clang"
    export CC_x86_64_linux_android="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android21-clang"
    
    # Detect OS for NDK path
    if [[ "$OSTYPE" == "darwin"* ]]; then
        NDK_HOST="darwin-x86_64"
    else
        NDK_HOST="linux-x86_64"
    fi
    
    # Update CC paths for detected OS
    export CC_aarch64_linux_android="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/$NDK_HOST/bin/aarch64-linux-android21-clang"
    export CC_armv7_linux_androideabi="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/$NDK_HOST/bin/armv7a-linux-androideabi21-clang"
    export CC_i686_linux_android="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/$NDK_HOST/bin/i686-linux-android21-clang"
    export CC_x86_64_linux_android="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/$NDK_HOST/bin/x86_64-linux-android21-clang"
    
    local targets=(
        "aarch64-linux-android"
        "armv7-linux-androideabi"
        "i686-linux-android"
        "x86_64-linux-android"
    )
    
    for target in "${targets[@]}"; do
        log_info "Building for target: $target"
        cargo build --target "$target" --release
        
        if [ $? -eq 0 ]; then
            log_success "Build successful for $target"
        else
            log_error "Build failed for $target"
            exit 1
        fi
    done
    
    update_progress
}

# Copy native libraries to Android project
copy_native_libraries() {
    log_info "Copying native libraries to Android project..."
    
    # Create jniLibs directories
    mkdir -p android/app/src/main/jniLibs/{arm64-v8a,armeabi-v7a,x86,x86_64}
    
    # Copy libraries
    cp target/aarch64-linux-android/release/librustyclint.so android/app/src/main/jniLibs/arm64-v8a/ 2>/dev/null || log_warning "ARM64 library not found"
    cp target/armv7-linux-androideabi/release/librustyclint.so android/app/src/main/jniLibs/armeabi-v7a/ 2>/dev/null || log_warning "ARMv7 library not found"
    cp target/i686-linux-android/release/librustyclint.so android/app/src/main/jniLibs/x86/ 2>/dev/null || log_warning "x86 library not found"
    cp target/x86_64-linux-android/release/librustyclint.so android/app/src/main/jniLibs/x86_64/ 2>/dev/null || log_warning "x86_64 library not found"
    
    log_success "Native libraries copied"
    update_progress
}

# Verify keystore
verify_keystore() {
    log_info "Verifying Android keystore..."
    
    local keystore_path="android/keystore/release.keystore"
    local properties_path="android/keystore.properties"
    
    if [ ! -f "$keystore_path" ]; then
        log_error "Keystore not found at: $keystore_path"
        log_info "Run ./scripts/generate-android-keystore.sh to create one"
        exit 1
    fi
    
    if [ ! -f "$properties_path" ]; then
        log_error "Keystore properties not found at: $properties_path"
        exit 1
    fi
    
    # Load keystore properties
    source "$properties_path"
    
    # Verify keystore can be accessed
    if keytool -list -keystore "$keystore_path" -storepass "$storePassword" -alias "$keyAlias" &>/dev/null; then
        log_success "Keystore verification passed"
    else
        log_error "Keystore verification failed. Check passwords and alias."
        exit 1
    fi
    
    update_progress
}

# Build Android AAB
build_android_aab() {
    log_info "Building Android App Bundle (AAB)..."
    
    cd android
    
    # Clean previous builds
    ./gradlew clean
    
    # Build signed AAB
    ./gradlew bundleRelease
    
    if [ $? -eq 0 ]; then
        log_success "AAB build successful"
        
        # Check output file
        local aab_path="app/build/outputs/bundle/release/app-release.aab"
        if [ -f "$aab_path" ]; then
            local size=$(du -h "$aab_path" | cut -f1)
            log_success "AAB created: $aab_path (Size: $size)"
        else
            log_error "AAB file not found at expected location"
            exit 1
        fi
    else
        log_error "AAB build failed"
        exit 1
    fi
    
    cd ..
    update_progress
}

# Verify AAB signature
verify_aab_signature() {
    log_info "Verifying AAB signature..."
    
    local aab_path="android/app/build/outputs/bundle/release/app-release.aab"
    
    if command -v bundletool &> /dev/null; then
        bundletool validate --bundle="$aab_path"
        if [ $? -eq 0 ]; then
            log_success "AAB signature verification passed"
        else
            log_error "AAB signature verification failed"
            exit 1
        fi
    else
        log_warning "bundletool not found. Skipping AAB validation."
    fi
    
    update_progress
}

# Upload to Google Play (optional)
upload_to_play_store() {
    log_info "Checking for Google Play upload configuration..."
    
    if [ -f "android/fastlane/Fastfile" ] && [ -n "$GOOGLE_PLAY_SERVICE_ACCOUNT" ]; then
        log_info "Uploading to Google Play Store..."
        
        cd android
        fastlane deploy
        
        if [ $? -eq 0 ]; then
            log_success "Upload to Google Play Store successful"
        else
            log_error "Upload to Google Play Store failed"
            exit 1
        fi
        
        cd ..
    else
        log_warning "Google Play upload not configured. Skipping upload."
        log_info "To enable upload:"
        log_info "1. Set up Fastlane configuration"
        log_info "2. Set GOOGLE_PLAY_SERVICE_ACCOUNT environment variable"
    fi
    
    update_progress
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."
    
    local report_file="deployment-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" << EOF
Android Deployment Report
========================
Date: $(date)
Platform: Android
Build Type: Release

Files Generated:
- AAB: android/app/build/outputs/bundle/release/app-release.aab

Native Libraries:
- ARM64: $([ -f "android/app/src/main/jniLibs/arm64-v8a/librustyclint.so" ] && echo "✓" || echo "✗")
- ARMv7: $([ -f "android/app/src/main/jniLibs/armeabi-v7a/librustyclint.so" ] && echo "✓" || echo "✗")
- x86: $([ -f "android/app/src/main/jniLibs/x86/librustyclint.so" ] && echo "✓" || echo "✗")
- x86_64: $([ -f "android/app/src/main/jniLibs/x86_64/librustyclint.so" ] && echo "✓" || echo "✗")

Build Status: SUCCESS
Total Steps: $TOTAL_STEPS
Completed Steps: $CURRENT_STEP

Next Steps:
1. Test the AAB on internal track
2. Upload to Google Play Console
3. Submit for review

EOF

    log_success "Report generated: $report_file"
}

# Main automation function
run_automation() {
    local automation_type="$1"
    
    case "$automation_type" in
        "full")
            TOTAL_STEPS=8
            log_info "Starting full Android deployment automation..."
            check_prerequisites
            setup_ndk
            add_rust_targets
            build_rust_android
            copy_native_libraries
            verify_keystore
            build_android_aab
            verify_aab_signature
            upload_to_play_store
            ;;
        "build")
            TOTAL_STEPS=6
            log_info "Starting Android build automation..."
            check_prerequisites
            setup_ndk
            add_rust_targets
            build_rust_android
            copy_native_libraries
            verify_keystore
            build_android_aab
            ;;
        "rust")
            TOTAL_STEPS=3
            log_info "Starting Rust build automation..."
            check_prerequisites
            setup_ndk
            add_rust_targets
            build_rust_android
            ;;
        *)
            log_error "Unknown automation type: $automation_type"
            log_info "Available types: full, build, rust"
            exit 1
            ;;
    esac
    
    generate_report
    log_success "Android automation completed successfully!"
}

# Script entry point
if [ $# -eq 0 ]; then
    echo "Usage: $0 <automation_type>"
    echo "Types: full, build, rust"
    exit 1
fi

run_automation "$1"`,
          language: 'bash'
        }
      },
      features: [
        'Visual deployment progress tracking',
        'Step-by-step deployment assistant',
        'Automated build and deployment scripts',
        'Intelligent troubleshooting',
        'Security best practices',
        'CI/CD integration'
      ],
      useCase: 'For teams that need a robust, visual deployment pipeline for Android apps with progress tracking and automation.',
      techStack: ['React', 'TypeScript', 'Android', 'Gradle', 'Bash', 'CI/CD']
    },
    {
      id: 'ios-deployment-pipeline',
      name: 'iOS Deployment Pipeline',
      description: 'Complete deployment pipeline for iOS apps with visual progress tracking and automation',
      category: 'deployment',
      difficulty: 'Advanced',
      tags: ['iOS', 'CI/CD', 'Deployment', 'Automation'],
      icon: <Smartphone className="w-8 h-8 text-blue-400" />,
      estimatedTime: '60 minutes',
      files: {},
      features: [
        'Visual deployment progress tracking',
        'Step-by-step deployment assistant',
        'Automated build and deployment scripts',
        'Code signing management',
        'App Store Connect integration',
        'TestFlight distribution'
      ],
      useCase: 'For teams that need a robust, visual deployment pipeline for iOS apps with progress tracking and automation.',
      techStack: ['React', 'TypeScript', 'iOS', 'Xcode', 'Bash', 'CI/CD']
    },
    {
      id: 'web-deployment-pipeline',
      name: 'Web Deployment Pipeline',
      description: 'Complete deployment pipeline for web apps with visual progress tracking and automation',
      category: 'deployment',
      difficulty: 'Intermediate',
      tags: ['Web', 'CI/CD', 'Deployment', 'Automation'],
      icon: <Globe className="w-8 h-8 text-purple-400" />,
      estimatedTime: '45 minutes',
      files: {},
      features: [
        'Visual deployment progress tracking',
        'Step-by-step deployment assistant',
        'Automated build and deployment scripts',
        'CDN configuration',
        'Multi-environment support',
        'Performance optimization'
      ],
      useCase: 'For teams that need a robust, visual deployment pipeline for web apps with progress tracking and automation.',
      techStack: ['React', 'TypeScript', 'Vite', 'Netlify/Vercel', 'GitHub Actions']
    }
  ];

  const filteredTemplates = templates.filter(template => {
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
    const matchesSearch = searchQuery === '' || 
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    return matchesCategory && matchesSearch;
  });

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-gray-800 rounded-lg max-w-6xl w-full p-6 border border-gray-700 my-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-white mb-2">Project Templates</h2>
            <p className="text-gray-400">Start with a pre-configured template to accelerate your development</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-xl"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="flex gap-6">
          {/* Sidebar */}
          <div className="w-64 flex-shrink-0">
            <div className="mb-6">
              <div className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search templates..."
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-3">Categories</h3>
              <div className="space-y-2">
                {categories.map((category) => (
                  <button
                    key={category.id}
                    onClick={() => setSelectedCategory(category.id)}
                    className={`flex items-center space-x-2 w-full px-3 py-2 rounded-lg transition-colors ${
                      selectedCategory === category.id
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    {category.icon}
                    <span>{category.name}</span>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-3">Difficulty</h3>
              <div className="space-y-2">
                {['Beginner', 'Intermediate', 'Advanced', 'Expert'].map((level) => (
                  <label key={level} className="flex items-center space-x-2 text-gray-300">
                    <input type="checkbox" className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500" />
                    <span>{level}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Templates Grid */}
          <div className="flex-1">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {filteredTemplates.map((template) => (
                <div
                  key={template.id}
                  className="bg-gray-700 rounded-lg p-6 border border-gray-600 hover:border-blue-500 transition-colors cursor-pointer"
                  onClick={() => setSelectedTemplate(template)}
                >
                  <div className="flex items-start space-x-4">
                    <div className="p-3 bg-gray-800 rounded-lg">
                      {template.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-white mb-2">{template.name}</h3>
                      <p className="text-gray-300 text-sm mb-3">{template.description}</p>
                      
                      <div className="flex items-center space-x-4 text-sm text-gray-400 mb-3">
                        <div className="flex items-center space-x-1">
                          <Clock className="w-4 h-4" />
                          <span>{template.estimatedTime}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Star className="w-4 h-4 text-yellow-400" />
                          <span>{template.difficulty}</span>
                        </div>
                      </div>
                      
                      <div className="flex flex-wrap gap-2">
                        {template.tags.map((tag) => (
                          <span key={tag} className="px-2 py-1 bg-gray-600 text-gray-300 rounded text-xs">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Template Details Modal */}
        {selectedTemplate && (
          <div className="fixed inset-0 bg-black/80 z-60 flex items-center justify-center p-4">
            <div className="bg-gray-800 rounded-lg max-w-4xl w-full p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div className="p-4 bg-gray-700 rounded-lg">
                    {selectedTemplate.icon}
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-white">{selectedTemplate.name}</h3>
                    <p className="text-gray-400">{selectedTemplate.description}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedTemplate(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <h4 className="text-lg font-semibold text-white mb-3">Features</h4>
                  <ul className="space-y-2">
                    {selectedTemplate.features.map((feature, index) => (
                      <li key={index} className="flex items-start space-x-2 text-gray-300">
                        <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-semibold text-white mb-3">Tech Stack</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedTemplate.techStack.map((tech) => (
                      <span key={tech} className="px-3 py-1 bg-blue-900/30 text-blue-300 rounded-full text-sm">
                        {tech}
                      </span>
                    ))}
                  </div>
                  
                  <h4 className="text-lg font-semibold text-white mt-6 mb-3">Use Case</h4>
                  <p className="text-gray-300">{selectedTemplate.useCase}</p>
                </div>
              </div>

              <div className="bg-gray-700 rounded-lg p-4 mb-6">
                <h4 className="text-lg font-semibold text-white mb-3">Included Files</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {Object.keys(selectedTemplate.files).map((fileName) => (
                    <div key={fileName} className="flex items-center space-x-2 text-gray-300">
                      <Code className="w-4 h-4 text-blue-400" />
                      <span>{fileName}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex justify-end space-x-4">
                <button
                  onClick={() => setSelectedTemplate(null)}
                  className="px-4 py-2 border border-gray-600 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    onSelectTemplate(selectedTemplate);
                    setSelectedTemplate(null);
                    onClose();
                  }}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  Use Template
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProjectTemplates;