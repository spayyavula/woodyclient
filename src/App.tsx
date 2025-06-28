import React, { useState, useEffect } from 'react';
import { supabase } from './lib/supabase';
import { isSupabaseConfigured } from './lib/supabase';
import LandingPage from './components/LandingPage';
import LoginPage from './components/auth/LoginPage';
import SignupPage from './components/auth/SignupPage';
import SuccessPage from './components/SuccessPage';
import PricingPage from './components/PricingPage';
import UserProfile from './components/UserProfile';
import FileExplorer from './components/FileExplorer';
import CodeEditor from './components/CodeEditor';
import Terminal from './components/Terminal';
import Toolbar from './components/Toolbar';
import TabBar from './components/TabBar';
import StatusBar from './components/StatusBar';
import CollaborationPanel from './components/CollaborationPanel';
import DeveloperMarketplace from './components/DeveloperMarketplace';
import ProjectTemplates from './components/ProjectTemplates';
import MobilePreview from './components/MobilePreview';
import DemoMode from './components/DemoMode';
import StripeTestSuite from './components/StripeTestSuite';

interface Tab {
  id: string;
  name: string;
  content: string;
  language: string;
  isDirty: boolean;
}

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

function App() {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [showSuccess, setShowSuccess] = useState(false);
  const [showPricing, setShowPricing] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);
  const [showMobilePreview, setShowMobilePreview] = useState(false);
  const [showDemo, setShowDemo] = useState(false);
  const [showStripeTests, setShowStripeTests] = useState(false);
  const [isTerminalVisible, setIsTerminalVisible] = useState(false);
  const [isCollaborationVisible, setIsCollaborationVisible] = useState(false);
  const [isMarketplaceVisible, setIsMarketplaceVisible] = useState(false);
  
  const [tabs, setTabs] = useState<Tab[]>([
    {
      id: 'main.rs',
      name: 'main.rs',
      content: `use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct MobileApp {
    pub name: String,
    pub version: String,
    pub platform: Platform,
    pub features: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Platform {
    Android,
    iOS,
    Flutter,
    ReactNative,
}

impl MobileApp {
    pub fn new(name: String, platform: Platform) -> Self {
        Self {
            name,
            version: "1.0.0".to_string(),
            platform,
            features: Vec::new(),
        }
    }
    
    pub fn add_feature(&mut self, feature: String) {
        self.features.push(feature);
    }
    
    pub fn build(&self) -> Result<String, String> {
        match self.platform {
            Platform::Android => self.build_android(),
            Platform::iOS => self.build_ios(),
            Platform::Flutter => self.build_flutter(),
            Platform::ReactNative => self.build_react_native(),
        }
    }
    
    fn build_android(&self) -> Result<String, String> {
        println!("🤖 Building for Android...");
        println!("📦 Compiling Rust code to JNI...");
        println!("🔧 Generating Kotlin bindings...");
        println!("📱 Creating APK...");
        Ok("Android build successful! 🎉".to_string())
    }
    
    fn build_ios(&self) -> Result<String, String> {
        println!("🍎 Building for iOS...");
        println!("📦 Compiling Rust code to C bindings...");
        println!("🔧 Generating Swift bindings...");
        println!("📱 Creating iOS framework...");
        Ok("iOS build successful! 🎉".to_string())
    }
    
    fn build_flutter(&self) -> Result<String, String> {
        println!("🎯 Building for Flutter...");
        println!("📦 Compiling Rust code with flutter_rust_bridge...");
        println!("🔧 Generating Dart bindings...");
        println!("📱 Building Flutter app...");
        Ok("Flutter build successful! 🎉".to_string())
    }
    
    fn build_react_native(&self) -> Result<String, String> {
        println!("⚛️ Building for React Native...");
        println!("📦 Compiling Rust code to native modules...");
        println!("🔧 Generating TypeScript bindings...");
        println!("📱 Building React Native app...");
        Ok("React Native build successful! 🎉".to_string())
    }
}

fn main() {
    println!("🦀 Welcome to Mobile Rust Development!");
    
    let mut app = MobileApp::new(
        "MyAwesomeApp".to_string(),
        Platform::Flutter,
    );
    
    app.add_feature("Authentication".to_string());
    app.add_feature("Real-time Chat".to_string());
    app.add_feature("Push Notifications".to_string());
    app.add_feature("Offline Storage".to_string());
    
    println!("📱 App: {:?}", app);
    
    match app.build() {
        Ok(message) => println!("✅ {}", message),
        Err(error) => println!("❌ Build failed: {}", error),
    }
    
    println!("🚀 Ready for deployment!");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mobile_app_creation() {
        let app = MobileApp::new("TestApp".to_string(), Platform::Android);
        assert_eq!(app.name, "TestApp");
        assert_eq!(app.version, "1.0.0");
    }
    
    #[test]
    fn test_add_feature() {
        let mut app = MobileApp::new("TestApp".to_string(), Platform::iOS);
        app.add_feature("GPS".to_string());
        assert_eq!(app.features.len(), 1);
        assert_eq!(app.features[0], "GPS");
    }
}`,
      language: 'rust',
      isDirty: false,
    },
  ]);
  
  const [activeTab, setActiveTab] = useState('main.rs');
  const [selectedFile, setSelectedFile] = useState('mobile-rust-app/0/src/main.rs');
  const [buildStatus, setBuildStatus] = useState<'idle' | 'building' | 'success' | 'error'>('idle');

  useEffect(() => {
    // Check URL for success parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('success') === 'true') {
      setShowSuccess(true);
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }

    // Check for existing session
    const checkSession = async () => {
      try {
        if (isSupabaseConfigured) {
          const { data: { session } } = await supabase.auth.getSession();
          setUser(session?.user ?? null);
        } else {
          // In demo mode, always start with no user to show landing page
          setUser(null);
        }
      } catch (error) {
        console.error('Error checking session:', error);
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    checkSession();

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleLogin = async (email: string, password: string) => {
    setAuthLoading(true);
    setAuthError(null);

    // Handle demo mode
    if (!isSupabaseConfigured) {
      // In demo mode, simulate successful login
      setTimeout(() => {
        setUser({ 
          id: 'demo-user', 
          email: email,
          created_at: new Date().toISOString(),
          last_sign_in_at: new Date().toISOString()
        });
        setAuthLoading(false);
      }, 1000);
      return;
    }

    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        setAuthError(error.message);
      }
    } catch (err: any) {
      setAuthError('Authentication failed. Please try again.');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleSignup = async (email: string, password: string) => {
    setAuthLoading(true);
    setAuthError(null);

    if (password.length < 6) {
      setAuthError('Password must be at least 6 characters long');
      setAuthLoading(false);
      return;
    }

    // Handle demo mode
    if (!isSupabaseConfigured) {
      // In demo mode, simulate successful signup
      setTimeout(() => {
        setUser({ 
          id: 'demo-user', 
          email: email,
          created_at: new Date().toISOString(),
          last_sign_in_at: new Date().toISOString()
        });
        setAuthLoading(false);
      }, 1000);
      return;
    }

    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: undefined, // Disable email confirmation
        },
      });

      if (error) {
        setAuthError(error.message);
      }
    } catch (err: any) {
      setAuthError('Signup failed. Please try again.');
    } finally {
      setAuthLoading(false);
    }
  };


  const handleLogout = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setShowProfile(false);
  };

  const handleFileSelect = (filePath: string) => {
    setSelectedFile(filePath);
    const fileName = filePath.split('/').pop() || 'unknown';
    
    // Check if tab already exists
    const existingTab = tabs.find(tab => tab.id === fileName);
    if (!existingTab) {
      // Create new tab with sample content based on file type
      const newTab: Tab = {
        id: fileName,
        name: fileName,
        content: getFileContent(fileName),
        language: getLanguageFromFileName(fileName),
        isDirty: false,
      };
      setTabs(prev => [...prev, newTab]);
    }
    setActiveTab(fileName);
  };

  const getFileContent = (fileName: string): string => {
    // Return appropriate content based on file type
    if (fileName.endsWith('.rs')) {
      return `// Rust file: ${fileName}
fn main() {
    println!("Hello from {}!", "${fileName}");
}`;
    } else if (fileName.endsWith('.dart')) {
      return `// Flutter/Dart file: ${fileName}
void main() {
  print('Hello from ${fileName}!');
}`;
    } else if (fileName.endsWith('.kt')) {
      return `// Kotlin file: ${fileName}
fun main() {
    println("Hello from ${fileName}!")
}`;
    }
    return `// ${fileName}
console.log('Hello from ${fileName}!');`;
  };

  const getLanguageFromFileName = (fileName: string): string => {
    if (fileName.endsWith('.rs')) return 'rust';
    if (fileName.endsWith('.dart')) return 'dart';
    if (fileName.endsWith('.kt')) return 'kotlin';
    if (fileName.endsWith('.ts') || fileName.endsWith('.tsx')) return 'typescript';
    if (fileName.endsWith('.js') || fileName.endsWith('.jsx')) return 'javascript';
    if (fileName.endsWith('.py')) return 'python';
    if (fileName.endsWith('.java')) return 'java';
    if (fileName.endsWith('.swift')) return 'swift';
    if (fileName.endsWith('.m')) return 'objective-c';
    if (fileName.endsWith('.toml')) return 'toml';
    if (fileName.endsWith('.xml')) return 'xml';
    if (fileName.endsWith('.json')) return 'json';
    return 'text';
  };

  const handleTabClose = (tabId: string) => {
    setTabs(prev => prev.filter(tab => tab.id !== tabId));
    if (activeTab === tabId) {
      const remainingTabs = tabs.filter(tab => tab.id !== tabId);
      setActiveTab(remainingTabs.length > 0 ? remainingTabs[0].id : '');
    }
  };

  const handleCodeChange = (content: string) => {
    setTabs(prev => prev.map(tab => 
      tab.id === activeTab 
        ? { ...tab, content, isDirty: true }
        : tab
    ));
  };

  const handleSave = () => {
    setTabs(prev => prev.map(tab => 
      tab.id === activeTab 
        ? { ...tab, isDirty: false }
        : tab
    ));
  };

  const handleRun = () => {
    setBuildStatus('building');
    setTimeout(() => {
      setBuildStatus('success');
      setTimeout(() => setBuildStatus('idle'), 3000);
    }, 2000);
  };

  const handleSelectTemplate = (template: Template) => {
    // Clear existing tabs
    setTabs([]);
    
    // Create tabs from template files
    const newTabs: Tab[] = Object.entries(template.files).map(([fileName, fileData]) => ({
      id: fileName,
      name: fileName,
      content: fileData.content,
      language: fileData.language,
      isDirty: false,
    }));
    
    setTabs(newTabs);
    if (newTabs.length > 0) {
      setActiveTab(newTabs[0].id);
    }
    
    setShowTemplates(false);
  };

  const currentTab = tabs.find(tab => tab.id === activeTab);
  const lineCount = currentTab?.content.split('\n').length || 0;

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  if (showSuccess) {
    return <SuccessPage onContinue={() => setShowSuccess(false)} />;
  }

  if (!user) {
    return (
      <LandingPage 
        onLogin={handleLogin}
        onSignup={handleSignup}
        loading={authLoading}
        error={authError}
      />
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      <Toolbar 
        user={user}
        onShowPricing={() => setShowPricing(true)}
        onShowProfile={() => setShowProfile(true)}
        onToggleTerminal={() => setIsTerminalVisible(!isTerminalVisible)}
        onToggleCollaboration={() => setIsCollaborationVisible(!isCollaborationVisible)}
        onToggleMarketplace={() => setIsMarketplaceVisible(!isMarketplaceVisible)}
        onSave={handleSave}
        onRun={handleRun}
        onShowTemplates={() => setShowTemplates(true)}
        onShowMobilePreview={() => setShowMobilePreview(true)}
        onShowStripeTests={() => setShowStripeTests(true)}
      />
      
      <div className="flex flex-1 overflow-hidden">
        <div className="w-64 flex-shrink-0">
          <FileExplorer 
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
          />
        </div>
        
        <div className="flex-1 flex flex-col">
          <TabBar 
            tabs={tabs}
            activeTab={activeTab}
            onTabSelect={setActiveTab}
            onTabClose={handleTabClose}
          />
          
          <div className="flex-1 overflow-hidden">
            {currentTab ? (
              <CodeEditor 
                content={currentTab.content}
                onChange={handleCodeChange}
                language={currentTab.language}
                fileName={currentTab.name}
              />
            ) : (
              <div className="flex items-center justify-center h-full bg-gray-900 text-gray-400">
                <div className="text-center">
                  <div className="text-6xl mb-4">🦀</div>
                  <h2 className="text-2xl font-bold mb-2">Welcome to Rust Cloud IDE</h2>
                  <p className="mb-4">Select a file to start coding or choose a template to get started</p>
                  <button
                    onClick={() => setShowTemplates(true)}
                    className="px-6 py-3 bg-orange-600 hover:bg-orange-700 rounded-lg font-medium transition-colors"
                  >
                    Browse Templates
                  </button>
                </div>
              </div>
            )}
          </div>
          
          {isTerminalVisible && (
            <Terminal 
              isVisible={isTerminalVisible}
              onToggle={() => setIsTerminalVisible(!isTerminalVisible)}
            />
          )}
        </div>
        
        {isCollaborationVisible && (
          <CollaborationPanel 
            isVisible={isCollaborationVisible}
            onToggle={() => setIsCollaborationVisible(!isCollaborationVisible)}
            currentFile={currentTab?.name || ''}
            onUserCursorUpdate={() => {}}
          />
        )}
      </div>
      
      <StatusBar 
        currentFile={currentTab?.name || ''}
        language={currentTab?.language || ''}
        lineCount={lineCount}
        currentLine={1}
        buildStatus={buildStatus}
        collaboratorCount={3}
      />

      {/* Modals */}
      {showPricing && (
        <PricingPage onClose={() => setShowPricing(false)} />
      )}

      {showProfile && (
        <UserProfile 
          user={user}
          onClose={() => setShowProfile(false)}
          onLogout={handleLogout}
        />
      )}

      {showTemplates && (
        <ProjectTemplates 
          isVisible={showTemplates}
          onClose={() => setShowTemplates(false)}
          onSelectTemplate={handleSelectTemplate}
        />
      )}

      {showMobilePreview && (
        <MobilePreview 
          isVisible={showMobilePreview}
          onClose={() => setShowMobilePreview(false)}
          currentCode={currentTab?.content || ''}
          platform="flutter"
        />
      )}

      {isMarketplaceVisible && (
        <DeveloperMarketplace 
          isVisible={isMarketplaceVisible}
          onToggle={() => setIsMarketplaceVisible(!isMarketplaceVisible)}
        />
      )}

      {showStripeTests && (
        <StripeTestSuite 
          isVisible={showStripeTests}
          onClose={() => setShowStripeTests(false)}
        />
      )}

      <DemoMode 
        isActive={showDemo}
        onToggle={() => setShowDemo(!showDemo)}
      />
    </div>
  );
}

export default App;