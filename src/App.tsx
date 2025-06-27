import React, { useState } from 'react';
import { useEffect } from 'react';
import { supabase } from './lib/supabase';
import LoginPage from './components/auth/LoginPage';
import SignupPage from './components/auth/SignupPage';
import PricingPage from './components/PricingPage';
import SuccessPage from './components/SuccessPage';
import UserProfile from './components/UserProfile';
import FileExplorer from './components/FileExplorer';
import TabBar from './components/TabBar';
import CodeEditor from './components/CodeEditor';
import Terminal from './components/Terminal';
import Toolbar from './components/Toolbar';
import StatusBar from './components/StatusBar';
import ProjectTemplates from './components/ProjectTemplates';
import DemoMode from './components/DemoMode';
import CollaborationPanel from './components/CollaborationPanel';
import DeveloperMarketplace from './components/DeveloperMarketplace';
import MobilePreview from './components/MobilePreview';
import { User, CreditCard } from 'lucide-react';

interface Tab {
  id: string;
  name: string;
  content: string;
  isDirty: boolean;
  language: string;
}

function App() {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');
  const [showPricing, setShowPricing] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  
  const [tabs, setTabs] = useState<Tab[]>([
    {
      id: 'main.rs',
      name: 'main.rs',
      content: `fn main() {
    println!("Hello, Rust Cloud IDE!");
    
    let message = "Welcome to your Rust development environment";
    println!("{}", message);
    
    // Your code here
    let numbers = vec![1, 2, 3, 4, 5];
    let sum: i32 = numbers.iter().sum();
    println!("Sum of numbers: {}", sum);
}`,
      isDirty: false,
      language: 'rust',
    },
  ]);

  const [activeTab, setActiveTab] = useState('main.rs');
  const [selectedFile, setSelectedFile] = useState('rust-project/src/main.rs');
  const [terminalVisible, setTerminalVisible] = useState(false);
  const [buildStatus, setBuildStatus] = useState<'idle' | 'building' | 'success' | 'error'>('idle');
  const [templatesVisible, setTemplatesVisible] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [collaborationVisible, setCollaborationVisible] = useState(false);
  const [marketplaceVisible, setMarketplaceVisible] = useState(false);
  const [mobilePreviewVisible, setMobilePreviewVisible] = useState(false);
  const [collaborators, setCollaborators] = useState([
    {
      userId: '2',
      userName: 'Maria Rodriguez',
      line: 15,
      column: 8,
      color: '#EF4444'
    },
    {
      userId: '4',
      userName: 'Sarah Johnson',
      line: 23,
      column: 12,
      color: '#10B981'
    }
  ]);

  useEffect(() => {
    // Check if we're on the success page
    if (window.location.pathname === '/success') {
      setShowSuccess(true);
    }

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      setLoading(false);
    });

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleAuthSuccess = () => {
    // Auth state will be updated by the listener
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    setShowProfile(false);
  };

  const handleSuccessContinue = () => {
    setShowSuccess(false);
    // Clear the success URL
    window.history.replaceState({}, '', '/');
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="text-6xl mb-4">🦀</div>
          <div className="w-8 h-8 border-2 border-orange-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    );
  }

  if (showSuccess) {
    return <SuccessPage onContinue={handleSuccessContinue} />;
  }

  if (!user) {
    if (authMode === 'signup') {
      return (
        <SignupPage
          onSuccess={handleAuthSuccess}
          onSwitchToLogin={() => setAuthMode('login')}
        />
      );
    }
    return (
      <LoginPage
        onSuccess={handleAuthSuccess}
        onSwitchToSignup={() => setAuthMode('signup')}
      />
    );
  }

  const fileTemplates: Record<string, { content: string; language: string }> = {
    'main.rs': {
      content: `fn main() {
    println!("Hello, Mobile Rust!");
    
    // Initialize mobile app
    mobile_app::init();
    mobile_app::run();
}`,
      language: 'rust',
    },
    'lib.rs': {
      content: `pub mod mobile_utils;
pub mod ui;
pub mod platform;

use flutter_rust_bridge::frb;

#[frb(sync)]
pub fn init_mobile_app() -> String {
    "Mobile Rust App Initialized".to_string()
}

#[frb(sync)]
pub fn get_platform_info() -> String {
    platform::get_current_platform()
}

pub fn process_touch_event(x: f64, y: f64) -> bool {
    ui::handle_touch(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_init() {
        let result = init_mobile_app();
        assert!(result.contains("Initialized"));
    }
}`,
      language: 'rust',
    },
    'mobile_utils.rs': {
      content: `use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TouchEvent {
    pub x: f64,
    pub y: f64,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub platform: String,
    pub version: String,
    pub screen_width: u32,
    pub screen_height: u32,
}

pub fn format_device_info(info: &DeviceInfo) -> String {
    format!("{} {} ({}x{})", info.platform, info.version, info.screen_width, info.screen_height)
}

pub fn validate_touch_bounds(event: &TouchEvent, width: u32, height: u32) -> bool {
    event.x >= 0.0 && event.x <= width as f64 && 
    event.y >= 0.0 && event.y <= height as f64
}`,
      language: 'rust',
    },
    'ui.rs': {
      content: `use crate::mobile_utils::{TouchEvent, DeviceInfo};

pub struct MobileUI {
    device_info: DeviceInfo,
    touch_handlers: Vec<Box<dyn Fn(&TouchEvent) -> bool>>,
}

impl MobileUI {
    pub fn new(device_info: DeviceInfo) -> Self {
        Self {
            device_info,
            touch_handlers: Vec::new(),
        }
    }
    
    pub fn add_touch_handler<F>(&mut self, handler: F) 
    where 
        F: Fn(&TouchEvent) -> bool + 'static 
    {
        self.touch_handlers.push(Box::new(handler));
    }
    
    pub fn render_frame(&self) -> String {
        format!("Rendering frame for {}", self.device_info.platform)
    }
}

pub fn handle_touch(x: f64, y: f64) -> bool {
    let event = TouchEvent {
        x,
        y,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    };
    
    // Process touch event
    println!("Touch event at ({}, {})", event.x, event.y);
    true
}`,
      language: 'rust',
    },
    'platform.rs': {
      content: `#[cfg(target_os = "android")]
pub fn get_current_platform() -> String {
    "Android".to_string()
}

#[cfg(target_os = "ios")]
pub fn get_current_platform() -> String {
    "iOS".to_string()
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub fn get_current_platform() -> String {
    "Desktop".to_string()
}

pub fn get_platform_capabilities() -> Vec<String> {
    vec![
        "Touch Input".to_string(),
        "Accelerometer".to_string(),
        "Camera".to_string(),
        "GPS".to_string(),
        "Push Notifications".to_string(),
    ]
}

pub fn request_permission(permission: &str) -> bool {
    println!("Requesting permission: {}", permission);
    // Simulate permission granted
    true
}`,
      language: 'rust',
    },
    'neural_network.rs': {
      content: `use candle_core::{Device, Tensor, Result};
use candle_nn::{Module, VarBuilder, linear, Linear};

#[derive(Debug)]
pub struct NeuralNetwork {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
    device: Device,
}

impl NeuralNetwork {
    pub fn new(vs: VarBuilder, input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self> {
        let layer1 = linear(input_size, hidden_size, vs.pp("layer1"))?;
        let layer2 = linear(hidden_size, hidden_size, vs.pp("layer2"))?;
        let layer3 = linear(hidden_size, output_size, vs.pp("layer3"))?;
        
        Ok(Self {
            layer1,
            layer2,
            layer3,
            device: vs.device().clone(),
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.layer1.forward(input)?;
        let x = x.relu()?;
        let x = self.layer2.forward(&x)?;
        let x = x.relu()?;
        let output = self.layer3.forward(&x)?;
        Ok(output)
    }
}`,
      language: 'rust',
    },
    'computer_vision.rs': {
      content: `use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    imgcodecs::{imread, imwrite, IMREAD_COLOR},
    imgproc::{rectangle, put_text, FONT_HERSHEY_SIMPLEX, LINE_8},
    objdetect::CascadeClassifier,
    prelude::*,
    videoio::{VideoCapture, CAP_ANY},
    Result,
};

pub struct ComputerVisionPipeline {
    face_cascade: CascadeClassifier,
    object_detector: ObjectDetector,
}

impl ComputerVisionPipeline {
    pub fn new() -> Result<Self> {
        let face_cascade = CascadeClassifier::new("haarcascade_frontalface_alt.xml")?;
        let object_detector = ObjectDetector::new("yolov5s.onnx")?;
        
        Ok(Self {
            face_cascade,
            object_detector,
        })
    }
    
    pub fn process_frame(&mut self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        let mut detections = Vec::new();
        
        // Object detection
        let objects = self.object_detector.detect(frame)?;
        detections.extend(objects);
        
        // Face detection
        let faces = self.detect_faces(frame)?;
        detections.extend(faces);
        
        Ok(detections)
    }
}`,
      language: 'rust',
    },
    'reinforcement_learning.rs': {
      content: `use std::collections::VecDeque;
use rand::Rng;

pub struct DQNAgent {
    state_size: usize,
    action_size: usize,
    learning_rate: f64,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
    gamma: f64,
    replay_buffer: ReplayBuffer,
}

impl DQNAgent {
    pub fn new(state_size: usize, action_size: usize, learning_rate: f64, buffer_size: usize) -> Self {
        Self {
            state_size,
            action_size,
            learning_rate,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            gamma: 0.95,
            replay_buffer: ReplayBuffer::new(buffer_size),
        }
    }
    
    pub fn act(&mut self, state: &[f32]) -> usize {
        if rand::thread_rng().gen::<f64>() <= self.epsilon {
            // Random action (exploration)
            rand::thread_rng().gen_range(0..self.action_size)
        } else {
            // Greedy action (exploitation)
            self.predict_action(state)
        }
    }
    
    pub fn train(&mut self, batch_size: usize) -> Result<f64, String> {
        if self.replay_buffer.len() < batch_size {
            return Ok(0.0);
        }
        
        let batch = self.replay_buffer.sample(batch_size);
        // Training logic here
        Ok(0.0)
    }
}`,
      language: 'rust',
    },
    'nlp_transformer.rs': {
      content: `use std::collections::HashMap;

pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
}

pub struct MultiHeadAttention {
    config: TransformerConfig,
    query_weights: Vec<Vec<f32>>,
    key_weights: Vec<Vec<f32>>,
    value_weights: Vec<Vec<f32>>,
}

impl MultiHeadAttention {
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            query_weights: vec![vec![0.0; config.hidden_size]; config.hidden_size],
            key_weights: vec![vec![0.0; config.hidden_size]; config.hidden_size],
            value_weights: vec![vec![0.0; config.hidden_size]; config.hidden_size],
            config,
        }
    }
    
    pub fn forward(&self, hidden_states: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Attention mechanism implementation
        hidden_states.to_vec()
    }
}`,
      language: 'rust',
    },
    'MainActivity.kt': {
      content: `package com.example.mobilerustapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier

class MainActivity : ComponentActivity() {
    
    companion object {
        init {
            System.loadLibrary("mobile_rust_app")
        }
    }
    
    external fun initRustApp(): String
    external fun handleTouch(x: Float, y: Float): Boolean
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Rust backend
        val initResult = initRustApp()
        println("Rust init: $initResult")
        
        setContent {
            MaterialTheme {
                Surface(
    );
  }
}

class MobileHomePage extends StatefulWidget {
  const MobileHomePage({super.key});

  @override
  State<MobileHomePage> createState() => _MobileHomePageState();
}

class _MobileHomePageState extends State<MobileHomePage> {
  String _rustMessage = '';
  
  @override
  void initState() {
    super.initState();
    _initializeRust();
  }
  
  Future<void> _initializeRust() async {
    try {
      final message = await RustLib.instance.initMobileApp();
      setState(() {
        _rustMessage = message;
      });
    } catch (e) {
      print('Error initializing Rust: $e');
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Mobile Rust App'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text('🦀 Rust + Flutter'),
            const SizedBox(height: 20),
            Text(_rustMessage),
          ],
        ),
        {
          name: 'react-native',
          type: 'folder',
          isOpen: false,
          children: [
            { name: 'App.tsx', type: 'file' },
            { name: 'index.js', type: 'file' },
            { name: 'package.json', type: 'file' },
            { name: 'metro.config.js', type: 'file' },
            { name: 'babel.config.js', type: 'file' },
            {
              name: 'android',
              type: 'folder',
              isOpen: false,
              children: [
                { name: 'MainActivity.java', type: 'file' },
                { name: 'MainApplication.java', type: 'file' },
              ],
            },
            {
              name: 'ios',
              type: 'folder',
              isOpen: false,
              children: [
                { name: 'AppDelegate.mm', type: 'file' },
                { name: 'Info.plist', type: 'file' },
              ],
            },
          ],
        },
      ),
    );
  }
}`,
      language: 'dart',
    },
    'App.tsx': {
      content: `import React from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  useColorScheme,
  View,
  TouchableOpacity,
  Alert,
} from 'react-native';

import {
  Colors,
  DebugInstructions,
  Header,
  LearnMoreLinks,
  ReloadInstructions,
} from 'react-native/Libraries/NewAppScreen';

// Import Rust bridge
import { NativeModules } from 'react-native';
const { RustBridge } = NativeModules;

function App(): JSX.Element {
  const isDarkMode = useColorScheme() === 'dark';

  const backgroundStyle = {
    backgroundColor: isDarkMode ? Colors.darker : Colors.lighter,
  };

  const handleRustCall = async () => {
    try {
      const result = await RustBridge.initMobileApp();
      Alert.alert('Rust Response', result);
    } catch (error) {
      Alert.alert('Error', 'Failed to call Rust function');
    }
  };

  const handleTouchEvent = async (x: number, y: number) => {
    try {
      const result = await RustBridge.handleTouch(x, y);
      console.log('Touch handled:', result);
    } catch (error) {
      console.error('Touch handling failed:', error);
    }
  };

  return (
    <SafeAreaView style={backgroundStyle}>
      <StatusBar
        barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        backgroundColor={backgroundStyle.backgroundColor}
      />
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        style={backgroundStyle}>
        <Header />
        <View
          style={{
            backgroundColor: isDarkMode ? Colors.black : Colors.white,
          }}>
          <View style={styles.sectionContainer}>
            <Text style={[styles.sectionTitle, { color: isDarkMode ? Colors.white : Colors.black }]}>
              🦀 Rust + React Native
            </Text>
            <Text style={[styles.sectionDescription, { color: isDarkMode ? Colors.light : Colors.dark }]}>
              This app demonstrates integration between React Native and Rust backend.
            </Text>
            <TouchableOpacity
              style={styles.button}
              onPress={handleRustCall}>
              <Text style={styles.buttonText}>Call Rust Function</Text>
            </TouchableOpacity>
          </View>
          
          <View style={styles.sectionContainer}>
            <Text style={[styles.sectionTitle, { color: isDarkMode ? Colors.white : Colors.black }]}>
              Touch Events
            </Text>
            <TouchableOpacity
              style={styles.touchArea}
              onPress={(event) => {
                const { locationX, locationY } = event.nativeEvent;
                handleTouchEvent(locationX, locationY);
              }}>
              <Text style={styles.touchText}>Tap here to test touch events</Text>
            </TouchableOpacity>
          </View>
          
          <LearnMoreLinks />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
  },
  sectionDescription: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: '400',
  },
  button: {
    backgroundColor: '#FF6B35',
    padding: 12,
    borderRadius: 8,
    marginTop: 16,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  touchArea: {
    backgroundColor: '#E8F4FD',
    padding: 20,
    borderRadius: 8,
    marginTop: 16,
    borderWidth: 2,
    borderColor: '#007AFF',
    borderStyle: 'dashed',
  },
  touchText: {
    color: '#007AFF',
    fontSize: 16,
    textAlign: 'center',
  },
});

export default App;`,
      language: 'typescript',
    },
    'index.js': {
      content: `import { AppRegistry } from 'react-native';
import App from './App';
import { name as appName } from './package.json';

AppRegistry.registerComponent(appName, () => App);`,
      language: 'javascript',
    },
    'package.json': {
      content: `{
  "name": "MobileRustApp",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "android": "react-native run-android",
    "ios": "react-native run-ios",
    "lint": "eslint .",
    "start": "react-native start",
    "test": "jest",
    "build:rust": "cargo build --release",
    "build:android": "cd android && ./gradlew assembleRelease",
    "build:ios": "cd ios && xcodebuild -workspace MobileRustApp.xcworkspace -scheme MobileRustApp -configuration Release"
  },
  "dependencies": {
    "react": "18.2.0",
    "react-native": "0.72.6",
    "@react-native-community/cli": "^11.3.6"
  },
  "devDependencies": {
    "@babel/core": "^7.20.0",
    "@babel/preset-env": "^7.20.0",
    "@babel/runtime": "^7.20.0",
    "@react-native/eslint-config": "^0.72.2",
    "@react-native/metro-config": "^0.72.11",
    "@tsconfig/react-native": "^3.0.0",
    "@types/react": "^18.0.24",
    "@types/react-test-renderer": "^18.0.0",
    "babel-jest": "^29.2.1",
    "eslint": "^8.19.0",
    "jest": "^29.2.1",
    "metro-react-native-babel-preset": "0.76.8",
    "prettier": "^2.4.1",
    "react-test-renderer": "18.2.0",
    "typescript": "4.8.4"
  },
  "engines": {
    "node": ">=16"
  }
}`,
      language: 'json',
    },
    'metro.config.js': {
      content: `const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://facebook.github.io/metro/docs/configuration
 *
 * @type {import('metro-config').MetroConfig}
 */
const config = {};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);`,
      language: 'javascript',
    },
    'babel.config.js': {
      content: `module.exports = {
  presets: ['module:metro-react-native-babel-preset'],
  plugins: [
    [
      'module-resolver',
      {
        root: ['./src'],
        extensions: ['.ios.js', '.android.js', '.js', '.ts', '.tsx', '.json'],
        alias: {
          '@': './src',
        },
      },
    ],
  ],
};`,
      language: 'javascript',
    },
    'MainActivity.java': {
      content: `package com.mobilerustapp;

import com.facebook.react.ReactActivity;
import com.facebook.react.ReactActivityDelegate;
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint;
import com.facebook.react.defaults.DefaultReactActivityDelegate;

public class MainActivity extends ReactActivity {

  /**
   * Returns the name of the main component registered from JavaScript. This is used to schedule
   * rendering of the component.
   */
  @Override
  protected String getMainComponentName() {
    return "MobileRustApp";
  }

  /**
   * Returns the instance of the {@link ReactActivityDelegate}. Here we use a util class {@link
   * DefaultReactActivityDelegate} which allows you to easily enable Fabric and Concurrent React
   * (aka React 18) with two boolean flags.
   */
  @Override
  protected ReactActivityDelegate createReactActivityDelegate() {
    return new DefaultReactActivityDelegate(
        this,
        getMainComponentName(),
        // If you opted-in for the New Architecture, we enable the Fabric Renderer.
        DefaultNewArchitectureEntryPoint.getFabricEnabled());
  }

  static {
    // Load the Rust native library
    System.loadLibrary("mobile_rust_app");
  }
}`,
      language: 'java',
    },
    'MainApplication.java': {
      content: `package com.mobilerustapp;

import android.app.Application;
import com.facebook.react.PackageList;
import com.facebook.react.ReactApplication;
import com.facebook.react.ReactNativeHost;
import com.facebook.react.ReactPackage;
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint;
import com.facebook.react.defaults.DefaultReactNativeHost;
import com.facebook.soloader.SoLoader;
import java.util.List;

public class MainApplication extends Application implements ReactApplication {

  private final ReactNativeHost mReactNativeHost =
      new DefaultReactNativeHost(this) {
        @Override
        public boolean getUseDeveloperSupport() {
          return BuildConfig.DEBUG;
        }

        @Override
        protected List<ReactPackage> getPackages() {
          @SuppressWarnings("UnnecessaryLocalVariable")
          List<ReactPackage> packages = new PackageList(this).getPackages();
          // Add the Rust bridge package
          packages.add(new RustBridgePackage());
          return packages;
        }

        @Override
        protected String getJSMainModuleName() {
          return "index";
        }

        @Override
        protected boolean isNewArchEnabled() {
          return DefaultNewArchitectureEntryPoint.getNewArchEnabled();
        }

        @Override
        protected Boolean isHermesEnabled() {
          return DefaultNewArchitectureEntryPoint.getHermesEnabled();
        }
      };

  @Override
  public ReactNativeHost getReactNativeHost() {
    return mReactNativeHost;
  }

  @Override
  public void onCreate() {
    super.onCreate();
    SoLoader.init(this, /* native exopackage */ false);
    if (DefaultNewArchitectureEntryPoint.getNewArchEnabled()) {
      // If you opted-in for the New Architecture, we load the native entry point for this app.
      DefaultNewArchitectureEntryPoint.load();
    }
    ReactNativeFlipper.initializeFlipper(this, getReactNativeHost().getReactInstanceManager());
  }
}`,
      language: 'java',
    },
    'AppDelegate.mm': {
      content: `#import "AppDelegate.h"

#import <React/RCTBundleURLProvider.h>

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
  self.moduleName = @"MobileRustApp";
  // You can add your custom initial props in the dictionary below.
  // They will be passed down to the ViewController used by React Native.
  self.initialProps = @{};

  return [super application:application didFinishLaunchingWithOptions:launchOptions];
}

- (NSURL *)sourceURLForBridge:(RCTBridge *)bridge
{
#if DEBUG
  return [[RCTBundleURLProvider sharedSettings] jsBundleURLForBundleRoot:@"index"];
#else
  return [[NSBundle mainBundle] URLForResource:@"main" withExtension:@"jsbundle"];
#endif
}

@end`,
      language: 'objective-c',
    },
    'Info.plist': {
      content: `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleDisplayName</key>
	<string>Mobile Rust App</string>
	<key>CFBundleExecutable</key>
	<string>$(EXECUTABLE_NAME)</string>
	<key>CFBundleIdentifier</key>
	<string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>$(PRODUCT_NAME)</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>$(MARKETING_VERSION)</string>
	<key>CFBundleSignature</key>
	<string>????</string>
	<key>CFBundleVersion</key>
	<string>$(CURRENT_PROJECT_VERSION)</string>
	<key>LSRequiresIPhoneOS</key>
	<true/>
	<key>NSAppTransportSecurity</key>
	<dict>
		<key>NSExceptionDomains</key>
		<dict>
			<key>localhost</key>
			<dict>
				<key>NSExceptionAllowsInsecureHTTPLoads</key>
				<true/>
			</dict>
		</dict>
	</dict>
	<key>NSLocationWhenInUseUsageDescription</key>
	<string>This app needs access to location when open.</string>
	<key>UILaunchStoryboardName</key>
	<string>LaunchScreen</string>
	<key>UIRequiredDeviceCapabilities</key>
	<array>
		<string>armv7</string>
	</array>
	<key>UISupportedInterfaceOrientations</key>
	<array>
		<string>UIInterfaceOrientationPortrait</string>
		<string>UIInterfaceOrientationLandscapeLeft</string>
		<string>UIInterfaceOrientationLandscapeRight</string>
	</array>
	<key>UIViewControllerBasedStatusBarAppearance</key>
	<false/>
</dict>
</plist>`,
      language: 'xml',
    },
    'main.py': {
      content: `#!/usr/bin/env python3
"""
AI/ML Research Environment
High-performance Python for machine learning and data science
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def train_model(model, train_loader, criterion, optimizer, epochs=100):
    """Train the neural network model"""
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}')
    
    return losses

def evaluate_model(model, test_loader):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            actuals.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(actuals, predictions)
    report = classification_report(actuals, predictions)
    
    return accuracy, report, predictions, actuals

def visualize_results(losses, predictions, actuals):
    """Create visualizations for model performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(losses)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actuals, predictions)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax2)
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("🐍 Python AI/ML Research Environment")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate sample data (replace with your dataset)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 3, 1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = NeuralNetwork(input_size=10, hidden_size=64, output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model...")
    losses = train_model(model, train_loader, criterion, optimizer, epochs=50)
    
    print("\\nEvaluating model...")
    accuracy, report, predictions, actuals = evaluate_model(model, test_loader)
    
    print(f"\\nTest Accuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(report)
    
    # Visualize results
    visualize_results(losses, predictions, actuals)
    
    print("\\n✅ Training and evaluation complete!")

if __name__ == "__main__":
    main()`,
      language: 'python',
    },
    'requirements.txt': {
      content: `# Core ML/AI Libraries
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
tensorflow>=2.13.0
keras>=2.13.0

# Data Science & Analysis
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Computer Vision
opencv-python>=4.8.0
Pillow>=10.0.0
albumentations>=1.3.0

# Natural Language Processing
transformers>=4.30.0
tokenizers>=0.13.0
datasets>=2.13.0
nltk>=3.8.0
spacy>=3.6.0

# Deep Learning Utilities
lightning>=2.0.0
wandb>=0.15.0
tensorboard>=2.13.0

# Jupyter & Development
jupyter>=1.0.0
ipykernel>=6.23.0
notebook>=6.5.0
jupyterlab>=4.0.0

# Data Processing
h5py>=3.9.0
tqdm>=4.65.0
requests>=2.31.0

# Reinforcement Learning
gym>=0.29.0
stable-baselines3>=2.0.0

# Optimization & Hyperparameter Tuning
optuna>=3.2.0
ray[tune]>=2.5.0

# Model Deployment
fastapi>=0.100.0
uvicorn>=0.22.0
streamlit>=1.24.0

# Distributed Computing
dask>=2023.6.0
ray>=2.5.0

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0

# Time Series
prophet>=1.1.0
statsmodels>=0.14.0

# Graph Neural Networks
torch-geometric>=2.3.0
networkx>=3.1.0

# Quantum Computing
qiskit>=0.43.0
cirq>=1.1.0`,
      language: 'text',
    },
    'computer_vision.py': {
      content: `#!/usr/bin/env python3
"""
Computer Vision Pipeline
Advanced computer vision with PyTorch and OpenCV
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ObjectDetector:
    """Advanced object detection using pre-trained models"""
    
    def __init__(self, model_name: str = 'yolov5s'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_name)
        self.classes = self._load_classes()
        
    def _load_model(self, model_name: str):
        """Load pre-trained detection model"""
        if 'yolo' in model_name.lower():
            model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            model.to(self.device)
            return model
        else:
            # Load other models (RCNN, SSD, etc.)
            pass
    
    def _load_classes(self) -> List[str]:
        """Load class names for COCO dataset"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee'
        ]
    
    def detect(self, image: np.ndarray, confidence: float = 0.5) -> List[dict]:
        """Detect objects in image"""
        results = self.model(image)
        detections = []
        
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > confidence:
                x1, y1, x2, y2 = map(int, box)
                class_name = self.classes[int(cls)]
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf),
                    'class': class_name,
                    'class_id': int(cls)
                })
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image

class ImageClassifier:
    """Image classification using transfer learning"""
    
    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model(num_classes, pretrained)
        self.transform = self._get_transforms()
        
    def _create_model(self, num_classes: int, pretrained: bool) -> nn.Module:
        """Create ResNet model with custom classifier"""
        model = models.resnet50(pretrained=pretrained)
        
        # Modify final layer for custom number of classes
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        model.to(self.device)
        return model
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """Predict class for single image"""
        self.model.eval()
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        return predicted_class, confidence

class FaceDetector:
    """Face detection and recognition pipeline"""
    
    def __init__(self):
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces.tolist()
    
    def detect_eyes(self, image: np.ndarray, face_region: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Detect eyes within face region"""
        x, y, w, h = face_region
        roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        # Adjust coordinates relative to full image
        adjusted_eyes = [(ex + x, ey + y, ew, eh) for ex, ey, ew, eh in eyes]
        return adjusted_eyes
    
    def draw_detections(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                       eyes: Optional[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
        """Draw face and eye detections on image"""
        result = image.copy()
        
        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Draw eye rectangles
        if eyes:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(result, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        return result

def process_video_stream(detector: ObjectDetector, source: int = 0):
    """Process video stream with real-time object detection"""
    cap = cv2.VideoCapture(source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detector.detect(frame, confidence=0.5)
        
        # Visualize detections
        result_frame = detector.visualize_detections(frame, detections)
        
        # Display frame
        cv2.imshow('Object Detection', result_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main computer vision pipeline demonstration"""
    print("🔍 Computer Vision Pipeline")
    print("=" * 40)
    
    # Initialize detectors
    object_detector = ObjectDetector()
    face_detector = FaceDetector()
    classifier = ImageClassifier(num_classes=10)  # Custom dataset
    
    print("✅ Models loaded successfully!")
    print("📹 Starting video processing...")
    print("Press 'q' to quit")
    
    # Process video stream (uncomment to run)
    # process_video_stream(object_detector)
    
    print("🎯 Computer vision pipeline ready!")

if __name__ == "__main__":
    main()`,
      language: 'python',
    },
    'nlp_transformer.py': {
      content: `#!/usr/bin/env python3
"""
Natural Language Processing with Transformers
Advanced NLP using Hugging Face Transformers
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, AutoModelForTokenClassification,
    pipeline, Trainer, TrainingArguments
)
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class TextClassifier:
    """Text classification using pre-trained transformers"""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        
    def preprocess_text(self, texts: List[str], max_length: int = 512) -> Dict:
        """Tokenize and encode texts"""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment/classification for texts"""
        self.model.eval()
        inputs = self.preprocess_text(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        results = []
        for i, pred in enumerate(predictions):
            predicted_class = torch.argmax(pred).item()
            confidence = pred[predicted_class].item()
            results.append({
                'text': texts[i],
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': pred.cpu().numpy().tolist()
            })
        
        return results
    
    def fine_tune(self, train_texts: List[str], train_labels: List[int], 
                  val_texts: List[str], val_labels: List[int], 
                  epochs: int = 3, batch_size: int = 16):
        """Fine-tune model on custom dataset"""
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, padding=True)
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train model
        trainer.train()
        
        return trainer

class QuestionAnswering:
    """Question answering using BERT-based models"""
    
    def __init__(self, model_name: str = "bert-large-uncased-whole-word-masking-finetuned-squad"):
        self.qa_pipeline = pipeline("question-answering", model=model_name)
    
    def answer_question(self, context: str, question: str) -> Dict:
        """Answer question based on given context"""
        result = self.qa_pipeline(question=question, context=context)
        return {
            'question': question,
            'context': context,
            'answer': result['answer'],
            'confidence': result['score'],
            'start': result['start'],
            'end': result['end']
        }
    
    def batch_qa(self, contexts: List[str], questions: List[str]) -> List[Dict]:
        """Process multiple question-answer pairs"""
        results = []
        for context, question in zip(contexts, questions):
            result = self.answer_question(context, question)
            results.append(result)
        return results

class NamedEntityRecognition:
    """Named Entity Recognition using transformers"""
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        self.ner_pipeline = pipeline("ner", 
                                   model=model_name, 
                                   tokenizer=model_name,
                                   aggregation_strategy="simple")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text"""
        entities = self.ner_pipeline(text)
        
        processed_entities = []
        for entity in entities:
            processed_entities.append({
                'text': entity['word'],
                'label': entity['entity_group'],
                'confidence': entity['score'],
                'start': entity['start'],
                'end': entity['end']
            })
        
        return processed_entities
    
    def visualize_entities(self, text: str, entities: List[Dict]) -> str:
        """Create HTML visualization of entities"""
        html = text
        
        # Sort entities by start position (reverse order for replacement)
        entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        colors = {
            'PER': '#FFB6C1',  # Light pink for persons
            'ORG': '#98FB98',  # Pale green for organizations
            'LOC': '#87CEEB',  # Sky blue for locations
            'MISC': '#DDA0DD'  # Plum for miscellaneous
        }
        
        for entity in entities_sorted:
            start, end = entity['start'], entity['end']
            label = entity['label']
            color = colors.get(label, '#FFFFE0')  # Light yellow as default
            
            replacement = f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{label}: {entity["confidence"]:.2f}">{entity["text"]}</span>'
            html = html[:start] + replacement + html[end:]
        
        return html

class TextSummarization:
    """Text summarization using transformer models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> Dict:
        """Generate summary of input text"""
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        
        return {
            'original_text': text,
            'summary': summary[0]['summary_text'],
            'compression_ratio': len(summary[0]['summary_text']) / len(text),
            'original_length': len(text),
            'summary_length': len(summary[0]['summary_text'])
        }
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict]:
        """Summarize multiple texts"""
        return [self.summarize(text, **kwargs) for text in texts]

class SentimentAnalysis:
    """Advanced sentiment analysis with emotion detection"""
    
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.emotion_pipeline = pipeline("text-classification", 
                                       model="j-hartmann/emotion-english-distilroberta-base")
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment of texts"""
        sentiments = self.sentiment_pipeline(texts)
        emotions = self.emotion_pipeline(texts)
        
        results = []
        for i, (sentiment, emotion) in enumerate(zip(sentiments, emotions)):
            results.append({
                'text': texts[i],
                'sentiment': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'emotion': emotion['label'],
                'emotion_score': emotion['score']
            })
        
        return results
    
    def visualize_sentiment_distribution(self, results: List[Dict]):
        """Create visualization of sentiment distribution"""
        sentiments = [r['sentiment'] for r in results]
        emotions = [r['emotion'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sentiment distribution
        sentiment_counts = pd.Series(sentiments).value_counts()
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax1.set_title('Sentiment Distribution')
        
        # Emotion distribution
        emotion_counts = pd.Series(emotions).value_counts()
        ax2.bar(emotion_counts.index, emotion_counts.values)
        ax2.set_title('Emotion Distribution')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main NLP pipeline demonstration"""
    print("🤖 Advanced NLP with Transformers")
    print("=" * 50)
    
    # Sample texts for demonstration
    sample_texts = [
        "I love this new AI model! It's incredibly accurate and fast.",
        "The weather today is terrible, I'm feeling quite sad.",
        "Apple Inc. is planning to release a new iPhone next year in California.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    # Initialize NLP models
    print("🔄 Loading models...")
    
    classifier = TextClassifier()
    qa_model = QuestionAnswering()
    ner_model = NamedEntityRecognition()
    summarizer = TextSummarization()
    sentiment_analyzer = SentimentAnalysis()
    
    print("✅ Models loaded successfully!")
    
    # Demonstrate sentiment analysis
    print("\\n📊 Sentiment Analysis:")
    sentiment_results = sentiment_analyzer.analyze_sentiment(sample_texts)
    for result in sentiment_results:
        print(f"Text: {result['text'][:50]}...")
        print(f"Sentiment: {result['sentiment']} ({result['sentiment_score']:.3f})")
        print(f"Emotion: {result['emotion']} ({result['emotion_score']:.3f})\\n")
    
    # Demonstrate NER
    print("🏷️ Named Entity Recognition:")
    ner_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    entities = ner_model.extract_entities(ner_text)
    for entity in entities:
        print(f"Entity: {entity['text']} | Type: {entity['label']} | Confidence: {entity['confidence']:.3f}")
    
    # Demonstrate Question Answering
    print("\\n❓ Question Answering:")
    context = "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017."
    question = "When was the transformer architecture introduced?"
    qa_result = qa_model.answer_question(context, question)
    print(f"Question: {qa_result['question']}")
    print(f"Answer: {qa_result['answer']} (Confidence: {qa_result['confidence']:.3f})")
    
    print("\\n🎯 NLP pipeline demonstration complete!")

if __name__ == "__main__":
    main()`,
      language: 'python',
    },
    'reinforcement_learning.py': {
      content: `#!/usr/bin/env python3
"""
Reinforcement Learning Framework
Advanced RL algorithms with PyTorch and Stable-Baselines3
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional
import stable_baselines3 as sb3
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import wandb

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    """Deep Q-Network for value-based RL"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 1e-3, 
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size: int = 32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        experiences = self.memory.sample(batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

class PolicyNetwork(nn.Module):
    """Policy network for actor-critic methods"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        
        return policy, value

class A2CAgent:
    """Advantage Actor-Critic Agent"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 1e-3, gamma: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.network = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.log_probs = []
        self.values = []
        self.rewards = []
        
    def act(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Choose action and return log probability and value"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy, value = self.network(state_tensor)
        
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def update(self):
        """Update policy and value networks"""
        returns = []
        G = 0
        
        # Calculate returns
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        
        for log_prob, value, G in zip(self.log_probs, self.values, returns):
            advantage = G - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.mse_loss(value, torch.tensor([[G]]).to(self.device)))
        
        total_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        
        return total_loss.item()

class RLTrainer:
    """Reinforcement Learning Training Framework"""
    
    def __init__(self, env_name: str = 'CartPole-v1'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
    def train_dqn(self, episodes: int = 1000, target_update: int = 100) -> List[float]:
        """Train DQN agent"""
        agent = DQNAgent(self.state_size, self.action_size)
        scores = []
        losses = []
        
        for episode in range(episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Handle new gym API
            
            total_reward = 0
            episode_losses = []
            
            while True:
                action = agent.act(state)
                result = self.env.step(action)
                
                if len(result) == 5:  # New gym API
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:  # Old gym API
                    next_state, reward, done, _ = result
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Train the agent
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
                
                if done:
                    break
            
            scores.append(total_reward)
            if episode_losses:
                losses.append(np.mean(episode_losses))
            
            # Update target network
            if episode % target_update == 0:
                agent.update_target_network()
            
            # Print progress
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        return scores
    
    def train_a2c(self, episodes: int = 1000) -> List[float]:
        """Train A2C agent"""
        agent = A2CAgent(self.state_size, self.action_size)
        scores = []
        
        for episode in range(episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            total_reward = 0
            
            while True:
                action, log_prob, value = agent.act(state)
                result = self.env.step(action)
                
                if len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = result
                
                agent.log_probs.append(log_prob)
                agent.values.append(value)
                agent.rewards.append(reward)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Update agent
            loss = agent.update()
            scores.append(total_reward)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode {episode}, Average Score: {avg_score:.2f}")
        
        return scores
    
    def train_stable_baselines(self, algorithm: str = 'PPO', total_timesteps: int = 100000):
        """Train using Stable-Baselines3"""
        
        # Create vectorized environment
        env = make_vec_env(self.env_name, n_envs=4)
        
        # Choose algorithm
        if algorithm == 'PPO':
            model = PPO('MlpPolicy', env, verbose=1)
        elif algorithm == 'DQN':
            model = DQN('MlpPolicy', env, verbose=1)
        elif algorithm == 'A2C':
            model = A2C('MlpPolicy', env, verbose=1)
        elif algorithm == 'SAC':
            model = SAC('MlpPolicy', env, verbose=1)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train the model
        model.learn(total_timesteps=total_timesteps)
        
        return model
    
    def evaluate_agent(self, model, episodes: int = 100) -> float:
        """Evaluate trained agent"""
        total_rewards = []
        
        for _ in range(episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            total_reward = 0
            
            while True:
                if hasattr(model, 'predict'):  # Stable-Baselines3 model
                    action, _ = model.predict(state, deterministic=True)
                else:  # Custom agent
                    action = model.act(state)
                
                result = self.env.step(action)
                
                if len(result) == 5:
                    state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    state, reward, done, _ = result
                
                total_reward += reward
                
                if done:
                    break
            
            total_rewards.append(total_reward)
        
        return np.mean(total_rewards)
    
    def plot_training_results(self, scores: List[float], window: int = 100):
        """Plot training progress"""
        plt.figure(figsize=(12, 6))
        
        # Plot raw scores
        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.title('Training Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # Plot moving average
        plt.subplot(1, 2, 2)
        moving_avg = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
        plt.plot(moving_avg)
        plt.title(f'Moving Average (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main reinforcement learning demonstration"""
    print("🎮 Reinforcement Learning Framework")
    print("=" * 50)
    
    # Initialize trainer
    trainer = RLTrainer('CartPole-v1')
    
    print(f"Environment: {trainer.env_name}")
    print(f"State size: {trainer.state_size}")
    print(f"Action size: {trainer.action_size}")
    
    # Train DQN agent
    print("\\n🧠 Training DQN Agent...")
    dqn_scores = trainer.train_dqn(episodes=500)
    
    # Train A2C agent
    print("\\n🎯 Training A2C Agent...")
    a2c_scores = trainer.train_a2c(episodes=500)
    
    # Train with Stable-Baselines3
    print("\\n🚀 Training with Stable-Baselines3 (PPO)...")
    ppo_model = trainer.train_stable_baselines('PPO', total_timesteps=50000)
    
    # Evaluate models
    print("\\n📊 Evaluating models...")
    ppo_performance = trainer.evaluate_agent(ppo_model)
    print(f"PPO Average Score: {ppo_performance:.2f}")
    
    # Plot results
    trainer.plot_training_results(dqn_scores)
    trainer.plot_training_results(a2c_scores)
    
    print("\\n✅ Reinforcement learning training complete!")

if __name__ == "__main__":
    main()`,
      language: 'python',
    },
    'data_science.py': {
      content: `#!/usr/bin/env python3
"""
Data Science & Analytics Pipeline
Comprehensive data analysis with pandas, numpy, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Comprehensive data analysis and visualization toolkit"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.original_data = data.copy()
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
    def basic_info(self) -> Dict:
        """Get basic information about the dataset"""
        info = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns
        }
        
        return info
    
    def descriptive_statistics(self) -> pd.DataFrame:
        """Generate comprehensive descriptive statistics"""
        desc_stats = self.data.describe(include='all')
        
        # Add additional statistics
        additional_stats = pd.DataFrame(index=['skewness', 'kurtosis', 'variance'])
        
        for col in self.numeric_columns:
            additional_stats.loc['skewness', col] = self.data[col].skew()
            additional_stats.loc['kurtosis', col] = self.data[col].kurtosis()
            additional_stats.loc['variance', col] = self.data[col].var()
        
        return pd.concat([desc_stats, additional_stats])
    
    def detect_outliers(self, method: str = 'iqr') -> Dict[str, List]:
        """Detect outliers using IQR or Z-score method"""
        outliers = {}
        
        for col in self.numeric_columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = self.data[(self.data[col] < lower_bound) | 
                                          (self.data[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                outlier_indices = self.data[z_scores > 3].index.tolist()
            
            outliers[col] = outlier_indices
        
        return outliers
    
    def correlation_analysis(self) -> pd.DataFrame:
        """Perform correlation analysis"""
        return self.data[self.numeric_columns].corr()
    
    def create_visualizations(self):
        """Create comprehensive data visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Distribution plots for numeric variables
        n_numeric = len(self.numeric_columns)
        if n_numeric > 0:
            for i, col in enumerate(self.numeric_columns[:6]):  # Limit to first 6
                plt.subplot(4, 3, i + 1)
                self.data[col].hist(bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
        
        # 2. Correlation heatmap
        if len(self.numeric_columns) > 1:
            plt.subplot(4, 3, 7)
            corr_matrix = self.correlation_analysis()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('Correlation Matrix')
        
        # 3. Box plots for outlier detection
        if len(self.numeric_columns) > 0:
            plt.subplot(4, 3, 8)
            self.data[self.numeric_columns[:4]].boxplot()
            plt.title('Box Plots for Outlier Detection')
            plt.xticks(rotation=45)
        
        # 4. Missing values heatmap
        plt.subplot(4, 3, 9)
        sns.heatmap(self.data.isnull(), cbar=True, yticklabels=False, 
                   cmap='viridis', xticklabels=True)
        plt.title('Missing Values Pattern')
        plt.xticks(rotation=45)
        
        # 5. Categorical variable counts
        if len(self.categorical_columns) > 0:
            plt.subplot(4, 3, 10)
            col = self.categorical_columns[0]
            value_counts = self.data[col].value_counts().head(10)
            value_counts.plot(kind='bar')
            plt.title(f'Top 10 Categories in {col}')
            plt.xticks(rotation=45)
        
        # 6. Pairplot for numeric variables (sample)
        if len(self.numeric_columns) >= 2:
            plt.subplot(4, 3, 11)
            sample_cols = self.numeric_columns[:3]
            for i, col1 in enumerate(sample_cols):
                for j, col2 in enumerate(sample_cols):
                    if i < j:
                        plt.scatter(self.data[col1], self.data[col2], alpha=0.5)
                        plt.xlabel(col1)
                        plt.ylabel(col2)
                        plt.title(f'{col1} vs {col2}')
                        break
                if i < len(sample_cols) - 1:
                    break
        
        plt.tight_layout()
        plt.show()
    
    def interactive_dashboard(self):
        """Create interactive dashboard with Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution', 'Correlation Heatmap', 
                          'Scatter Plot', 'Box Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if len(self.numeric_columns) > 0:
            # Distribution plot
            col = self.numeric_columns[0]
            fig.add_trace(
                go.Histogram(x=self.data[col], name=f'{col} Distribution'),
                row=1, col=1
            )
            
            # Correlation heatmap
            if len(self.numeric_columns) > 1:
                corr_matrix = self.correlation_analysis()
                fig.add_trace(
                    go.Heatmap(z=corr_matrix.values, 
                             x=corr_matrix.columns, 
                             y=corr_matrix.columns,
                             colorscale='RdBu',
                             name='Correlation'),
                    row=1, col=2
                )
            
            # Scatter plot
            if len(self.numeric_columns) >= 2:
                fig.add_trace(
                    go.Scatter(x=self.data[self.numeric_columns[0]], 
                             y=self.data[self.numeric_columns[1]],
                             mode='markers',
                             name='Scatter Plot'),
                    row=2, col=1
                )
            
            # Box plot
            fig.add_trace(
                go.Box(y=self.data[self.numeric_columns[0]], 
                      name=f'{self.numeric_columns[0]} Box Plot'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Interactive Data Analysis Dashboard")
        fig.show()

class MLPipeline:
    """Machine Learning Pipeline for automated model training and evaluation"""
    
    def __init__(self, data: pd.DataFrame, target_column: str):
        self.data = data.copy()
        self.target_column = target_column
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """Prepare data for machine learning"""
        
        # Separate features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        
        # Handle categorical variables
        categorical_columns = self.X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
        
        # Handle missing values
        self.X = self.X.fillna(self.X.mean())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        print(f"Data prepared: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
    
    def train_classification_models(self):
        """Train multiple classification models"""
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Logistic Regression':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
    
    def train_regression_models(self):
        """Train multiple regression models"""
        
        models = {
            'Linear Regression': LinearRegression(),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Linear Regression':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            self.models[name] = model
            self.results[name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    def perform_clustering(self, n_clusters: int = 3):
        """Perform K-means clustering"""
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.X_train_scaled)
        
        # Add cluster labels to training data
        self.X_train['cluster'] = clusters
        
        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_train_scaled)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'K-means Clustering (k={n_clusters}) - PCA Visualization')
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.show()
        
        return clusters, kmeans
    
    def feature_importance_analysis(self):
        """Analyze feature importance from tree-based models"""
        
        tree_models = ['Random Forest', 'Gradient Boosting']
        
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = self.X.columns
                    
                    # Create feature importance plot
                    plt.figure(figsize=(10, 6))
                    indices = np.argsort(importances)[::-1]
                    
                    plt.bar(range(len(importances)), importances[indices])
                    plt.title(f'Feature Importance - {model_name}')
                    plt.xlabel('Features')
                    plt.ylabel('Importance')
                    plt.xticks(range(len(importances)), 
                             [feature_names[i] for i in indices], rotation=45)
                    plt.tight_layout()
                    plt.show()
    
    def model_comparison(self):
        """Compare model performances"""
        
        if not self.results:
            print("No models trained yet!")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, results in self.results.items():
            if 'accuracy' in results:  # Classification
                comparison_data.append({
                    'Model': model_name,
                    'Metric': 'Accuracy',
                    'Value': results['accuracy']
                })
            elif 'r2' in results:  # Regression
                comparison_data.append({
                    'Model': model_name,
                    'Metric': 'R²',
                    'Value': results['r2']
                })
                comparison_data.append({
                    'Model': model_name,
                    'Metric': 'RMSE',
                    'Value': results['rmse']
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Create comparison plot
            plt.figure(figsize=(12, 6))
            
            for metric in df_comparison['Metric'].unique():
                metric_data = df_comparison[df_comparison['Metric'] == metric]
                plt.subplot(1, len(df_comparison['Metric'].unique()), 
                           list(df_comparison['Metric'].unique()).index(metric) + 1)
                plt.bar(metric_data['Model'], metric_data['Value'])
                plt.title(f'Model Comparison - {metric}')
                plt.xticks(rotation=45)
                plt.ylabel(metric)
            
            plt.tight_layout()
            plt.show()
            
            return df_comparison

def generate_sample_data() -> pd.DataFrame:
    """Generate sample dataset for demonstration"""
    np.random.seed(42)
    
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
    }
    
    # Create target variable (salary prediction)
    data['salary'] = (
        data['income'] * 0.8 + 
        data['education_years'] * 2000 + 
        data['experience'] * 1500 + 
        np.random.normal(0, 5000, n_samples)
    )
    
    # Create classification target (high earner)
    data['high_earner'] = (data['salary'] > data['salary'].median()).astype(int)
    
    return pd.DataFrame(data)

def main():
    """Main data science pipeline demonstration"""
    print("📊 Data Science & Analytics Pipeline")
    print("=" * 50)
    
    # Generate sample data
    print("🔄 Generating sample dataset...")
    df = generate_sample_data()
    
    print(f"Dataset created with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Initialize data analyzer
    analyzer = DataAnalyzer(df)
    
    # Basic analysis
    print("\\n📈 Basic Data Analysis:")
    info = analyzer.basic_info()
    print(f"Shape: {info['shape']}")
    print(f"Numeric columns: {len(info['numeric_columns'])}")
    print(f"Categorical columns: {len(info['categorical_columns'])}")
    
    # Descriptive statistics
    print("\\n📊 Descriptive Statistics:")
    desc_stats = analyzer.descriptive_statistics()
    print(desc_stats.round(2))
    
    # Outlier detection
    print("\\n🎯 Outlier Detection:")
    outliers = analyzer.detect_outliers()
    for col, outlier_indices in outliers.items():
        if outlier_indices:
            print(f"{col}: {len(outlier_indices)} outliers detected")
    
    # Create visualizations
    print("\\n📈 Creating visualizations...")
    analyzer.create_visualizations()
    
    # Machine Learning Pipeline
    print("\\n🤖 Machine Learning Pipeline:")
    
    # Classification task
    print("\\n🎯 Classification Task (Predicting High Earners):")
    ml_classifier = MLPipeline(df, 'high_earner')
    ml_classifier.prepare_data()
    ml_classifier.train_classification_models()
    
    # Regression task
    print("\\n📈 Regression Task (Predicting Salary):")
    ml_regressor = MLPipeline(df.drop(columns=['high_earner']), 'salary')
    ml_regressor.prepare_data()
    ml_regressor.train_regression_models()
    
    # Feature importance
    print("\\n🔍 Feature Importance Analysis:")
    ml_regressor.feature_importance_analysis()
    
    # Model comparison
    print("\\n⚖️ Model Comparison:")
    comparison = ml_regressor.model_comparison()
    if comparison is not None:
        print(comparison)
    
    # Clustering analysis
    print("\\n🎨 Clustering Analysis:")
    clusters, kmeans_model = ml_regressor.perform_clustering(n_clusters=3)
    
    print("\\n✅ Data science pipeline complete!")
    print("📊 All analyses and visualizations have been generated.")

if __name__ == "__main__":
    main()`,
      language: 'python',
    },
    'Cargo.toml': {
      content: `[package]
name = "mobile-rust-app"
version = "0.1.0"
edition = "2021"

[lib]
name = "mobile_rust_app"
crate-type = ["cdylib", "staticlib"]

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
flutter_rust_bridge = "2.0"
jni = "0.21"

[target.'cfg(target_os = "android")'.dependencies]
android_logger = "0.13"

[target.'cfg(target_os = "ios")'.dependencies]
objc = "0.2"`,
      language: 'toml',
    },
    'flutter_rust_bridge.yaml': {
      content: `rust_input: src/lib.rs
dart_output: lib/bridge_generated.dart
c_output: ios/Classes/bridge_generated.h

# Android configuration
android:
  package_name: com.example.mobilerustapp
  
# iOS configuration  
ios:
  class_name: MobileRustAppPlugin`,
      language: 'toml',
    },
    'setup.py': {
      content: `#!/usr/bin/env python3
"""
Setup script for Python AI/ML environment
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-ml-research-env",
    version="0.1.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="Comprehensive AI/ML research environment with Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/ai-ml-research-env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-research=main:main",
        ],
    },
)`,
      language: 'python',
    },
    'README.md': {
      content: `# AI/ML Research Environment

A comprehensive Python-based environment for artificial intelligence and machine learning research, featuring state-of-the-art libraries and tools.

## 🚀 Features

### Core AI/ML Capabilities
- **Deep Learning**: PyTorch, TensorFlow, Keras
- **Computer Vision**: OpenCV, PIL, Albumentations
- **Natural Language Processing**: Transformers, NLTK, spaCy
- **Reinforcement Learning**: Stable-Baselines3, Gym
- **Data Science**: Pandas, NumPy, Scikit-learn

### Advanced Features
- **GPU Acceleration**: CUDA support for training
- **Distributed Training**: Multi-GPU and multi-node support
- **Model Deployment**: FastAPI, Streamlit integration
- **Experiment Tracking**: Weights & Biases, TensorBoard
- **Hyperparameter Optimization**: Optuna, Ray Tune

### Visualization & Analysis
- **Static Plots**: Matplotlib, Seaborn
- **Interactive Dashboards**: Plotly, Streamlit
- **Model Interpretability**: SHAP, LIME
- **Performance Monitoring**: Real-time metrics

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- Git

### Quick Setup
\`\`\`bash
# Clone the repository
git clone https://github.com/example/ai-ml-research-env.git
cd ai-ml-research-env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
\`\`\`

### GPU Setup (Optional)
\`\`\`bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
\`\`\`

## 🎯 Quick Start

### 1. Neural Network Training
\`\`\`python
from main import NeuralNetwork, train_model
import torch

# Create and train a neural network
model = NeuralNetwork(input_size=10, hidden_size=64, output_size=3)
# Training code here...
\`\`\`

### 2. Computer Vision Pipeline
\`\`\`python
from computer_vision import ObjectDetector, ImageClassifier

# Initialize models
detector = ObjectDetector()
classifier = ImageClassifier()

# Process images
detections = detector.detect(image)
class_pred, confidence = classifier.predict(image)
\`\`\`

### 3. NLP with Transformers
\`\`\`python
from nlp_transformer import TextClassifier, QuestionAnswering

# Text classification
classifier = TextClassifier()
results = classifier.predict(["This is amazing!", "I'm not happy"])

# Question answering
qa = QuestionAnswering()
answer = qa.answer_question(context, question)
\`\`\`

### 4. Reinforcement Learning
\`\`\`python
from reinforcement_learning import RLTrainer

# Train RL agent
trainer = RLTrainer('CartPole-v1')
scores = trainer.train_dqn(episodes=1000)
\`\`\`

### 5. Data Science Pipeline
\`\`\`python
from data_science import DataAnalyzer, MLPipeline

# Analyze data
analyzer = DataAnalyzer(df)
analyzer.create_visualizations()

# Train ML models
ml_pipeline = MLPipeline(df, target_column='target')
ml_pipeline.prepare_data()
ml_pipeline.train_classification_models()
\`\`\`

## 📊 Supported Use Cases

### Research & Development
- **Academic Research**: Implement and test new algorithms
- **Prototyping**: Rapid development of AI solutions
- **Benchmarking**: Compare model performances
- **Experimentation**: A/B testing for ML models

### Industry Applications
- **Computer Vision**: Object detection, image classification
- **NLP**: Sentiment analysis, text generation, Q&A systems
- **Recommendation Systems**: Collaborative and content-based filtering
- **Time Series**: Forecasting and anomaly detection
- **Optimization**: Hyperparameter tuning and AutoML

### Educational Use
- **Learning**: Hands-on AI/ML education
- **Teaching**: Classroom demonstrations
- **Workshops**: Interactive tutorials
- **Certification**: Practical skill development

## 🛠️ Development Tools

### Code Quality
\`\`\`bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run tests
pytest tests/
\`\`\`

### Jupyter Integration
\`\`\`bash
# Install Jupyter
pip install jupyter jupyterlab

# Start Jupyter Lab
jupyter lab
\`\`\`

### Experiment Tracking
\`\`\`python
import wandb

# Initialize experiment tracking
wandb.init(project="ai-research")

# Log metrics
wandb.log({"accuracy": 0.95, "loss": 0.05})
\`\`\`

## 📈 Performance Optimization

### GPU Utilization
- Automatic GPU detection and usage
- Mixed precision training support
- Memory optimization techniques
- Distributed training capabilities

### Model Optimization
- Model quantization and pruning
- ONNX export for deployment
- TensorRT optimization
- Edge deployment support

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
\`\`\`bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
\`\`\`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for transformer models and datasets
- OpenAI for inspiring AI research
- The open-source community for amazing tools and libraries

## 📞 Support

- **Documentation**: [Read the Docs](https://ai-ml-research-env.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/example/ai-ml-research-env/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/ai-ml-research-env/discussions)
- **Email**: research@example.com

---

**Happy Researching! 🧠🚀**`,
      language: 'markdown',
    },
  };

  const handleFileSelect = (filePath: string) => {
    const fileName = filePath.split('/').pop() || '';
    setSelectedFile(filePath);

    // Check if tab already exists
    const existingTab = tabs.find(tab => tab.id === fileName);
    if (existingTab) {
      setActiveTab(fileName);
      return;
    }

    // Create new tab
    const template = fileTemplates[fileName] || { content: '', language: 'text' };
    const newTab: Tab = {
      id: fileName,
      name: fileName,
      content: template.content,
      isDirty: false,
      language: template.language,
    };

    setTabs(prev => [...prev, newTab]);
    setActiveTab(fileName);
  };

  const handleTabClose = (tabId: string) => {
    setTabs(prev => prev.filter(tab => tab.id !== tabId));
    if (activeTab === tabId) {
      const remainingTabs = tabs.filter(tab => tab.id !== tabId);
      setActiveTab(remainingTabs.length > 0 ? remainingTabs[0].id : '');
    }
  };

  const handleContentChange = (content: string) => {
    setTabs(prev =>
      prev.map(tab =>
        tab.id === activeTab
          ? { ...tab, content, isDirty: true }
          : tab
      )
    );
  };

  const handleSave = () => {
    setTabs(prev =>
      prev.map(tab =>
        tab.id === activeTab
          ? { ...tab, isDirty: false }
          : tab
      )
    );
    setBuildStatus('success');
    setTimeout(() => setBuildStatus('idle'), 3000);
  };

  const handleCursorChange = (line: number, column: number) => {
    // In a real implementation, this would send cursor position to other collaborators
    console.log(`Cursor moved to line ${line}, column ${column}`);
  };

  const handleUserCursorUpdate = (userId: string, cursor: { line: number; column: number; file: string }) => {
    setCollaborators(prev => 
      prev.map(collab => 
        collab.userId === userId 
          ? { ...collab, line: cursor.line, column: cursor.column }
          : collab
      )
    );
  };

  const handleSelectTemplate = (template: any) => {
    // Clear existing tabs
    setTabs([]);
    setActiveTab('');
    
    // Create tabs from template files
    const newTabs: Tab[] = Object.entries(template.files).map(([fileName, fileData]: [string, any]) => ({
      id: fileName,
      name: fileName,
      content: fileData.content,
      isDirty: false,
      language: fileData.language,
    }));
    
    setTabs(newTabs);
    if (newTabs.length > 0) {
      setActiveTab(newTabs[0].id);
    }
    
    setTemplatesVisible(false);
    setBuildStatus('idle');
  };

  const handleRun = () => {
    setBuildStatus('building');
    setTerminalVisible(true);
    setTimeout(() => {
      setBuildStatus('success');
      setTimeout(() => setBuildStatus('idle'), 3000);
    }, 2000);
  };

  const currentTab = tabs.find(tab => tab.id === activeTab);
  const lineCount = currentTab ? currentTab.content.split('\n').length : 0;

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      {/* Toolbar */}
      <Toolbar
        user={user}
        onShowPricing={() => setShowPricing(true)}
        onShowProfile={() => setShowProfile(true)}
        onToggleTerminal={() => setTerminalVisible(!terminalVisible)}
        onToggleCollaboration={() => setCollaborationVisible(!collaborationVisible)}
        onToggleMarketplace={() => setMarketplaceVisible(!marketplaceVisible)}
        onSave={handleSave}
        onRun={handleRun}
        onShowTemplates={() => setTemplatesVisible(true)}
        onShowMobilePreview={() => setMobilePreviewVisible(true)}
      />

      {/* Main Content */}
      <div className="flex-1 flex min-h-0">
        {/* Sidebar */}
        <div className="w-64 flex-shrink-0 relative z-20">
          <FileExplorer
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
          />
        </div>

        {/* Editor Area */}
        <div className={`flex-1 flex flex-col min-w-0 ${collaborationVisible ? 'mr-80' : ''} transition-all duration-300`}>
          {tabs.length > 0 && (
            <TabBar
              tabs={tabs}
              activeTab={activeTab}
              onTabSelect={setActiveTab}
              onTabClose={handleTabClose}
            />
          )}

          <div className="flex-1 min-h-0">
            {currentTab ? (
              <CodeEditor
                content={currentTab.content}
                onChange={handleContentChange}
                collaborators={collaborators.filter(c => c.userId !== 'current-user')}
                onCursorChange={handleCursorChange}
                language={currentTab.language}
                fileName={currentTab.name}
              />
            ) : (
              <div className="h-full flex items-center justify-center bg-gray-900 text-gray-500">
                <div className="text-center">
                  <div className="text-6xl mb-4">🦀</div>
                  <h2 className="text-xl font-semibold mb-2">Rust Cloud IDE</h2>
                  <p className="mb-4">Select a file from the explorer to start coding</p>
                  <div className="space-y-2">
                    <button
                      onClick={() => setTemplatesVisible(true)}
                      className="bg-orange-600 hover:bg-orange-700 text-white px-6 py-2 rounded-lg transition-colors"
                    >
                      Browse Templates
                    </button>
                    <div>
                      <button
                        onClick={() => setDemoMode(true)}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
                      >
                        Take a Tour
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Collaboration Panel */}
        {collaborationVisible && (
          <div className="fixed right-0 top-0 bottom-0 z-30">
            <CollaborationPanel
              isVisible={collaborationVisible}
              onToggle={() => setCollaborationVisible(!collaborationVisible)}
              currentFile={currentTab?.name || ''}
              onUserCursorUpdate={handleUserCursorUpdate}
            />
          </div>
        )}
      </div>

      {/* Terminal */}
      <Terminal
        isVisible={terminalVisible}
        onToggle={() => setTerminalVisible(!terminalVisible)}
      />

      {/* Status Bar */}
      <StatusBar
        currentFile={currentTab?.name || ''}
        language={currentTab?.language || ''}
        lineCount={lineCount}
        currentLine={1}
        buildStatus={buildStatus}
        collaboratorCount={collaborators.length}
      />

      {/* Project Templates Modal */}
      <ProjectTemplates
        isVisible={templatesVisible}
        onClose={() => setTemplatesVisible(false)}
        onSelectTemplate={handleSelectTemplate}
      />

      {/* Pricing Modal */}
      {showPricing && (
        <PricingPage onClose={() => setShowPricing(false)} />
      )}

      {/* User Profile Modal */}
      {showProfile && (
        <UserProfile
          user={user}
          onClose={() => setShowProfile(false)}
          onLogout={handleLogout}
        />
      )}

      {/* Demo Mode */}
      <DemoMode
        isActive={demoMode}
        onToggle={() => setDemoMode(!demoMode)}
      />

      {/* Developer Marketplace */}
      <DeveloperMarketplace
        isVisible={marketplaceVisible}
        onToggle={() => setMarketplaceVisible(!marketplaceVisible)}
      />

      {/* Mobile Preview */}
      <MobilePreview
        isVisible={mobilePreviewVisible}
        onClose={() => setMobilePreviewVisible(false)}
        currentCode={currentTab?.content || ''}
        platform="android"
      />
    </div>
  );
}

export default App;