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
    </div>
  );
}

export default App;