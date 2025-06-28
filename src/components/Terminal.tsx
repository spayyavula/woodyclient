import React, { useState, useRef, useEffect } from 'react';
import { Terminal as TerminalIcon, X, Minimize2 } from 'lucide-react';
import { TERMINAL_CONSTANTS, formatCommandOutput, isTerminalPrompt } from '../utils/stringUtils';

interface TerminalProps {
  isVisible: boolean;
  onToggle: () => void;
}

const Terminal: React.FC<TerminalProps> = ({ isVisible, onToggle }) => {
  const [history, setHistory] = useState<string[]>([
    TERMINAL_CONSTANTS.WELCOME,
    TERMINAL_CONSTANTS.HELP,
  ]);
  const [currentCommand, setCurrentCommand] = useState('');
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [history]);

  const executeCommand = (command: string) => {
    const cmd = command.trim();
    let output = '';

    switch (cmd) {
      case 'help':
        output = `Available commands:
  cargo build    - Build the project
  cargo run      - Run the project
  cargo test     - Run tests
  cargo android  - Build for Android
  cargo ios      - Build for iOS
  flutter build  - Build Flutter app
  flutter run    - Run Flutter app
  adb devices    - List connected Android devices
  xcrun simctl   - iOS simulator commands
  cargo clean    - Clean build artifacts
  ls            - List files
  clear         - Clear terminal
  help          - Show this help`;
        break;
      case 'cargo build':
        output = `   Compiling mobile-rust-app v0.1.0
    Finished dev [unoptimized + debuginfo] target(s) in 2.34s`;
        break;
      case 'cargo run':
        output = `   Compiling mobile-rust-app v0.1.0
    Finished dev [unoptimized + debuginfo] target(s) in 1.23s
     Running \`target/debug/mobile-rust-app\`
Hello, rustyclint!
rustyclint Initialized
Platform: Desktop`;
        break;
      case 'cargo android':
        output = `   Compiling mobile-rust-app v0.1.0
    Finished release [optimized] target(s) in 4.56s
Building Android APK...
APK built successfully: target/android/mobile-rust-app.apk`;
        break;
      case 'cargo ios':
        output = `   Compiling mobile-rust-app v0.1.0
    Finished release [optimized] target(s) in 3.21s
Building iOS framework...
iOS framework built successfully`;
        break;
      case 'flutter build':
        output = `Building Flutter app...
Running "flutter packages get" in mobile-rust-app...
Launching lib/main.dart on iPhone 15 Pro in debug mode...
Built build/ios/iphoneos/Runner.app`;
        break;
      case 'flutter run':
        output = `Launching lib/main.dart on iPhone 15 Pro in debug mode...
🔥  To hot reload changes while running, press "r". To hot restart (and rebuild state), press "R".
An Observatory debugger and profiler on iPhone 15 Pro is available at: http://127.0.0.1:54321/
For a more detailed help message, press "h". To quit, press "q".

Application started successfully!`;
        break;
      case 'flutter doctor':
        output = `Doctor summary (to see all details, run flutter doctor -v):
[✓] Flutter (Channel stable, 3.16.0, on macOS 14.0 23A344 darwin-arm64, locale en-US)
[✓] Android toolchain - develop for Android devices (Android SDK version 34.0.0)
[✓] Xcode - develop for iOS and macOS (Xcode 15.0)
[✓] Chrome - develop for the web
[✓] Android Studio (version 2023.1)
[✓] VS Code (version 1.84.2)
[✓] Connected device (3 available)
[✓] Network resources

• No issues found!`;
        break;
      case 'flutter clean':
        output = `Deleting build...
Deleting .dart_tool...
Deleting .packages...
Deleting pubspec.lock...
Deleting .flutter-plugins...
Deleting .flutter-plugins-dependencies...`;
        break;
      case 'npx react-native start':
        output = `
                     ######                ######
                   ###     ####        ####     ###
                  ##          ###    ###          ##
                  ##             ####             ##
                  ##             ####             ##
                  ##           ##    ##           ##
                  ##         ###      ###         ##
                   ##  ########################  ##
                    ##  #####################  ##
                     ####  #################  ####
                       ########         ########
                          #####       #####
                             ###     ###
                                ## ##
                                 ###
                                  #

                      Welcome to React Native!
                        Learn once, write anywhere

┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Running Metro Bundler on port 8081.                                        │
│                                                                              │
│  Keep Metro running while developing on any JS projects. Feel free to       │
│  close this tab and run your own Metro instance if you prefer.              │
│                                                                              │
│  https://github.com/facebook/react-native                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

Looking for JS files in
   /home/project/platforms/react-native

Metro waiting on 'ws://localhost:8081/ws'...`;
        break;
      case 'npx react-native run-android':
        output = `info Running jetifier to migrate libraries to AndroidX. You can disable it using "--no-jetifier" flag.
Jetifier found 967 file(s) to forward-jetify. Using 4 workers...
info Starting JS server...
info Building and installing the app on the device (cd android && ./gradlew app:installDebug)...

> Task :app:installDebug
Installing APK 'app-debug.apk' on 'Pixel_3a_API_30_x86(AVD) - 11' for app:debug
Installed on 1 device.

BUILD SUCCESSFUL in 45s
27 actionable tasks: 27 executed
info Connecting to the development server...
info Starting the app on 'emulator-5554'...
Starting: Intent { cmp=com.mobilerustapp/.MainActivity }`;
        break;
      case 'adb devices':
        output = `List of devices attached
emulator-5554	device
R58M123ABCD	device`;
        break;
      case 'cargo test':
        output = `   Compiling mobile-rust-app v0.1.0
    Finished test [unoptimized + debuginfo] target(s) in 1.45s
     Running unittests src/lib.rs

running 4 tests
test tests::test_mobile_init ... ok
test tests::test_touch_validation ... ok
test tests::test_platform_detection ... ok
test tests::test_ui_rendering ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.15s`;
        break;
      case 'cargo clean':
        output = 'Cleaning target directory...';
        break;
      case 'python main.py':
        output = `🐍 Python AI/ML Research Environment
==================================================
✅ PyTorch version: 2.1.0+cu118
✅ CUDA available: True
✅ GPU device: NVIDIA RTX 4090
📊 Loading datasets...
🧠 Initializing neural network...
🚀 Training started...
Epoch 1/100: Loss = 0.8234, Accuracy = 0.6543
Epoch 10/100: Loss = 0.3421, Accuracy = 0.8765
Epoch 50/100: Loss = 0.1234, Accuracy = 0.9456
Epoch 100/100: Loss = 0.0567, Accuracy = 0.9823
✅ Training complete! Model saved to models/neural_net.pth`;
        break;
      case 'python computer_vision.py':
        output = `🔍 Computer Vision Pipeline
========================================
📦 Loading YOLO model...
✅ YOLOv5s model loaded successfully
🎥 Initializing camera feed...
📸 Processing frame 1: 3 objects detected
   - Person (confidence: 0.89)
   - Car (confidence: 0.76)
   - Traffic light (confidence: 0.92)
🎯 Real-time detection active
📊 FPS: 30.2, Memory: 2.1GB`;
        break;
      case 'python nlp_transformer.py':
        output = `🤖 Advanced NLP with Transformers
==================================================
📥 Loading BERT model...
✅ bert-base-uncased loaded successfully
🔄 Processing text samples...
📊 Sentiment Analysis Results:
   - "I love this!" → Positive (0.94)
   - "This is terrible" → Negative (0.87)
🏷️ Named Entity Recognition:
   - "Apple Inc." → ORG (0.99)
   - "New York" → LOC (0.95)
✅ NLP pipeline ready!`;
        break;
      case 'python data_science.py':
        output = `📊 Data Science & Analytics Pipeline
==================================================
📈 Dataset loaded: 10,000 rows, 15 columns
🔍 Exploratory Data Analysis:
   - Missing values: 2.3%
   - Outliers detected: 156 samples
   - Correlation analysis complete
🤖 Training ML models:
   - Random Forest: 94.2% accuracy
   - Gradient Boosting: 95.7% accuracy
   - Logistic Regression: 89.3% accuracy
📊 Visualizations generated
✅ Analysis complete!`;
        break;
      case 'pip install -r requirements.txt':
        output = `Collecting torch>=2.0.0
  Downloading torch-2.1.0+cu118-cp311-cp311-linux_x86_64.whl (2.0 GB)
Collecting torchvision>=0.15.0
  Downloading torchvision-0.16.0+cu118-cp311-cp311-linux_x86_64.whl (6.9 MB)
Collecting tensorflow>=2.13.0
  Downloading tensorflow-2.13.0-cp311-cp311-linux_x86_64.whl (524.1 MB)
Collecting transformers>=4.30.0
  Downloading transformers-4.33.2-py3-none-any.whl (7.6 MB)
Collecting opencv-python>=4.8.0
  Downloading opencv_python-4.8.1.78-cp37-abi3-linux_x86_64.whl (61.0 MB)
Installing collected packages: torch, torchvision, tensorflow, transformers, opencv-python...
Successfully installed torch-2.1.0+cu118 torchvision-0.16.0+cu118 tensorflow-2.13.0 transformers-4.33.2 opencv-python-4.8.1.78`;
        break;
      case 'jupyter lab':
        output = `[I 2024-01-15 10:30:45.123 ServerApp] jupyterlab | extension was successfully linked.
[I 2024-01-15 10:30:45.456 ServerApp] Writing Jupyter server cookie secret to /home/user/.local/share/jupyter/runtime/jupyter_cookie_secret
[I 2024-01-15 10:30:45.789 ServerApp] Serving at http://localhost:8888/lab?token=abc123def456
[I 2024-01-15 10:30:45.890 ServerApp] Use Control-C to stop this server and shut down all kernels
[I 2024-01-15 10:30:46.123 ServerApp] 
    To access the server, open this file in a browser:
        file:///home/user/.local/share/jupyter/runtime/jpserver-12345-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=abc123def456
        http://127.0.0.1:8888/lab?token=abc123def456`;
        break;
      case 'python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"':
        output = 'CUDA available: True';
        break;
      case 'ls':
        output = 'Cargo.toml  Cargo.lock  README.md  src/  platforms/  tests/  assets/  target/  flutter_rust_bridge.yaml  main.py  requirements.txt  computer_vision.py  nlp_transformer.py  data_science.py  setup.py';
        break;
      case 'clear':
        setHistory([TERMINAL_CONSTANTS.WELCOME]);
        setCurrentCommand('');
        return;
      default:
        if (cmd) {
          output = `Command not found: ${cmd}. Type "help" for available commands.`;
        }
    }

    // Use utility function for safe command formatting
    const newEntries = formatCommandOutput(command, output);
    
    setHistory(prev => [...prev, ...newEntries]);
    setCurrentCommand('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      executeCommand(currentCommand);
    }
  };

  if (!isVisible) return null;

  return (
    <div className="bg-gray-900 border-t border-gray-700 flex flex-col h-64">
      <div className="flex items-center justify-between bg-gray-800 px-4 py-2 border-b border-gray-700">
        <div className="flex items-center">
          <TerminalIcon className="w-4 h-4 mr-2 text-gray-400" />
          <span className="text-sm font-medium text-gray-200">Terminal</span>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={onToggle}
            className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200"
          >
            <Minimize2 className="w-4 h-4" />
          </button>
        </div>
      </div>
      <div
        ref={terminalRef}
        className="flex-1 p-4 overflow-y-auto font-mono text-sm text-gray-100"
      >
        {history.map((line, index) => (
          <div key={index} className={isTerminalPrompt(line) ? 'text-green-400' : 'text-gray-300'}>
            {line}
          </div>
        ))}
        <div className="flex items-center text-green-400">
          <span>{TERMINAL_CONSTANTS.PROMPT}</span>
          <input
            type="text"
            autoComplete="off"
            value={currentCommand}
            onChange={(e) => setCurrentCommand(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-transparent outline-none text-gray-100 ml-1"
            placeholder="Enter command..."
            autoFocus
          />
        </div>
      </div>
    </div>
  );
};

export default Terminal;