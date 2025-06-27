import React, { useState, useRef, useEffect } from 'react';
import { Terminal as TerminalIcon, X, Minimize2 } from 'lucide-react';

interface TerminalProps {
  isVisible: boolean;
  onToggle: () => void;
}

const Terminal: React.FC<TerminalProps> = ({ isVisible, onToggle }) => {
  const [history, setHistory] = useState<string[]>([
    '$ Welcome to Rust Cloud IDE Terminal',
    '$ Type "help" for available commands',
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
Hello, Mobile Rust!
Mobile Rust App Initialized
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
      case 'ls':
        output = 'Cargo.toml  Cargo.lock  README.md  src/  platforms/  tests/  assets/  target/  flutter_rust_bridge.yaml';
        break;
      case 'clear':
        setHistory(['$ Welcome to Mobile Rust IDE Terminal']);
        setCurrentCommand('');
        return;
      default:
        if (cmd) {
          output = `Command not found: ${cmd}. Type "help" for available commands.`;
        }
    }

    setHistory(prev => [...prev, `$ ${command}`, output].filter(Boolean));
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
          <div key={index} className={line.startsWith('$') ? 'text-green-400' : 'text-gray-300'}>
            {line}
          </div>
        ))}
        <div className="flex items-center text-green-400">
          <span>$ </span>
          <input
            type="text"
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