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
 rustyclint scan - Run security analysis
 rustyclint fix  - Auto-fix vulnerabilities
 rustyclint bench - Performance benchmark
 rustyclint audit - Security audit
  cargo clean    - Clean build artifacts
  ls            - List files
  clear         - Clear terminal
  help          - Show this help`;
        break;
      case 'cargo build':
        output = `   Compiling rustyclint v0.1.0
    Finished dev [unoptimized + debuginfo] target(s) in 2.34s`;
        break;
      case 'cargo run':
        output = `   Compiling rustyclint v0.1.0
    Finished dev [unoptimized + debuginfo] target(s) in 1.23s
     Running \`target/debug/rustyclint\`
Hello, rustyclint!
rustyclint Initialized
Security Engine: Active
Performance Mode: Optimized`;
        break;
      case 'rustyclint scan':
        output = `🔍 Security Analysis Starting...
📊 Analyzing 47,392 lines of code
🛡️  Checking OWASP Top 10 vulnerabilities
⚡ Performance: 10.2M lines/second

Results:
✅ 0 Critical vulnerabilities
⚠️  2 Medium-risk issues found
🔧 3 Performance optimizations suggested

Analysis completed in 0.08 seconds`;
        break;
      case 'rustyclint fix':
        output = `🔧 Auto-fixing vulnerabilities...

Fixed Issues:
✅ SQL injection vulnerability in auth.rs:42
✅ XSS prevention in template.rs:156
✅ Buffer overflow protection in parser.rs:89

Performance Improvements:
⚡ Optimized memory allocation (-23% usage)
⚡ Parallel processing enabled (+340% speed)

All fixes applied successfully!`;
        break;
      case 'rustyclint bench':
        output = `🚀 Performance Benchmark

Analysis Speed:
├─ Lines per second: 10,247,832
├─ Memory usage: 45.2 MB
├─ CPU utilization: 12%
└─ Response time: <50ms

Security Checks:
├─ Vulnerability detection: 99.97% accuracy
├─ False positive rate: 0.03%
├─ Coverage: OWASP Top 10 + Custom rules
└─ Compliance: SOC 2, GDPR, HIPAA ready

Platform Performance:
├─ Native (Windows/macOS/Linux): 100%
├─ WebAssembly (Browser): 95%
└─ Cloud (Auto-scaling): 99.99% uptime`;
        break;
      case 'rustyclint audit':
        output = `🛡️  Security Audit Report

Encryption Status:
✅ AES-256 encryption active
✅ TLS 1.3 for data in transit
✅ Zero-trust architecture verified

Compliance Check:
✅ SOC 2 Type II compliant
✅ GDPR data protection verified
✅ HIPAA security controls active
✅ PCI DSS requirements met

Vulnerability Scan:
✅ 0 Critical issues
✅ 0 High-risk vulnerabilities
⚠️  2 Medium-risk items (non-blocking)

Security Score: 98/100 (Excellent)`;
        break;
      case 'cargo test':
        output = `   Compiling rustyclint v0.1.0
    Finished test [unoptimized + debuginfo] target(s) in 1.45s
     Running unittests src/lib.rs

running 8 tests
test tests::test_security_scanner ... ok
test tests::test_vulnerability_detection ... ok
test tests::test_performance_analysis ... ok
test tests::test_encryption_validation ... ok
test tests::test_compliance_check ... ok
test tests::test_memory_safety ... ok
test tests::test_parallel_processing ... ok
test tests::test_zero_trust_auth ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.08s`;
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
        output = 'Cargo.toml  Cargo.lock  README.md  src/  security/  tests/  benchmarks/  target/  rustyclint.toml  main.py  requirements.txt  security_scanner.py  performance_analyzer.py  compliance_checker.py  setup.py';
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