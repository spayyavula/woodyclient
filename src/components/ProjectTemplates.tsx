import React, { useState } from 'react';
import { 
  X, 
  Smartphone, 
  Globe, 
  Gamepad2, 
  Wifi, 
  DollarSign, 
  MessageSquare, 
  Database,
  Brain,
  Zap,
  Eye,
  Target,
  TrendingUp,
  Cpu,
  Network,
  BarChart3,
  Bot,
  Camera,
  Mic,
  FileText,
  Activity,
  Layers,
  GitBranch,
  Cloud,
  Shield,
  Rocket
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

  const templates: Template[] = [
    // AI/ML Templates
    {
      id: 'neural-network-rust',
      name: 'Neural Network Engine',
      description: 'High-performance neural network implementation in Rust with GPU acceleration',
      category: 'AI/ML',
      difficulty: 'Advanced',
      tags: ['Neural Networks', 'GPU', 'CUDA', 'Performance'],
      icon: <Brain className="w-6 h-6 text-purple-400" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Build custom neural networks for computer vision, NLP, and predictive analytics',
      techStack: ['Rust', 'CUDA', 'OpenCL', 'Candle', 'Tch'],
      features: [
        'GPU-accelerated training',
        'Custom layer implementations',
        'Automatic differentiation',
        'Model serialization',
        'Distributed training'
      ],
      files: {
        'main.rs': {
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
    
    pub fn train(&mut self, data: &[(Tensor, Tensor)], epochs: usize, learning_rate: f64) -> Result<()> {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (input, target) in data {
                let output = self.forward(input)?;
                // Calculate loss and backpropagate
                // Implementation details...
            }
            println!("Epoch {}: Loss = {}", epoch, total_loss);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    println!("Neural Network Engine initialized on {:?}", device);
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "neural-network-engine"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = "0.3"
candle-nn = "0.3"
candle-transformers = "0.3"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }`,
          language: 'toml',
        }
      }
    },
    // Python AI/ML Templates
    {
      id: 'python-ml-research',
      name: 'Python AI/ML Research Lab',
      description: 'Complete Python environment for cutting-edge AI research with PyTorch, TensorFlow, and latest ML libraries',
      category: 'AI/ML',
      difficulty: 'Intermediate',
      tags: ['Python', 'PyTorch', 'TensorFlow', 'Research', 'Deep Learning'],
      icon: <Brain className="w-6 h-6 text-purple-400" />,
      estimatedTime: '2-3 weeks',
      useCase: 'Build and train state-of-the-art AI models for research and production',
      techStack: ['Python', 'PyTorch', 'TensorFlow', 'Scikit-learn', 'Pandas'],
      features: [
        'Pre-configured ML environment',
        'GPU acceleration support',
        'Jupyter notebook integration',
        'Experiment tracking with W&B',
        'Model deployment ready'
      ],
      files: {
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

def main():
    print("🐍 Python AI/ML Research Environment")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate sample data
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 3, 1000)
    
    # Initialize model
    model = NeuralNetwork(input_size=10, hidden_size=64, output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("✅ AI/ML environment ready!")
    print("🚀 Start building amazing AI models!")

if __name__ == "__main__":
    main()`,
          language: 'python',
        },
        'requirements.txt': {
          content: `# Core ML/AI Libraries
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Computer Vision
opencv-python>=4.8.0
Pillow>=10.0.0

# NLP
transformers>=4.30.0
datasets>=2.13.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.23.0`,
          language: 'text',
        }
      }
    },
    {
      id: 'python-deep-learning',
      name: 'Deep Learning with PyTorch',
      description: 'Advanced deep learning models with PyTorch, including CNNs, RNNs, and Transformers',
      category: 'AI/ML',
      difficulty: 'Advanced',
      tags: ['Deep Learning', 'PyTorch', 'Neural Networks', 'CNN', 'RNN'],
      icon: <Zap className="w-6 h-6 text-yellow-400" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Build advanced deep learning models for various AI applications',
      techStack: ['Python', 'PyTorch', 'CUDA', 'TensorBoard', 'Weights & Biases'],
      features: [
        'Custom neural architectures',
        'GPU acceleration',
        'Experiment tracking',
        'Model checkpointing',
        'Distributed training'
      ],
      files: {
        'deep_learning.py': {
          content: `#!/usr/bin/env python3
"""
Deep Learning with PyTorch
Advanced neural networks and deep learning models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.writer = SummaryWriter()
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return total_loss / len(dataloader), 100. * correct / total

def main():
    print("🧠 Deep Learning with PyTorch")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    model = ConvNet(num_classes=10)
    trainer = Trainer(model, device)
    
    print("✅ Deep learning environment ready!")
    print("🎯 Start training your neural networks!")

if __name__ == "__main__":
    main()`,
          language: 'python',
        }
      }
    },
    {
      id: 'python-computer-vision',
      name: 'Computer Vision with OpenCV',
      description: 'Advanced computer vision with OpenCV, YOLO, and real-time image processing',
      category: 'AI/ML',
      difficulty: 'Advanced',
      tags: ['Computer Vision', 'OpenCV', 'YOLO', 'Object Detection'],
      icon: <Camera className="w-6 h-6 text-green-400" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Build production-ready computer vision applications',
      techStack: ['Python', 'OpenCV', 'YOLO', 'PyTorch', 'Ultralytics'],
      features: [
        'Object detection and tracking',
        'Real-time video processing',
        'Image classification',
        'Face recognition',
        'Edge deployment'
      ],
      files: {
        'computer_vision.py': {
          content: `#!/usr/bin/env python3
"""
Computer Vision with OpenCV
Advanced computer vision and object detection
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        print(f"🎯 Model loaded on {self.device}")
    
    def detect_objects(self, image_path, conf_threshold=0.5):
        results = self.model(image_path, conf=conf_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': self.model.names[int(box.cls[0])]
                    }
                    detections.append(detection)
        
        return detections
    
    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame)
            annotated_frame = results[0].plot()
            
            cv2.imshow('Object Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"📹 Processed {frame_count} frames")

def main():
    print("👁️ Computer Vision with OpenCV")
    print("=" * 50)
    
    detector = ObjectDetector()
    print("✅ Computer vision pipeline ready!")
    print("🎯 Start detecting objects in images and videos!")

if __name__ == "__main__":
    main()`,
          language: 'python',
        }
      }
    },
    {
      id: 'python-nlp-transformers',
      name: 'NLP with Transformers',
      description: 'Natural language processing using Hugging Face Transformers and large language models',
      category: 'AI/ML',
      difficulty: 'Advanced',
      tags: ['NLP', 'Transformers', 'BERT', 'GPT', 'LLM'],
      icon: <FileText className="w-6 h-6 text-blue-400" />,
      estimatedTime: '2-3 weeks',
      useCase: 'Build intelligent text processing and language understanding systems',
      techStack: ['Python', 'Transformers', 'PyTorch', 'Datasets', 'Tokenizers'],
      features: [
        'Text classification',
        'Named entity recognition',
        'Question answering',
        'Text generation',
        'Sentiment analysis'
      ],
      files: {
        'nlp_transformers.py': {
          content: `#!/usr/bin/env python3
"""
NLP with Transformers
Advanced natural language processing with Hugging Face
"""

import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, AutoModelForQuestionAnswering, AutoModelForCausalLM
)
import numpy as np
from typing import List, Dict, Optional

class NLPPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Initialize different models for various tasks
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.text_generator = pipeline(
            "text-generation",
            model="gpt2",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            result = self.sentiment_analyzer(text)[0]
            results.append({
                'text': text,
                'sentiment': result['label'],
                'confidence': result['score']
            })
        return results
    
    def answer_question(self, context: str, question: str) -> Dict:
        result = self.qa_pipeline(question=question, context=context)
        return {
            'question': question,
            'answer': result['answer'],
            'confidence': result['score'],
            'start': result['start'],
            'end': result['end']
        }
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        result = self.text_generator(
            prompt, 
            max_length=max_length, 
            num_return_sequences=1,
            temperature=0.7
        )
        return result[0]['generated_text']

class TextClassifier:
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        
    def predict(self, texts: List[str]) -> List[Dict]:
        self.model.eval()
        results = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions).item()
                confidence = predictions[0][predicted_class].item()
                
                results.append({
                    'text': text,
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })
        
        return results

def main():
    print("🤖 NLP with Transformers")
    print("=" * 50)
    
    # Initialize NLP pipeline
    nlp = NLPPipeline()
    
    # Example usage
    sample_texts = [
        "I love this new AI model!",
        "This is terrible and disappointing.",
        "The weather is nice today."
    ]
    
    print("📊 Analyzing sentiment...")
    sentiments = nlp.analyze_sentiment(sample_texts)
    for result in sentiments:
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print()
    
    print("✅ NLP pipeline ready!")
    print("🎯 Start processing text with transformers!")

if __name__ == "__main__":
    main()`,
          language: 'python',
        }
      }
    },
    {
      id: 'python-reinforcement-learning',
      name: 'Reinforcement Learning Lab',
      description: 'Advanced RL algorithms with Stable-Baselines3, custom environments, and multi-agent systems',
      category: 'AI/ML',
      difficulty: 'Expert',
      tags: ['Reinforcement Learning', 'RL', 'Stable-Baselines3', 'Gym', 'Multi-agent'],
      icon: <Target className="w-6 h-6 text-red-400" />,
      estimatedTime: '4-6 weeks',
      useCase: 'Build intelligent agents for games, robotics, and decision-making systems',
      techStack: ['Python', 'Stable-Baselines3', 'Gym', 'PyTorch', 'Ray'],
      features: [
        'Custom RL environments',
        'Multi-agent training',
        'Policy optimization',
        'Distributed training',
        'Real-time visualization'
      ],
      files: {
        'reinforcement_learning.py': {
          content: `#!/usr/bin/env python3
"""
Reinforcement Learning Lab
Advanced RL with Stable-Baselines3 and custom environments
"""

import gym
import numpy as np
import torch
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from typing import Dict, Any

class CustomEnvironment(gym.Env):
    """Custom RL environment for demonstration"""
    
    def __init__(self):
        super(CustomEnvironment, self).__init__()
        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(4)  # 4 possible actions
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        self.state = None
        self.steps = 0
        self.max_steps = 200
    
    def reset(self):
        self.state = np.random.uniform(-1, 1, 4).astype(np.float32)
        self.steps = 0
        return self.state
    
    def step(self, action):
        # Simple environment logic
        self.state += np.random.normal(0, 0.1, 4)
        self.steps += 1
        
        # Calculate reward
        reward = -np.sum(np.abs(self.state))
        
        # Check if episode is done
        done = self.steps >= self.max_steps or np.any(np.abs(self.state) > 5)
        
        info = {'steps': self.steps}
        
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        print(f"State: {self.state}, Steps: {self.steps}")

class RLTrainer:
    def __init__(self, env_name: str = "CartPole-v1"):
        self.env_name = env_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
    def create_environment(self, n_envs: int = 1):
        if self.env_name == "Custom":
            env = CustomEnvironment()
            if n_envs > 1:
                env = make_vec_env(lambda: env, n_envs=n_envs)
        else:
            env = make_vec_env(self.env_name, n_envs=n_envs)
        
        return env
    
    def train_agent(self, algorithm: str = "PPO", total_timesteps: int = 10000):
        env = self.create_environment()
        eval_env = self.create_environment()
        
        # Choose algorithm
        if algorithm == "PPO":
            model = PPO("MlpPolicy", env, verbose=1, device=self.device)
        elif algorithm == "DQN":
            model = DQN("MlpPolicy", env, verbose=1, device=self.device)
        elif algorithm == "A2C":
            model = A2C("MlpPolicy", env, verbose=1, device=self.device)
        elif algorithm == "SAC":
            model = SAC("MlpPolicy", env, verbose=1, device=self.device)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path='./best_model/',
            log_path='./logs/', 
            eval_freq=1000,
            deterministic=True, 
            render=False
        )
        
        print(f"🎯 Training {algorithm} agent...")
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        return model
    
    def evaluate_agent(self, model, n_episodes: int = 10):
        env = self.create_environment()
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        print(f"📊 Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        return episode_rewards

def main():
    print("🎮 Reinforcement Learning Lab")
    print("=" * 50)
    
    trainer = RLTrainer("CartPole-v1")
    
    # Train different algorithms
    algorithms = ["PPO", "DQN", "A2C"]
    
    for algo in algorithms:
        print(f"\n🤖 Training {algo} agent...")
        model = trainer.train_agent(algorithm=algo, total_timesteps=5000)
        
        print(f"📈 Evaluating {algo} agent...")
        rewards = trainer.evaluate_agent(model, n_episodes=5)
    
    print("✅ RL training complete!")
    print("🎯 Agents ready for deployment!")

if __name__ == "__main__":
    main()`,
          language: 'python',
        }
      }
    },
    // Mobile Development Templates
    {
      id: 'flutter-rust-mobile',
      name: 'Flutter + Rust Mobile App',
      description: 'Cross-platform mobile app with Rust backend and Flutter frontend',
      category: 'Mobile',
      difficulty: 'Intermediate',
      tags: ['Flutter', 'Rust', 'Mobile', 'Cross-platform'],
      icon: <Smartphone className="w-6 h-6 text-blue-400" />,
      estimatedTime: '2-3 weeks',
      useCase: 'Build high-performance mobile apps with native Rust logic',
      techStack: ['Flutter', 'Dart', 'Rust', 'FFI'],
      features: [
        'Cross-platform compatibility',
        'Native performance',
        'Rust business logic',
        'Flutter UI framework',
        'Hot reload development'
      ],
      files: {
        'main.dart': {
          content: `import 'package:flutter/material.dart';
import 'package:mobile_rust_app/bridge_generated.dart';

void main() {
  runApp(const MobileRustApp());
}

class MobileRustApp extends StatelessWidget {
  const MobileRustApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mobile Rust App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepOrange),
        useMaterial3: true,
      ),
      home: const MobileHomePage(),
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
      print('Error initializing Rust: \$e');
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
      ),
    );
  }
}`,
          language: 'dart',
        },
        'lib.rs': {
          content: `pub mod mobile_utils;

use flutter_rust_bridge::frb;

#[frb(sync)]
pub fn init_mobile_app() -> String {
    "Mobile Rust App Initialized".to_string()
}

#[frb(sync)]
pub fn get_platform_info() -> String {
    "Cross-platform Mobile App".to_string()
}`,
          language: 'rust',
        }
      }
    },
    // Web Development Templates
    {
      id: 'rust-web-api',
      name: 'High-Performance Web API',
      description: 'Blazing fast REST API built with Rust and Actix-web',
      category: 'Web',
      difficulty: 'Intermediate',
      tags: ['Web API', 'REST', 'Actix', 'Performance'],
      icon: <Globe className="w-6 h-6 text-green-400" />,
      estimatedTime: '1-2 weeks',
      useCase: 'Build scalable web APIs with excellent performance',
      techStack: ['Rust', 'Actix-web', 'PostgreSQL', 'Redis'],
      features: [
        'High-performance HTTP server',
        'Database integration',
        'Authentication middleware',
        'API documentation',
        'Docker deployment'
      ],
      files: {
        'main.rs': {
          content: `use actix_web::{web, App, HttpServer, Result, HttpResponse};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ApiResponse {
    message: String,
    status: String,
}

async fn health_check() -> Result<HttpResponse> {
    let response = ApiResponse {
        message: "API is running".to_string(),
        status: "healthy".to_string(),
    };
    Ok(HttpResponse::Ok().json(response))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Starting Rust Web API server...");
    
    HttpServer::new(|| {
        App::new()
            .route("/health", web::get().to(health_check))
            .route("/api/v1/status", web::get().to(health_check))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "rust-web-api"
version = "0.1.0"
edition = "2021"

[dependencies]
actix-web = "4.0"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "postgres"] }`,
          language: 'toml',
        }
      }
    },
    // Game Development Templates
    {
      id: 'rust-game-engine',
      name: 'Game Engine Framework',
      description: 'Modern game engine built with Rust and Bevy',
      category: 'Gaming',
      difficulty: 'Advanced',
      tags: ['Game Engine', 'Bevy', '3D Graphics', 'ECS'],
      icon: <Gamepad2 className="w-6 h-6 text-red-400" />,
      estimatedTime: '4-6 weeks',
      useCase: 'Create high-performance games with modern architecture',
      techStack: ['Rust', 'Bevy', 'WGPU', 'ECS'],
      features: [
        'Entity Component System',
        '3D rendering pipeline',
        'Physics simulation',
        'Audio system',
        'Cross-platform deployment'
      ],
      files: {
        'main.rs': {
          content: `use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (move_player, rotate_cube))
        .run();
}

#[derive(Component)]
struct Player;

#[derive(Component)]
struct RotatingCube;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Spawn a cube
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            transform: Transform::from_xyz(0.0, 0.5, 0.0),
            ..default()
        },
        RotatingCube,
    ));

    // Spawn a camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Spawn a light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
}

fn move_player(
    keyboard_input: Res<Input<KeyCode>>,
    mut query: Query<&mut Transform, With<Player>>,
) {
    for mut transform in &mut query {
        if keyboard_input.pressed(KeyCode::W) {
            transform.translation.z -= 0.1;
        }
        if keyboard_input.pressed(KeyCode::S) {
            transform.translation.z += 0.1;
        }
    }
}

fn rotate_cube(time: Res<Time>, mut query: Query<&mut Transform, With<RotatingCube>>) {
    for mut transform in &mut query {
        transform.rotate_y(time.delta_seconds());
    }
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "rust-game-engine"
version = "0.1.0"
edition = "2021"

[dependencies]
bevy = "0.12"
bevy_rapier3d = "0.23"`,
          language: 'toml',
        }
      }
    }
  ];

  const categories = ['all', 'AI/ML', 'Mobile', 'Web', 'Gaming', 'Blockchain', 'IoT', 'DevOps'];

  const filteredTemplates = templates.filter(template => {
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
    const matchesSearch = template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesSearch;
  });

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'text-green-400 bg-green-400/10';
      case 'Intermediate': return 'text-yellow-400 bg-yellow-400/10';
      case 'Advanced': return 'text-orange-400 bg-orange-400/10';
      case 'Expert': return 'text-red-400 bg-red-400/10';
      default: return 'text-gray-400 bg-gray-400/10';
    }
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-xl border border-gray-700 w-full max-w-6xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div>
            <h2 className="text-2xl font-bold text-white">Project Templates</h2>
            <p className="text-gray-400 mt-1">Choose a template to get started quickly</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Filters */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <input
                type="text"
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-orange-500"
              />
            </div>
            <div className="flex gap-2 flex-wrap">
              {categories.map(category => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedCategory === category
                      ? 'bg-orange-600 text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  {category === 'all' ? 'All' : category}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Templates Grid */}
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredTemplates.map(template => (
              <div
                key={template.id}
                className="bg-gray-800 rounded-lg border border-gray-700 hover:border-orange-500 transition-all duration-200 cursor-pointer group"
                onClick={() => onSelectTemplate(template)}
              >
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      {template.icon}
                      <div>
                        <h3 className="font-semibold text-white group-hover:text-orange-400 transition-colors">
                          {template.name}
                        </h3>
                        <span className={`text-xs px-2 py-1 rounded-full ${getDifficultyColor(template.difficulty)}`}>
                          {template.difficulty}
                        </span>
                      </div>
                    </div>
                  </div>

                  <p className="text-gray-400 text-sm mb-4 line-clamp-2">
                    {template.description}
                  </p>

                  <div className="space-y-3">
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Use Case:</p>
                      <p className="text-sm text-gray-300">{template.useCase}</p>
                    </div>

                    <div>
                      <p className="text-xs text-gray-500 mb-1">Tech Stack:</p>
                      <div className="flex flex-wrap gap-1">
                        {template.techStack.slice(0, 3).map(tech => (
                          <span key={tech} className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">
                            {tech}
                          </span>
                        ))}
                        {template.techStack.length > 3 && (
                          <span className="text-xs text-gray-500">+{template.techStack.length - 3} more</span>
                        )}
                      </div>
                    </div>

                    <div>
                      <p className="text-xs text-gray-500 mb-1">Key Features:</p>
                      <ul className="text-sm text-gray-300 space-y-1">
                        {template.features.slice(0, 2).map(feature => (
                          <li key={feature} className="flex items-center gap-2">
                            <div className="w-1 h-1 bg-orange-400 rounded-full"></div>
                            {feature}
                          </li>
                        ))}
                        {template.features.length > 2 && (
                          <li className="text-xs text-gray-500">+{template.features.length - 2} more features</li>
                        )}
                      </ul>
                    </div>

                    <div className="flex items-center justify-between pt-2 border-t border-gray-700">
                      <span className="text-xs text-gray-500">Est. Time: {template.estimatedTime}</span>
                      <div className="flex gap-1">
                        {template.tags.slice(0, 2).map(tag => (
                          <span key={tag} className="text-xs bg-orange-600/20 text-orange-400 px-2 py-1 rounded">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {filteredTemplates.length === 0 && (
            <div className="text-center py-12">
              <div className="text-gray-500 mb-2">No templates found</div>
              <p className="text-gray-600 text-sm">Try adjusting your search or category filter</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProjectTemplates;