import React, { useState } from 'react';
import { X, Smartphone, Globe, Gamepad2, Wifi, DollarSign, MessageSquare, Database, Brain, Zap, Eye, Target, TrendingUp, Cpu, Network, BarChart3, Bot, Camera, Mic, FileText, Activity, Layers, GitBranch, Cloud, Shield, Rocket, Server, Monitor, Thermometer, Lock, Radio, Gauge, Home, Car, Factory, Lightbulb, Wrench, Package, Container, Settings, Code, Terminal, HardDrive, Workflow, Timer, AlertTriangle, CheckCircle, Search, Microscope, Atom, Dna, Beaker, Calculator, PieChart, LineChart, Headphones, Video, Image, Music, Palette, Brush, Sparkles, Wand2, Fingerprint, ScanLine, Radar, Satellite, MapPin, Navigation, Compass, Route, Truck, Plane, Ship, Train, Building, Store, ShoppingBag, CreditCard, Coins, TrendingDown, Users, UserCheck, Heart, Stethoscope, Pill, TestTube, Syringe, Ban as Bandage, BookOpen, GraduationCap, School, Library, Bookmark, PenTool, Edit, Type, AlignLeft, Calendar, Clock, AlarmPlus as Alarm, Watch as Stopwatch, Hourglass, Sun, Moon, CloudRain, Snowflake, Wind, Umbrella, TreePine, Leaf, Flower, Sprout, Recycle, Battery, Plug, Power, Fuel, Flame, Droplets, Waves, Mountain, Scan as Volcano, Globe2, Earth, Star, Telescope, Rocket as RocketIcon, Orbit, Atom as AtomIcon, Dna as DnaIcon, Microscope as MicroscopeIcon, FlaskConical, TestTube2, Pipette, Scale, Ruler, Triangle, Square, Circle, Hexagon, Pentagon, Diamond, Hash, Binary, Braces, Brackets, Parentheses, Quote, AtSign, Percent, Plus, Minus, Equal, EqualNot as NotEqual, Theater as GreaterThan, Shapes as LessThan, Infinity, Pi, Sigma, Delete as Delta, Lamp as Lambda, Vegan as Omega, Album as Alpha, Bed as Beta, Drama as Gamma, Sheet as Theta, Ship as Phi, Music as Psi, Ship as Chi, Ghost as Rho, Tag as Tau, Music as Mu, Nut as Nu, AArrowDown as Xi, MicrowaveIcon as Omicron, FileUpIcon as Upsilon, HandMetal as Zeta, Star as Eta, Bot as Iota, Map as Kappa } from 'lucide-react';

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
    // AI/ML Templates - Rust
    {
      id: 'neural-network-rust',
      name: 'Neural Network Engine',
      description: 'High-performance neural network implementation in Rust with GPU acceleration',
      category: 'AI/ML',
      difficulty: 'Advanced',
      tags: ['Neural Networks', 'GPU', 'CUDA', 'Performance', 'Rust'],
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
    println!("🧠 Neural Network Engine initialized on {:?}", device);
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
    {
      id: 'rust-computer-vision',
      name: 'Computer Vision Pipeline',
      description: 'Real-time computer vision with Rust, OpenCV bindings, and YOLO integration',
      category: 'AI/ML',
      difficulty: 'Advanced',
      tags: ['Computer Vision', 'OpenCV', 'YOLO', 'Real-time', 'Rust'],
      icon: <Camera className="w-6 h-6 text-green-400" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Build production-ready computer vision applications with Rust performance',
      techStack: ['Rust', 'OpenCV', 'ONNX', 'TensorRT', 'CUDA'],
      features: [
        'Real-time object detection',
        'Video stream processing',
        'Edge deployment ready',
        'Multi-threaded processing',
        'Custom model integration'
      ],
      files: {
        'main.rs': {
          content: `use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    imgcodecs::{imread, IMREAD_COLOR},
    imgproc::{rectangle, put_text, FONT_HERSHEY_SIMPLEX},
    objdetect::HOGDescriptor,
    prelude::*,
    videoio::{VideoCapture, CAP_ANY},
    highgui::{imshow, wait_key, named_window, WINDOW_AUTOSIZE},
};
use std::sync::{Arc, Mutex};
use std::thread;

pub struct ObjectDetector {
    model_path: String,
    confidence_threshold: f32,
    nms_threshold: f32,
}

impl ObjectDetector {
    pub fn new(model_path: String) -> Self {
        Self {
            model_path,
            confidence_threshold: 0.5,
            nms_threshold: 0.4,
        }
    }
    
    pub fn detect_objects(&self, frame: &Mat) -> opencv::Result<Vec<Detection>> {
        let mut detections = Vec::new();
        
        // YOLO detection implementation
        // This would integrate with ONNX runtime or TensorRT
        
        Ok(detections)
    }
    
    pub fn process_video_stream(&self, camera_id: i32) -> opencv::Result<()> {
        let mut cap = VideoCapture::new(camera_id, CAP_ANY)?;
        let mut frame = Mat::default();
        
        named_window("Object Detection", WINDOW_AUTOSIZE)?;
        
        loop {
            cap.read(&mut frame)?;
            if frame.empty() {
                break;
            }
            
            let detections = self.detect_objects(&frame)?;
            let annotated_frame = self.draw_detections(&frame, &detections)?;
            
            imshow("Object Detection", &annotated_frame)?;
            
            if wait_key(1)? == 27 { // ESC key
                break;
            }
        }
        
        Ok(())
    }
    
    fn draw_detections(&self, frame: &Mat, detections: &[Detection]) -> opencv::Result<Mat> {
        let mut result = frame.clone();
        
        for detection in detections {
            let rect = Rect::new(
                detection.bbox.x,
                detection.bbox.y,
                detection.bbox.width,
                detection.bbox.height,
            );
            
            rectangle(
                &mut result,
                rect,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                8,
                0,
            )?;
            
            let label = format!("{}: {:.2}", detection.class_name, detection.confidence);
            put_text(
                &mut result,
                &label,
                Point::new(detection.bbox.x, detection.bbox.y - 10),
                FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                1,
                8,
                false,
            )?;
        }
        
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub class_id: i32,
    pub class_name: String,
}

fn main() -> opencv::Result<()> {
    println!("👁️ Computer Vision Pipeline with Rust");
    println!("=====================================");
    
    let detector = ObjectDetector::new("models/yolo.onnx".to_string());
    detector.process_video_stream(0)?;
    
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "rust-computer-vision"
version = "0.1.0"
edition = "2021"

[dependencies]
opencv = { version = "0.88", features = ["opencv-4"] }
ort = "1.16"
ndarray = "0.15"
image = "0.24"
tokio = { version = "1.0", features = ["full"] }`,
          language: 'toml',
        }
      }
    },
    {
      id: 'rust-nlp-transformer',
      name: 'NLP Transformer Engine',
      description: 'High-performance NLP with Rust, transformer models, and tokenization',
      category: 'AI/ML',
      difficulty: 'Expert',
      tags: ['NLP', 'Transformers', 'BERT', 'GPT', 'Rust'],
      icon: <FileText className="w-6 h-6 text-blue-400" />,
      estimatedTime: '4-6 weeks',
      useCase: 'Build intelligent text processing and language understanding systems',
      techStack: ['Rust', 'Candle', 'Tokenizers', 'ONNX', 'HuggingFace'],
      features: [
        'Transformer model inference',
        'Custom tokenization',
        'Multi-language support',
        'Batch processing',
        'Memory-efficient attention'
      ],
      files: {
        'main.rs': {
          content: `use candle_core::{Device, Tensor, Result};
use candle_transformers::models::bert::{BertModel, Config};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use std::collections::HashMap;

pub struct NLPPipeline {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    vocab: HashMap<String, u32>,
}

impl NLPPipeline {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;
        
        // Load model configuration
        let config = Config::bert_base_uncased();
        
        // Initialize model (simplified - would load from checkpoint)
        let vs = VarBuilder::zeros(candle_core::DType::F32, &device);
        let model = BertModel::load(&vs, &config)?;
        
        Ok(Self {
            model,
            tokenizer,
            device,
            vocab: HashMap::new(),
        })
    }
    
    pub fn encode_text(&self, text: &str) -> Result<Tensor> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization failed: {}", e)))?;
        
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        let input_ids = Tensor::new(tokens.as_slice(), &self.device)?
            .unsqueeze(0)?; // Add batch dimension
        
        Ok(input_ids)
    }
    
    pub fn get_embeddings(&self, text: &str) -> Result<Tensor> {
        let input_ids = self.encode_text(text)?;
        let embeddings = self.model.forward(&input_ids)?;
        Ok(embeddings)
    }
    
    pub fn classify_sentiment(&self, text: &str) -> Result<SentimentResult> {
        let embeddings = self.get_embeddings(text)?;
        
        // Apply classification head (simplified)
        let logits = embeddings.mean(1)?; // Pool embeddings
        let probabilities = candle_nn::ops::softmax(&logits, 1)?;
        
        let probs_vec = probabilities.to_vec1::<f32>()?;
        let predicted_class = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        let sentiment = match predicted_class {
            0 => "Negative",
            1 => "Neutral", 
            2 => "Positive",
            _ => "Unknown",
        };
        
        Ok(SentimentResult {
            sentiment: sentiment.to_string(),
            confidence: probs_vec[predicted_class],
            probabilities: probs_vec,
        })
    }
    
    pub fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let embeddings = self.get_embeddings(text)?;
        
        // Named Entity Recognition implementation
        // This would use a specialized NER model
        
        Ok(vec![])
    }
    
    pub fn answer_question(&self, context: &str, question: &str) -> Result<String> {
        let context_embeddings = self.get_embeddings(context)?;
        let question_embeddings = self.get_embeddings(question)?;
        
        // Question answering implementation
        // This would use a QA-specific model head
        
        Ok("Answer would be extracted here".to_string())
    }
}

#[derive(Debug)]
pub struct SentimentResult {
    pub sentiment: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
}

#[derive(Debug)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

fn main() -> Result<()> {
    println!("🤖 NLP Transformer Engine with Rust");
    println!("====================================");
    
    // Initialize NLP pipeline
    let pipeline = NLPPipeline::new("models/bert-base-uncased", "tokenizers/bert-tokenizer.json")?;
    
    // Example usage
    let text = "I love this new AI model! It's incredibly fast and accurate.";
    
    println!("📝 Analyzing text: {}", text);
    
    // Sentiment analysis
    match pipeline.classify_sentiment(text) {
        Ok(result) => {
            println!("😊 Sentiment: {} (confidence: {:.3})", result.sentiment, result.confidence);
        }
        Err(e) => println!("❌ Sentiment analysis failed: {}", e),
    }
    
    // Get embeddings
    match pipeline.get_embeddings(text) {
        Ok(embeddings) => {
            println!("🔢 Generated embeddings with shape: {:?}", embeddings.shape());
        }
        Err(e) => println!("❌ Embedding generation failed: {}", e),
    }
    
    println!("✅ NLP pipeline ready for production!");
    
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "rust-nlp-transformer"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = "0.3"
candle-nn = "0.3"
candle-transformers = "0.3"
tokenizers = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }`,
          language: 'toml',
        }
      }
    },
    // AI/ML Templates - Python
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
    {
      id: 'react-native-rust',
      name: 'React Native + Rust',
      description: 'High-performance React Native app with Rust native modules',
      category: 'Mobile',
      difficulty: 'Advanced',
      tags: ['React Native', 'Rust', 'Native Modules', 'Performance'],
      icon: <Smartphone className="w-6 h-6 text-cyan-400" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Build React Native apps with Rust for performance-critical operations',
      techStack: ['React Native', 'TypeScript', 'Rust', 'JNI', 'Swift'],
      features: [
        'Native Rust modules',
        'Cross-platform performance',
        'TypeScript integration',
        'Hot reloading',
        'Platform-specific optimizations'
      ],
      files: {
        'App.tsx': {
          content: `import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { NativeModules } from 'react-native';

const { RustModule } = NativeModules;

interface RustModule {
  processData(data: string): Promise<string>;
  calculateFibonacci(n: number): Promise<number>;
  encryptData(data: string, key: string): Promise<string>;
  decryptData(encryptedData: string, key: string): Promise<string>;
}

const App: React.FC = () => {
  const [result, setResult] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  const processWithRust = async () => {
    setLoading(true);
    try {
      const data = "Hello from React Native!";
      const processed = await RustModule.processData(data);
      setResult(processed);
    } catch (error) {
      Alert.alert('Error', 'Failed to process data with Rust');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const calculateFibonacci = async () => {
    setLoading(true);
    try {
      const n = 40; // Large number to show performance
      const startTime = Date.now();
      const fibResult = await RustModule.calculateFibonacci(n);
      const endTime = Date.now();
      
      setResult(\`Fibonacci(\${n}) = \${fibResult}\\nCalculated in \${endTime - startTime}ms\`);
    } catch (error) {
      Alert.alert('Error', 'Failed to calculate Fibonacci');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const testEncryption = async () => {
    setLoading(true);
    try {
      const data = "Sensitive information";
      const key = "my-secret-key";
      
      const encrypted = await RustModule.encryptData(data, key);
      const decrypted = await RustModule.decryptData(encrypted, key);
      
      setResult(\`Original: \${data}\\nEncrypted: \${encrypted}\\nDecrypted: \${decrypted}\`);
    } catch (error) {
      Alert.alert('Error', 'Failed to encrypt/decrypt data');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f8f9fa" />
      <ScrollView contentInsetAdjustmentBehavior="automatic" style={styles.scrollView}>
        <View style={styles.header}>
          <Text style={styles.title}>🦀 React Native + Rust</Text>
          <Text style={styles.subtitle}>High-Performance Native Modules</Text>
        </View>

        <View style={styles.content}>
          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={processWithRust}
            disabled={loading}
          >
            <Text style={styles.buttonText}>Process Data with Rust</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={calculateFibonacci}
            disabled={loading}
          >
            <Text style={styles.buttonText}>Calculate Fibonacci (Performance Test)</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={testEncryption}
            disabled={loading}
          >
            <Text style={styles.buttonText}>Test Encryption/Decryption</Text>
          </TouchableOpacity>

          {result ? (
            <View style={styles.resultContainer}>
              <Text style={styles.resultTitle}>Result:</Text>
              <Text style={styles.resultText}>{result}</Text>
            </View>
          ) : null}

          {loading && (
            <View style={styles.loadingContainer}>
              <Text style={styles.loadingText}>Processing with Rust...</Text>
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollView: {
    flex: 1,
  },
  header: {
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
  },
  content: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  button: {
    backgroundColor: '#e74c3c',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    marginBottom: 16,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  buttonDisabled: {
    backgroundColor: '#bdc3c7',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  resultContainer: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    marginTop: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 12,
  },
  resultText: {
    fontSize: 14,
    color: '#34495e',
    fontFamily: 'monospace',
    lineHeight: 20,
  },
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  loadingText: {
    fontSize: 16,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
});

export default App;`,
          language: 'typescript',
        },
        'rust_module.rs': {
          content: `use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;

#[no_mangle]
pub extern "system" fn Java_com_reactnativerustapp_RustModule_processData(
    env: JNIEnv,
    _class: JClass,
    input: JString,
) -> jstring {
    let input_str: String = env
        .get_string(input)
        .expect("Couldn't get java string!")
        .into();
    
    let processed = format!("Processed by Rust: {}", input_str.to_uppercase());
    
    let output = env
        .new_string(processed)
        .expect("Couldn't create java string!");
    
    output.into_inner()
}

#[no_mangle]
pub extern "system" fn Java_com_reactnativerustapp_RustModule_calculateFibonacci(
    _env: JNIEnv,
    _class: JClass,
    n: i32,
) -> i64 {
    fibonacci(n as u64) as i64
}

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_reactnativerustapp_RustModule_encryptData(
    env: JNIEnv,
    _class: JClass,
    data: JString,
    key: JString,
) -> jstring {
    let data_str: String = env
        .get_string(data)
        .expect("Couldn't get data string!")
        .into();
    
    let key_str: String = env
        .get_string(key)
        .expect("Couldn't get key string!")
        .into();
    
    // Simple XOR encryption for demonstration
    let encrypted = simple_encrypt(&data_str, &key_str);
    
    let output = env
        .new_string(encrypted)
        .expect("Couldn't create java string!");
    
    output.into_inner()
}

#[no_mangle]
pub extern "system" fn Java_com_reactnativerustapp_RustModule_decryptData(
    env: JNIEnv,
    _class: JClass,
    encrypted_data: JString,
    key: JString,
) -> jstring {
    let encrypted_str: String = env
        .get_string(encrypted_data)
        .expect("Couldn't get encrypted data string!")
        .into();
    
    let key_str: String = env
        .get_string(key)
        .expect("Couldn't get key string!")
        .into();
    
    // Simple XOR decryption (same as encryption for XOR)
    let decrypted = simple_encrypt(&encrypted_str, &key_str);
    
    let output = env
        .new_string(decrypted)
        .expect("Couldn't create java string!");
    
    output.into_inner()
}

fn simple_encrypt(data: &str, key: &str) -> String {
    let key_bytes = key.as_bytes();
    let key_len = key_bytes.len();
    
    data.bytes()
        .enumerate()
        .map(|(i, byte)| byte ^ key_bytes[i % key_len])
        .collect::<Vec<u8>>()
        .iter()
        .map(|&b| format!("{:02x}", b))
        .collect::<String>()
}`,
          language: 'rust',
        }
      }
    },
    {
      id: 'kotlin-android-rust',
      name: 'Android Kotlin + Rust',
      description: 'Native Android app with Kotlin UI and Rust performance modules',
      category: 'Mobile',
      difficulty: 'Advanced',
      tags: ['Android', 'Kotlin', 'Rust', 'JNI', 'Performance'],
      icon: <Smartphone className="w-6 h-6 text-green-400" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Build high-performance Android apps with Rust native libraries',
      techStack: ['Android', 'Kotlin', 'Rust', 'JNI', 'Gradle'],
      features: [
        'Native Rust libraries',
        'JNI integration',
        'Kotlin coroutines',
        'Material Design',
        'Performance optimization'
      ],
      files: {
        'MainActivity.kt': {
          content: `package com.example.kotlinrustapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    
    companion object {
        init {
            System.loadLibrary("rustlib")
        }
    }
    
    // Native method declarations
    external fun processStringWithRust(input: String): String
    external fun calculatePrimeNumbers(limit: Int): IntArray
    external fun performMatrixMultiplication(matrixA: FloatArray, matrixB: FloatArray, size: Int): FloatArray
    external fun compressData(data: String): ByteArray
    external fun decompressData(compressedData: ByteArray): String
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            KotlinRustAppTheme {
                MainScreen()
            }
        }
    }
    
    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun MainScreen() {
        var results by remember { mutableStateOf(listOf<String>()) }
        var isLoading by remember { mutableStateOf(false) }
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "🦀 Android Kotlin + Rust",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 8.dp)
            )
            
            Text(
                text = "High-Performance Native Integration",
                fontSize = 16.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 32.dp)
            )
            
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.weight(1f)
            ) {
                item {
                    PerformanceTestButton(
                        title = "String Processing",
                        description = "Process text with Rust",
                        isLoading = isLoading,
                        onClick = {
                            isLoading = true
                            lifecycleScope.launch {
                                val result = withContext(Dispatchers.Default) {
                                    val input = "Hello from Android Kotlin!"
                                    val processed = processStringWithRust(input)
                                    "Input: $input\\nOutput: $processed"
                                }
                                results = results + result
                                isLoading = false
                            }
                        }
                    )
                }
                
                item {
                    PerformanceTestButton(
                        title = "Prime Number Calculation",
                        description = "Calculate primes up to 10,000",
                        isLoading = isLoading,
                        onClick = {
                            isLoading = true
                            lifecycleScope.launch {
                                val result = withContext(Dispatchers.Default) {
                                    val startTime = System.currentTimeMillis()
                                    val primes = calculatePrimeNumbers(10000)
                                    val endTime = System.currentTimeMillis()
                                    "Found \${primes.size} primes up to 10,000\\nTime: \${endTime - startTime}ms"
                                }
                                results = results + result
                                isLoading = false
                            }
                        }
                    )
                }
                
                item {
                    PerformanceTestButton(
                        title = "Matrix Multiplication",
                        description = "100x100 matrix multiplication",
                        isLoading = isLoading,
                        onClick = {
                            isLoading = true
                            lifecycleScope.launch {
                                val result = withContext(Dispatchers.Default) {
                                    val size = 100
                                    val matrixA = FloatArray(size * size) { it.toFloat() }
                                    val matrixB = FloatArray(size * size) { (it * 2).toFloat() }
                                    
                                    val startTime = System.currentTimeMillis()
                                    val resultMatrix = performMatrixMultiplication(matrixA, matrixB, size)
                                    val endTime = System.currentTimeMillis()
                                    
                                    "Matrix multiplication (\${size}x\${size})\\nTime: \${endTime - startTime}ms\\nResult sum: \${resultMatrix.sum()}"
                                }
                                results = results + result
                                isLoading = false
                            }
                        }
                    )
                }
                
                item {
                    PerformanceTestButton(
                        title = "Data Compression",
                        description = "Compress and decompress text",
                        isLoading = isLoading,
                        onClick = {
                            isLoading = true
                            lifecycleScope.launch {
                                val result = withContext(Dispatchers.Default) {
                                    val originalData = "This is a test string for compression. ".repeat(100)
                                    val compressed = compressData(originalData)
                                    val decompressed = decompressData(compressed)
                                    
                                    val compressionRatio = (compressed.size.toFloat() / originalData.length) * 100
                                    
                                    "Original size: \${originalData.length} bytes\\nCompressed size: \${compressed.size} bytes\\nCompression ratio: \${String.format("%.1f", compressionRatio)}%\\nDecompression successful: \${originalData == decompressed}"
                                }
                                results = results + result
                                isLoading = false
                            }
                        }
                    )
                }
            }
            
            if (results.isNotEmpty()) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = "Results:",
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )
                        
                        results.takeLast(3).forEach { result ->
                            Text(
                                text = result,
                                fontSize = 12.sp,
                                modifier = Modifier.padding(bottom = 8.dp)
                            )
                            Divider()
                        }
                    }
                }
            }
        }
    }
    
    @Composable
    fun PerformanceTestButton(
        title: String,
        description: String,
        isLoading: Boolean,
        onClick: () -> Unit
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            onClick = if (!isLoading) onClick else { {} }
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = title,
                    fontWeight = FontWeight.Medium,
                    fontSize = 16.sp
                )
                Text(
                    text = description,
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
                
                if (isLoading) {
                    LinearProgressIndicator(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun KotlinRustAppTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(),
        content = content
    )
}`,
          language: 'kotlin',
        },
        'rust_lib.rs': {
          content: `use jni::objects::{JClass, JString, JByteArray};
use jni::sys::{jstring, jintArray, jfloatArray, jbyteArray};
use jni::JNIEnv;
use std::ffi::CString;

#[no_mangle]
pub extern "system" fn Java_com_example_kotlinrustapp_MainActivity_processStringWithRust(
    env: JNIEnv,
    _class: JClass,
    input: JString,
) -> jstring {
    let input_str: String = env
        .get_string(input)
        .expect("Couldn't get java string!")
        .into();
    
    // Process the string with Rust
    let processed = format!("🦀 Rust processed: {} (length: {})", 
                          input_str.to_uppercase(), 
                          input_str.len());
    
    let output = env
        .new_string(processed)
        .expect("Couldn't create java string!");
    
    output.into_inner()
}

#[no_mangle]
pub extern "system" fn Java_com_example_kotlinrustapp_MainActivity_calculatePrimeNumbers(
    env: JNIEnv,
    _class: JClass,
    limit: i32,
) -> jintArray {
    let primes = sieve_of_eratosthenes(limit as usize);
    let primes_i32: Vec<i32> = primes.into_iter().map(|x| x as i32).collect();
    
    let result = env.new_int_array(primes_i32.len() as i32).unwrap();
    env.set_int_array_region(result, 0, &primes_i32).unwrap();
    
    result
}

fn sieve_of_eratosthenes(limit: usize) -> Vec<usize> {
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    if limit > 0 {
        is_prime[1] = false;
    }
    
    for i in 2..=((limit as f64).sqrt() as usize) {
        if is_prime[i] {
            for j in ((i * i)..=limit).step_by(i) {
                is_prime[j] = false;
            }
        }
    }
    
    (2..=limit).filter(|&i| is_prime[i]).collect()
}

#[no_mangle]
pub extern "system" fn Java_com_example_kotlinrustapp_MainActivity_performMatrixMultiplication(
    env: JNIEnv,
    _class: JClass,
    matrix_a: jfloatArray,
    matrix_b: jfloatArray,
    size: i32,
) -> jfloatArray {
    let size = size as usize;
    
    let a_elements = env.get_float_array_elements(matrix_a, jni::objects::ReleaseMode::NoCopyBack).unwrap();
    let b_elements = env.get_float_array_elements(matrix_b, jni::objects::ReleaseMode::NoCopyBack).unwrap();
    
    let a_slice = unsafe { std::slice::from_raw_parts(a_elements.as_ptr(), size * size) };
    let b_slice = unsafe { std::slice::from_raw_parts(b_elements.as_ptr(), size * size) };
    
    let mut result = vec![0.0f32; size * size];
    
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                result[i * size + j] += a_slice[i * size + k] * b_slice[k * size + j];
            }
        }
    }
    
    let result_array = env.new_float_array(result.len() as i32).unwrap();
    env.set_float_array_region(result_array, 0, &result).unwrap();
    
    result_array
}

#[no_mangle]
pub extern "system" fn Java_com_example_kotlinrustapp_MainActivity_compressData(
    env: JNIEnv,
    _class: JClass,
    data: JString,
) -> jbyteArray {
    let data_str: String = env
        .get_string(data)
        .expect("Couldn't get java string!")
        .into();
    
    // Simple run-length encoding for demonstration
    let compressed = run_length_encode(&data_str);
    
    let result = env.new_byte_array(compressed.len() as i32).unwrap();
    env.set_byte_array_region(result, 0, &compressed).unwrap();
    
    result
}

#[no_mangle]
pub extern "system" fn Java_com_example_kotlinrustapp_MainActivity_decompressData(
    env: JNIEnv,
    _class: JClass,
    compressed_data: JByteArray,
) -> jstring {
    let compressed_elements = env.get_byte_array_elements(compressed_data, jni::objects::ReleaseMode::NoCopyBack).unwrap();
    let compressed_slice = unsafe { 
        std::slice::from_raw_parts(
            compressed_elements.as_ptr() as *const u8, 
            env.get_array_length(compressed_data).unwrap() as usize
        ) 
    };
    
    let decompressed = run_length_decode(compressed_slice);
    
    let output = env
        .new_string(decompressed)
        .expect("Couldn't create java string!");
    
    output.into_inner()
}

fn run_length_encode(data: &str) -> Vec<i8> {
    let mut result = Vec::new();
    let bytes = data.as_bytes();
    
    if bytes.is_empty() {
        return result;
    }
    
    let mut current_char = bytes[0];
    let mut count = 1u8;
    
    for &byte in &bytes[1..] {
        if byte == current_char && count < 255 {
            count += 1;
        } else {
            result.push(count as i8);
            result.push(current_char as i8);
            current_char = byte;
            count = 1;
        }
    }
    
    result.push(count as i8);
    result.push(current_char as i8);
    
    result
}

fn run_length_decode(compressed: &[u8]) -> String {
    let mut result = Vec::new();
    
    for chunk in compressed.chunks(2) {
        if chunk.len() == 2 {
            let count = chunk[0];
            let byte = chunk[1];
            
            for _ in 0..count {
                result.push(byte);
            }
        }
    }
    
    String::from_utf8_lossy(&result).to_string()
}`,
          language: 'rust',
        }
      }
    },
    {
      id: 'swift-ios-rust',
      name: 'iOS Swift + Rust',
      description: 'Native iOS app with Swift UI and Rust performance libraries',
      category: 'Mobile',
      difficulty: 'Advanced',
      tags: ['iOS', 'Swift', 'Rust', 'FFI', 'Performance'],
      icon: <Smartphone className="w-6 h-6 text-blue-500" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Build high-performance iOS apps with Rust native libraries',
      techStack: ['iOS', 'Swift', 'Rust', 'FFI', 'Xcode'],
      features: [
        'Native Rust libraries',
        'Swift FFI integration',
        'SwiftUI interface',
        'iOS-specific optimizations',
        'Memory-safe interop'
      ],
      files: {
        'ContentView.swift': {
          content: `import SwiftUI

struct ContentView: View {
    @StateObject private var rustInterface = RustInterface()
    @State private var results: [String] = []
    @State private var isLoading = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                headerView
                
                ScrollView {
                    LazyVStack(spacing: 16) {
                        performanceTestButton(
                            title: "String Processing",
                            description: "Process text with Rust",
                            action: testStringProcessing
                        )
                        
                        performanceTestButton(
                            title: "Fibonacci Calculation",
                            description: "Calculate Fibonacci(40)",
                            action: testFibonacci
                        )
                        
                        performanceTestButton(
                            title: "Prime Numbers",
                            description: "Find primes up to 10,000",
                            action: testPrimeNumbers
                        )
                        
                        performanceTestButton(
                            title: "Image Processing",
                            description: "Apply filters with Rust",
                            action: testImageProcessing
                        )
                        
                        performanceTestButton(
                            title: "Cryptography",
                            description: "Encrypt/decrypt data",
                            action: testCryptography
                        )
                    }
                    .padding(.horizontal)
                }
                
                if !results.isEmpty {
                    resultsView
                }
                
                Spacer()
            }
            .navigationTitle("🦀 iOS + Rust")
            .navigationBarTitleDisplayMode(.large)
        }
    }
    
    private var headerView: some View {
        VStack(spacing: 8) {
            Text("🦀 iOS Swift + Rust")
                .font(.title)
                .fontWeight(.bold)
            
            Text("High-Performance Native Integration")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.top)
    }
    
    private func performanceTestButton(title: String, description: String, action: @escaping () -> Void) -> some View {
        Button(action: {
            if !isLoading {
                action()
            }
        }) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(title)
                            .font(.headline)
                            .foregroundColor(.primary)
                        
                        Text(description)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    if isLoading {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "chevron.right")
                            .foregroundColor(.accentColor)
                    }
                }
                
                if isLoading {
                    ProgressView()
                        .progressViewStyle(LinearProgressViewStyle())
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
        .disabled(isLoading)
    }
    
    private var resultsView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Results:")
                .font(.headline)
                .padding(.horizontal)
            
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(results.suffix(3).reversed(), id: \\.self) { result in
                        Text(result)
                            .font(.caption)
                            .padding()
                            .background(Color(.systemBackground))
                            .cornerRadius(8)
                            .shadow(radius: 1)
                    }
                }
                .padding(.horizontal)
            }
            .frame(maxHeight: 200)
        }
    }
    
    private func testStringProcessing() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let input = "Hello from iOS Swift!"
            let result = rustInterface.processString(input)
            
            DispatchQueue.main.async {
                results.append("String Processing:\\nInput: \\(input)\\nOutput: \\(result)")
                isLoading = false
            }
        }
    }
    
    private func testFibonacci() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let n: UInt32 = 40
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = rustInterface.calculateFibonacci(n)
            let endTime = CFAbsoluteTimeGetCurrent()
            let duration = (endTime - startTime) * 1000
            
            DispatchQueue.main.async {
                results.append("Fibonacci(\\(n)) = \\(result)\\nTime: \\(String(format: "%.2f", duration))ms")
                isLoading = false
            }
        }
    }
    
    private func testPrimeNumbers() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let limit: UInt32 = 10000
            let startTime = CFAbsoluteTimeGetCurrent()
            let primes = rustInterface.findPrimes(upTo: limit)
            let endTime = CFAbsoluteTimeGetCurrent()
            let duration = (endTime - startTime) * 1000
            
            DispatchQueue.main.async {
                results.append("Found \\(primes.count) primes up to \\(limit)\\nTime: \\(String(format: "%.2f", duration))ms")
                isLoading = false
            }
        }
    }
    
    private func testImageProcessing() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            // Simulate image processing
            let width: UInt32 = 1000
            let height: UInt32 = 1000
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = rustInterface.processImage(width: width, height: height)
            let endTime = CFAbsoluteTimeGetCurrent()
            let duration = (endTime - startTime) * 1000
            
            DispatchQueue.main.async {
                results.append("Image Processing (\\(width)x\\(height))\\nProcessed: \\(result)\\nTime: \\(String(format: "%.2f", duration))ms")
                isLoading = false
            }
        }
    }
    
    private func testCryptography() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let data = "Sensitive information for encryption"
            let key = "my-secret-key"
            
            let encrypted = rustInterface.encryptData(data, key: key)
            let decrypted = rustInterface.decryptData(encrypted, key: key)
            
            DispatchQueue.main.async {
                results.append("Cryptography Test:\\nOriginal: \\(data)\\nEncrypted: \\(encrypted)\\nDecrypted: \\(decrypted)\\nSuccess: \\(data == decrypted)")
                isLoading = false
            }
        }
    }
}

class RustInterface: ObservableObject {
    
    func processString(_ input: String) -> String {
        return input.withCString { cString in
            let result = rust_process_string(cString)
            return String(cString: result!)
        }
    }
    
    func calculateFibonacci(_ n: UInt32) -> UInt64 {
        return rust_fibonacci(n)
    }
    
    func findPrimes(upTo limit: UInt32) -> [UInt32] {
        var count: UInt32 = 0
        let primesPtr = rust_find_primes(limit, &count)
        
        guard let primes = primesPtr else { return [] }
        
        let primesArray = Array(UnsafeBufferPointer(start: primes, count: Int(count)))
        rust_free_primes(primes)
        
        return primesArray
    }
    
    func processImage(width: UInt32, height: UInt32) -> String {
        let result = rust_process_image(width, height)
        return String(cString: result!)
    }
    
    func encryptData(_ data: String, key: String) -> String {
        return data.withCString { dataPtr in
            key.withCString { keyPtr in
                let result = rust_encrypt_data(dataPtr, keyPtr)
                return String(cString: result!)
            }
        }
    }
    
    func decryptData(_ encryptedData: String, key: String) -> String {
        return encryptedData.withCString { dataPtr in
            key.withCString { keyPtr in
                let result = rust_decrypt_data(dataPtr, keyPtr)
                return String(cString: result!)
            }
        }
    }
}

// C function declarations for Rust FFI
@_silgen_name("rust_process_string")
func rust_process_string(_ input: UnsafePointer<CChar>) -> UnsafePointer<CChar>?

@_silgen_name("rust_fibonacci")
func rust_fibonacci(_ n: UInt32) -> UInt64

@_silgen_name("rust_find_primes")
func rust_find_primes(_ limit: UInt32, _ count: UnsafeMutablePointer<UInt32>) -> UnsafeMutablePointer<UInt32>?

@_silgen_name("rust_free_primes")
func rust_free_primes(_ primes: UnsafeMutablePointer<UInt32>)

@_silgen_name("rust_process_image")
func rust_process_image(_ width: UInt32, _ height: UInt32) -> UnsafePointer<CChar>?

@_silgen_name("rust_encrypt_data")
func rust_encrypt_data(_ data: UnsafePointer<CChar>, _ key: UnsafePointer<CChar>) -> UnsafePointer<CChar>?

@_silgen_name("rust_decrypt_data")
func rust_decrypt_data(_ encrypted_data: UnsafePointer<CChar>, _ key: UnsafePointer<CChar>) -> UnsafePointer<CChar>?

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}`,
          language: 'swift',
        },
        'rust_lib.rs': {
          content: `use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn rust_process_string(input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(input) };
    let input_str = c_str.to_str().unwrap();
    
    let processed = format!("🦀 Rust processed: {} (length: {})", 
                          input_str.to_uppercase(), 
                          input_str.len());
    
    let c_string = CString::new(processed).unwrap();
    c_string.into_raw()
}

#[no_mangle]
pub extern "C" fn rust_fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let mut a = 0u64;
            let mut b = 1u64;
            for _ in 2..=n {
                let temp = a + b;
                a = b;
                b = temp;
            }
            b
        }
    }
}

#[no_mangle]
pub extern "C" fn rust_find_primes(limit: u32, count: *mut u32) -> *mut u32 {
    let primes = sieve_of_eratosthenes(limit as usize);
    
    unsafe {
        *count = primes.len() as u32;
    }
    
    let mut primes_u32: Vec<u32> = primes.into_iter().map(|x| x as u32).collect();
    let ptr = primes_u32.as_mut_ptr();
    std::mem::forget(primes_u32);
    ptr
}

#[no_mangle]
pub extern "C" fn rust_free_primes(primes: *mut u32) {
    if !primes.is_null() {
        unsafe {
            let _ = Box::from_raw(primes);
        }
    }
}

fn sieve_of_eratosthenes(limit: usize) -> Vec<usize> {
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    if limit > 0 {
        is_prime[1] = false;
    }
    
    for i in 2..=((limit as f64).sqrt() as usize) {
        if is_prime[i] {
            for j in ((i * i)..=limit).step_by(i) {
                is_prime[j] = false;
            }
        }
    }
    
    (2..=limit).filter(|&i| is_prime[i]).collect()
}

#[no_mangle]
pub extern "C" fn rust_process_image(width: u32, height: u32) -> *const c_char {
    // Simulate image processing
    let pixels_processed = width * height;
    let result = format!("Processed {} pixels ({}x{})", pixels_processed, width, height);
    
    let c_string = CString::new(result).unwrap();
    c_string.into_raw()
}

#[no_mangle]
pub extern "C" fn rust_encrypt_data(data: *const c_char, key: *const c_char) -> *const c_char {
    let data_str = unsafe { CStr::from_ptr(data).to_str().unwrap() };
    let key_str = unsafe { CStr::from_ptr(key).to_str().unwrap() };
    
    let encrypted = simple_encrypt(data_str, key_str);
    let c_string = CString::new(encrypted).unwrap();
    c_string.into_raw()
}

#[no_mangle]
pub extern "C" fn rust_decrypt_data(encrypted_data: *const c_char, key: *const c_char) -> *const c_char {
    let encrypted_str = unsafe { CStr::from_ptr(encrypted_data).to_str().unwrap() };
    let key_str = unsafe { CStr::from_ptr(key).to_str().unwrap() };
    
    let decrypted = simple_decrypt(encrypted_str, key_str);
    let c_string = CString::new(decrypted).unwrap();
    c_string.into_raw()
}

fn simple_encrypt(data: &str, key: &str) -> String {
    let key_bytes = key.as_bytes();
    let key_len = key_bytes.len();
    
    data.bytes()
        .enumerate()
        .map(|(i, byte)| byte ^ key_bytes[i % key_len])
        .collect::<Vec<u8>>()
        .iter()
        .map(|&b| format!("{:02x}", b))
        .collect::<String>()
}

fn simple_decrypt(encrypted_hex: &str, key: &str) -> String {
    let key_bytes = key.as_bytes();
    let key_len = key_bytes.len();
    
    let encrypted_bytes: Vec<u8> = (0..encrypted_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&encrypted_hex[i..i + 2], 16).unwrap())
        .collect();
    
    encrypted_bytes
        .iter()
        .enumerate()
        .map(|(i, &byte)| (byte ^ key_bytes[i % key_len]) as char)
        .collect()
}`,
          language: 'rust',
        }
      }
    },
    // IoT Templates
    {
      id: 'rust-iot-sensor-hub',
      name: 'IoT Sensor Hub',
      description: 'Rust-based IoT sensor data collection and processing hub with MQTT',
      category: 'IoT',
      difficulty: 'Intermediate',
      tags: ['IoT', 'Sensors', 'MQTT', 'Real-time', 'Rust'],
      icon: <Thermometer className="w-6 h-6 text-orange-400" />,
      estimatedTime: '2-3 weeks',
      useCase: 'Collect and process sensor data from multiple IoT devices',
      techStack: ['Rust', 'MQTT', 'InfluxDB', 'Tokio', 'Serde'],
      features: [
        'Multi-sensor support',
        'MQTT communication',
        'Real-time data processing',
        'Time-series database',
        'Web dashboard'
      ],
      files: {
        'main.rs': {
          content: `use tokio::time::{interval, Duration};
use rumqttc::{MqttOptions, Client, QoS, Event, Packet};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    pub sensor_id: String,
    pub sensor_type: SensorType,
    pub value: f64,
    pub unit: String,
    pub timestamp: u64,
    pub location: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorType {
    Temperature,
    Humidity,
    Pressure,
    Light,
    Motion,
    AirQuality,
    SoilMoisture,
    WaterLevel,
}

#[derive(Debug)]
pub struct IoTHub {
    mqtt_client: Client,
    sensors: Arc<Mutex<HashMap<String, SensorReading>>>,
    data_store: Arc<Mutex<Vec<SensorReading>>>,
}

impl IoTHub {
    pub fn new(broker_host: &str, broker_port: u16) -> Result<Self, Box<dyn std::error::Error>> {
        let mut mqttoptions = MqttOptions::new("iot-hub", broker_host, broker_port);
        mqttoptions.set_keep_alive(Duration::from_secs(60));
        
        let (client, mut connection) = Client::new(mqttoptions, 10);
        
        // Subscribe to sensor topics
        client.subscribe("sensors/+/temperature", QoS::AtMostOnce)?;
        client.subscribe("sensors/+/humidity", QoS::AtMostOnce)?;
        client.subscribe("sensors/+/pressure", QoS::AtMostOnce)?;
        client.subscribe("sensors/+/light", QoS::AtMostOnce)?;
        client.subscribe("sensors/+/motion", QoS::AtMostOnce)?;
        client.subscribe("sensors/+/air_quality", QoS::AtMostOnce)?;
        
        let sensors = Arc::new(Mutex::new(HashMap::new()));
        let data_store = Arc::new(Mutex::new(Vec::new()));
        
        let sensors_clone = sensors.clone();
        let data_store_clone = data_store.clone();
        
        // Handle incoming MQTT messages
        tokio::spawn(async move {
            for (_, notification) in connection.iter().enumerate() {
                match notification {
                    Ok(Event::Incoming(Packet::Publish(publish))) => {
                        if let Ok(reading) = Self::parse_sensor_data(&publish.topic, &publish.payload) {
                            println!("📊 Received sensor data: {:?}", reading);
                            
                            // Store latest reading
                            sensors_clone.lock().await.insert(reading.sensor_id.clone(), reading.clone());
                            
                            // Store historical data
                            data_store_clone.lock().await.push(reading);
                        }
                    }
                    Ok(_) => {}
                    Err(e) => println!("❌ MQTT Error: {}", e),
                }
            }
        });
        
        Ok(Self {
            mqtt_client: client,
            sensors,
            data_store,
        })
    }
    
    fn parse_sensor_data(topic: &str, payload: &[u8]) -> Result<SensorReading, Box<dyn std::error::Error>> {
        let topic_parts: Vec<&str> = topic.split('/').collect();
        if topic_parts.len() != 3 {
            return Err("Invalid topic format".into());
        }
        
        let sensor_id = topic_parts[1].to_string();
        let sensor_type_str = topic_parts[2];
        
        let sensor_type = match sensor_type_str {
            "temperature" => SensorType::Temperature,
            "humidity" => SensorType::Humidity,
            "pressure" => SensorType::Pressure,
            "light" => SensorType::Light,
            "motion" => SensorType::Motion,
            "air_quality" => SensorType::AirQuality,
            _ => return Err("Unknown sensor type".into()),
        };
        
        let payload_str = String::from_utf8(payload.to_vec())?;
        let sensor_data: serde_json::Value = serde_json::from_str(&payload_str)?;
        
        let reading = SensorReading {
            sensor_id,
            sensor_type,
            value: sensor_data["value"].as_f64().unwrap_or(0.0),
            unit: sensor_data["unit"].as_str().unwrap_or("").to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            location: sensor_data["location"].as_str().unwrap_or("unknown").to_string(),
        };
        
        Ok(reading)
    }
    
    pub async fn get_latest_readings(&self) -> HashMap<String, SensorReading> {
        self.sensors.lock().await.clone()
    }
    
    pub async fn get_historical_data(&self, sensor_id: &str, limit: usize) -> Vec<SensorReading> {
        let data = self.data_store.lock().await;
        data.iter()
            .filter(|reading| reading.sensor_id == sensor_id)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    pub async fn publish_command(&self, device_id: &str, command: &str, payload: &str) -> Result<(), Box<dyn std::error::Error>> {
        let topic = format!("commands/{}/{}", device_id, command);
        self.mqtt_client.publish(topic, QoS::AtLeastOnce, false, payload)?;
        Ok(())
    }
    
    pub async fn start_monitoring(&self) {
        let mut interval = interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            let readings = self.get_latest_readings().await;
            println!("🔍 Current sensor readings:");
            
            for (sensor_id, reading) in readings {
                println!("  {} ({}): {:.2} {} at {}", 
                        sensor_id, 
                        format!("{:?}", reading.sensor_type),
                        reading.value, 
                        reading.unit,
                        reading.location);
                
                // Check for alerts
                self.check_alerts(&reading).await;
            }
            
            println!("📈 Total data points collected: {}", self.data_store.lock().await.len());
        }
    }
    
    async fn check_alerts(&self, reading: &SensorReading) {
        match reading.sensor_type {
            SensorType::Temperature => {
                if reading.value > 35.0 {
                    println!("🚨 HIGH TEMPERATURE ALERT: {} at {}°C", reading.sensor_id, reading.value);
                } else if reading.value < 5.0 {
                    println!("🧊 LOW TEMPERATURE ALERT: {} at {}°C", reading.sensor_id, reading.value);
                }
            }
            SensorType::Humidity => {
                if reading.value > 80.0 {
                    println!("💧 HIGH HUMIDITY ALERT: {} at {}%", reading.sensor_id, reading.value);
                }
            }
            SensorType::AirQuality => {
                if reading.value > 150.0 {
                    println!("🏭 POOR AIR QUALITY ALERT: {} AQI: {}", reading.sensor_id, reading.value);
                }
            }
            _ => {}
        }
    }
}

// Simulate sensor data for testing
async fn simulate_sensors(mqtt_client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let mut interval = interval(Duration::from_secs(10));
    let mut counter = 0;
    
    loop {
        interval.tick().await;
        counter += 1;
        
        // Simulate temperature sensor
        let temp_data = serde_json::json!({
            "value": 20.0 + (counter as f64 * 0.5) % 15.0,
            "unit": "°C",
            "location": "Living Room"
        });
        mqtt_client.publish("sensors/temp_001/temperature", QoS::AtMostOnce, false, temp_data.to_string())?;
        
        // Simulate humidity sensor
        let humidity_data = serde_json::json!({
            "value": 45.0 + (counter as f64 * 0.3) % 20.0,
            "unit": "%",
            "location": "Living Room"
        });
        mqtt_client.publish("sensors/hum_001/humidity", QoS::AtMostOnce, false, humidity_data.to_string())?;
        
        // Simulate air quality sensor
        let air_quality_data = serde_json::json!({
            "value": 50.0 + (counter as f64 * 2.0) % 100.0,
            "unit": "AQI",
            "location": "Outdoor"
        });
        mqtt_client.publish("sensors/air_001/air_quality", QoS::AtMostOnce, false, air_quality_data.to_string())?;
        
        println!("📡 Simulated sensor data sent (iteration {})", counter);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏠 IoT Sensor Hub Starting...");
    println!("==============================");
    
    // Create IoT Hub
    let hub = IoTHub::new("localhost", 1883)?;
    
    // Create a separate MQTT client for simulation
    let mut sim_options = rumqttc::MqttOptions::new("sensor-simulator", "localhost", 1883);
    sim_options.set_keep_alive(Duration::from_secs(60));
    let (sim_client, _) = rumqttc::Client::new(sim_options, 10);
    
    // Start sensor simulation
    let sim_client_clone = sim_client.clone();
    tokio::spawn(async move {
        if let Err(e) = simulate_sensors(&sim_client_clone).await {
            println!("❌ Sensor simulation error: {}", e);
        }
    });
    
    // Start monitoring
    println!("✅ IoT Hub initialized successfully!");
    println!("🔍 Starting sensor monitoring...");
    
    hub.start_monitoring().await;
    
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "iot-sensor-hub"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
rumqttc = "0.24"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
influxdb = "0.7"
reqwest = { version = "0.11", features = ["json"] }
uuid = { version = "1.0", features = ["v4"] }`,
          language: 'toml',
        }
      }
    },
    {
      id: 'rust-smart-home',
      name: 'Smart Home Controller',
      description: 'Comprehensive smart home automation system with device control and scheduling',
      category: 'IoT',
      difficulty: 'Advanced',
      tags: ['Smart Home', 'Automation', 'Device Control', 'Scheduling', 'Rust'],
      icon: <Home className="w-6 h-6 text-blue-400" />,
      estimatedTime: '4-5 weeks',
      useCase: 'Control and automate smart home devices with intelligent scheduling',
      techStack: ['Rust', 'MQTT', 'WebSocket', 'SQLite', 'Actix-web'],
      features: [
        'Device discovery and control',
        'Automated scheduling',
        'Energy monitoring',
        'Security system integration',
        'Mobile app support'
      ],
      files: {
        'main.rs': {
          content: `use actix_web::{web, App, HttpServer, Result, HttpResponse, middleware::Logger};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartDevice {
    pub id: String,
    pub name: String,
    pub device_type: DeviceType,
    pub location: String,
    pub status: DeviceStatus,
    pub properties: HashMap<String, serde_json::Value>,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    Light,
    Thermostat,
    SecurityCamera,
    DoorLock,
    MotionSensor,
    SmartPlug,
    Speaker,
    TV,
    AirConditioner,
    WaterHeater,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceStatus {
    Online,
    Offline,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationRule {
    pub id: String,
    pub name: String,
    pub trigger: Trigger,
    pub actions: Vec<Action>,
    pub enabled: bool,
    pub schedule: Option<Schedule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trigger {
    TimeOfDay { hour: u8, minute: u8 },
    SensorValue { device_id: String, condition: String, value: f64 },
    DeviceState { device_id: String, state: String },
    Sunrise,
    Sunset,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub device_id: String,
    pub command: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub days_of_week: Vec<u8>, // 0 = Sunday, 1 = Monday, etc.
    pub start_date: Option<String>,
    pub end_date: Option<String>,
}

#[derive(Debug)]
pub struct SmartHomeController {
    devices: Arc<Mutex<HashMap<String, SmartDevice>>>,
    automation_rules: Arc<Mutex<HashMap<String, AutomationRule>>>,
    energy_data: Arc<Mutex<Vec<EnergyReading>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyReading {
    pub device_id: String,
    pub power_consumption: f64, // watts
    pub timestamp: u64,
}

impl SmartHomeController {
    pub fn new() -> Self {
        Self {
            devices: Arc::new(Mutex::new(HashMap::new())),
            automation_rules: Arc::new(Mutex::new(HashMap::new())),
            energy_data: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub async fn add_device(&self, device: SmartDevice) {
        let mut devices = self.devices.lock().await;
        devices.insert(device.id.clone(), device);
    }
    
    pub async fn get_devices(&self) -> Vec<SmartDevice> {
        let devices = self.devices.lock().await;
        devices.values().cloned().collect()
    }
    
    pub async fn control_device(&self, device_id: &str, command: &str, parameters: HashMap<String, serde_json::Value>) -> Result<(), String> {
        let mut devices = self.devices.lock().await;
        
        if let Some(device) = devices.get_mut(device_id) {
            match device.device_type {
                DeviceType::Light => {
                    match command {
                        "turn_on" => {
                            device.properties.insert("state".to_string(), serde_json::Value::String("on".to_string()));
                            if let Some(brightness) = parameters.get("brightness") {
                                device.properties.insert("brightness".to_string(), brightness.clone());
                            }
                        }
                        "turn_off" => {
                            device.properties.insert("state".to_string(), serde_json::Value::String("off".to_string()));
                        }
                        "set_brightness" => {
                            if let Some(brightness) = parameters.get("brightness") {
                                device.properties.insert("brightness".to_string(), brightness.clone());
                            }
                        }
                        _ => return Err(format!("Unknown command for light: {}", command)),
                    }
                }
                DeviceType::Thermostat => {
                    match command {
                        "set_temperature" => {
                            if let Some(temp) = parameters.get("temperature") {
                                device.properties.insert("target_temperature".to_string(), temp.clone());
                            }
                        }
                        "set_mode" => {
                            if let Some(mode) = parameters.get("mode") {
                                device.properties.insert("mode".to_string(), mode.clone());
                            }
                        }
                        _ => return Err(format!("Unknown command for thermostat: {}", command)),
                    }
                }
                DeviceType::DoorLock => {
                    match command {
                        "lock" => {
                            device.properties.insert("locked".to_string(), serde_json::Value::Bool(true));
                        }
                        "unlock" => {
                            device.properties.insert("locked".to_string(), serde_json::Value::Bool(false));
                        }
                        _ => return Err(format!("Unknown command for door lock: {}", command)),
                    }
                }
                _ => {
                    // Generic property update for other devices
                    for (key, value) in parameters {
                        device.properties.insert(key, value);
                    }
                }
            }
            
            device.last_updated = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            println!("🏠 Device {} controlled: {}", device.name, command);
            Ok(())
        } else {
            Err(format!("Device not found: {}", device_id))
        }
    }
    
    pub async fn add_automation_rule(&self, rule: AutomationRule) {
        let mut rules = self.automation_rules.lock().await;
        rules.insert(rule.id.clone(), rule);
    }
    
    pub async fn execute_automation(&self, rule_id: &str) -> Result<(), String> {
        let rules = self.automation_rules.lock().await;
        
        if let Some(rule) = rules.get(rule_id) {
            if !rule.enabled {
                return Err("Automation rule is disabled".to_string());
            }
            
            println!("🤖 Executing automation rule: {}", rule.name);
            
            for action in &rule.actions {
                if let Err(e) = self.control_device(&action.device_id, &action.command, action.parameters.clone()).await {
                    println!("❌ Failed to execute action: {}", e);
                }
            }
            
            Ok(())
        } else {
            Err(format!("Automation rule not found: {}", rule_id))
        }
    }
    
    pub async fn get_energy_consumption(&self, device_id: Option<&str>) -> Vec<EnergyReading> {
        let energy_data = self.energy_data.lock().await;
        
        if let Some(device_id) = device_id {
            energy_data.iter()
                .filter(|reading| reading.device_id == device_id)
                .cloned()
                .collect()
        } else {
            energy_data.clone()
        }
    }
    
    pub async fn record_energy_reading(&self, reading: EnergyReading) {
        let mut energy_data = self.energy_data.lock().await;
        energy_data.push(reading);
        
        // Keep only last 1000 readings per device
        if energy_data.len() > 10000 {
            energy_data.drain(0..1000);
        }
    }
}

// REST API handlers
async fn get_devices(controller: web::Data<SmartHomeController>) -> Result<HttpResponse> {
    let devices = controller.get_devices().await;
    Ok(HttpResponse::Ok().json(devices))
}

async fn control_device(
    controller: web::Data<SmartHomeController>,
    path: web::Path<String>,
    command_data: web::Json<serde_json::Value>,
) -> Result<HttpResponse> {
    let device_id = path.into_inner();
    let command = command_data.get("command").and_then(|v| v.as_str()).unwrap_or("");
    let parameters = command_data.get("parameters")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<String, serde_json::Value>>()
        })
        .unwrap_or_default();
    
    match controller.control_device(&device_id, command, parameters).await {
        Ok(()) => Ok(HttpResponse::Ok().json(serde_json::json!({"status": "success"}))),
        Err(e) => Ok(HttpResponse::BadRequest().json(serde_json::json!({"error": e}))),
    }
}

async fn get_energy_data(
    controller: web::Data<SmartHomeController>,
    query: web::Query<HashMap<String, String>>,
) -> Result<HttpResponse> {
    let device_id = query.get("device_id").map(|s| s.as_str());
    let energy_data = controller.get_energy_consumption(device_id).await;
    Ok(HttpResponse::Ok().json(energy_data))
}

async fn initialize_demo_devices(controller: &SmartHomeController) {
    // Add demo devices
    let devices = vec![
        SmartDevice {
            id: "light_living_room".to_string(),
            name: "Living Room Light".to_string(),
            device_type: DeviceType::Light,
            location: "Living Room".to_string(),
            status: DeviceStatus::Online,
            properties: {
                let mut props = HashMap::new();
                props.insert("state".to_string(), serde_json::Value::String("off".to_string()));
                props.insert("brightness".to_string(), serde_json::Value::Number(serde_json::Number::from(100)));
                props
            },
            last_updated: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        },
        SmartDevice {
            id: "thermostat_main".to_string(),
            name: "Main Thermostat".to_string(),
            device_type: DeviceType::Thermostat,
            location: "Hallway".to_string(),
            status: DeviceStatus::Online,
            properties: {
                let mut props = HashMap::new();
                props.insert("current_temperature".to_string(), serde_json::Value::Number(serde_json::Number::from(22)));
                props.insert("target_temperature".to_string(), serde_json::Value::Number(serde_json::Number::from(21)));
                props.insert("mode".to_string(), serde_json::Value::String("auto".to_string()));
                props
            },
            last_updated: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        },
        SmartDevice {
            id: "door_lock_front".to_string(),
            name: "Front Door Lock".to_string(),
            device_type: DeviceType::DoorLock,
            location: "Front Door".to_string(),
            status: DeviceStatus::Online,
            properties: {
                let mut props = HashMap::new();
                props.insert("locked".to_string(), serde_json::Value::Bool(true));
                props.insert("battery_level".to_string(), serde_json::Value::Number(serde_json::Number::from(85)));
                props
            },
            last_updated: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        },
    ];
    
    for device in devices {
        controller.add_device(device).await;
    }
    
    // Add demo automation rule
    let automation_rule = AutomationRule {
        id: Uuid::new_v4().to_string(),
        name: "Evening Routine".to_string(),
        trigger: Trigger::TimeOfDay { hour: 18, minute: 0 },
        actions: vec![
            Action {
                device_id: "light_living_room".to_string(),
                command: "turn_on".to_string(),
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("brightness".to_string(), serde_json::Value::Number(serde_json::Number::from(80)));
                    params
                },
            },
        ],
        enabled: true,
        schedule: Some(Schedule {
            days_of_week: vec![1, 2, 3, 4, 5], // Monday to Friday
            start_date: None,
            end_date: None,
        }),
    };
    
    controller.add_automation_rule(automation_rule).await;
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    
    println!("🏠 Smart Home Controller Starting...");
    println!("====================================");
    
    let controller = web::Data::new(SmartHomeController::new());
    
    // Initialize demo devices
    initialize_demo_devices(&controller).await;
    
    println!("✅ Smart Home Controller initialized!");
    println!("🌐 Starting web server on http://localhost:8080");
    
    HttpServer::new(move || {
        App::new()
            .app_data(controller.clone())
            .wrap(Logger::default())
            .route("/api/devices", web::get().to(get_devices))
            .route("/api/devices/{device_id}/control", web::post().to(control_device))
            .route("/api/energy", web::get().to(get_energy_data))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "smart-home-controller"
version = "0.1.0"
edition = "2021"

[dependencies]
actix-web = "4.0"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4"] }
env_logger = "0.10"
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "sqlite"] }
rumqttc = "0.24"`,
          language: 'toml',
        }
      }
    },
    {
      id: 'rust-industrial-iot',
      name: 'Industrial IoT Monitor',
      description: 'Industrial-grade IoT monitoring system for manufacturing and process control',
      category: 'IoT',
      difficulty: 'Expert',
      tags: ['Industrial IoT', 'Manufacturing', 'Process Control', 'SCADA', 'Rust'],
      icon: <Factory className="w-6 h-6 text-gray-400" />,
      estimatedTime: '5-6 weeks',
      useCase: 'Monitor and control industrial equipment and processes',
      techStack: ['Rust', 'Modbus', 'OPC-UA', 'InfluxDB', 'Grafana'],
      features: [
        'Modbus/OPC-UA integration',
        'Real-time process monitoring',
        'Alarm management',
        'Historical data logging',
        'SCADA interface'
      ],
      files: {
        'main.rs': {
          content: `use tokio::time::{interval, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndustrialDevice {
    pub id: String,
    pub name: String,
    pub device_type: IndustrialDeviceType,
    pub location: String,
    pub protocol: CommunicationProtocol,
    pub address: String,
    pub status: DeviceStatus,
    pub parameters: HashMap<String, ProcessParameter>,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndustrialDeviceType {
    PLC,           // Programmable Logic Controller
    HMI,           // Human Machine Interface
    VFD,           // Variable Frequency Drive
    Sensor,        // Temperature, Pressure, Flow sensors
    Actuator,      // Valves, Motors
    Analyzer,      // Gas, Liquid analyzers
    PowerMeter,    // Energy monitoring
    SafetySystem,  // Emergency shutdown systems
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationProtocol {
    Modbus { slave_id: u8, baud_rate: u32 },
    OpcUa { endpoint: String },
    Ethernet { ip: String, port: u16 },
    Serial { port: String, baud_rate: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceStatus {
    Online,
    Offline,
    Maintenance,
    Alarm(String),
    Warning(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessParameter {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub alarm_high: Option<f64>,
    pub alarm_low: Option<f64>,
    pub warning_high: Option<f64>,
    pub warning_low: Option<f64>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alarm {
    pub id: String,
    pub device_id: String,
    pub parameter_name: String,
    pub alarm_type: AlarmType,
    pub severity: AlarmSeverity,
    pub message: String,
    pub value: f64,
    pub threshold: f64,
    pub timestamp: u64,
    pub acknowledged: bool,
    pub acknowledged_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlarmType {
    High,
    Low,
    DeviceOffline,
    CommunicationError,
    SafetyTrip,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlarmSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug)]
pub struct IndustrialIoTMonitor {
    devices: Arc<Mutex<HashMap<String, IndustrialDevice>>>,
    alarms: Arc<Mutex<Vec<Alarm>>>,
    historical_data: Arc<Mutex<Vec<ProcessDataPoint>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessDataPoint {
    pub device_id: String,
    pub parameter_name: String,
    pub value: f64,
    pub timestamp: u64,
}

impl IndustrialIoTMonitor {
    pub fn new() -> Self {
        Self {
            devices: Arc::new(Mutex::new(HashMap::new())),
            alarms: Arc::new(Mutex::new(Vec::new())),
            historical_data: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub async fn add_device(&self, device: IndustrialDevice) {
        let mut devices = self.devices.lock().await;
        devices.insert(device.id.clone(), device);
    }
    
    pub async fn read_device_data(&self, device_id: &str) -> Result<HashMap<String, f64>, String> {
        let devices = self.devices.lock().await;
        
        if let Some(device) = devices.get(device_id) {
            match &device.protocol {
                CommunicationProtocol::Modbus { slave_id, baud_rate } => {
                    self.read_modbus_data(device, *slave_id, *baud_rate).await
                }
                CommunicationProtocol::OpcUa { endpoint } => {
                    self.read_opcua_data(device, endpoint).await
                }
                CommunicationProtocol::Ethernet { ip, port } => {
                    self.read_ethernet_data(device, ip, *port).await
                }
                CommunicationProtocol::Serial { port, baud_rate } => {
                    self.read_serial_data(device, port, *baud_rate).await
                }
            }
        } else {
            Err(format!("Device not found: {}", device_id))
        }
    }
    
    async fn read_modbus_data(&self, device: &IndustrialDevice, slave_id: u8, baud_rate: u32) -> Result<HashMap<String, f64>, String> {
        // Simulate Modbus communication
        println!("📡 Reading Modbus data from {} (Slave ID: {}, Baud: {})", device.name, slave_id, baud_rate);
        
        let mut data = HashMap::new();
        
        match device.device_type {
            IndustrialDeviceType::PLC => {
                data.insert("input_voltage".to_string(), 230.0 + (rand::random::<f64>() - 0.5) * 10.0);
                data.insert("output_current".to_string(), 15.5 + (rand::random::<f64>() - 0.5) * 2.0);
                data.insert("cpu_usage".to_string(), 45.0 + rand::random::<f64>() * 20.0);
            }
            IndustrialDeviceType::VFD => {
                data.insert("motor_speed".to_string(), 1750.0 + (rand::random::<f64>() - 0.5) * 100.0);
                data.insert("motor_current".to_string(), 25.0 + (rand::random::<f64>() - 0.5) * 5.0);
                data.insert("frequency".to_string(), 50.0 + (rand::random::<f64>() - 0.5) * 1.0);
            }
            IndustrialDeviceType::Sensor => {
                data.insert("temperature".to_string(), 75.0 + (rand::random::<f64>() - 0.5) * 10.0);
                data.insert("pressure".to_string(), 2.5 + (rand::random::<f64>() - 0.5) * 0.5);
                data.insert("flow_rate".to_string(), 150.0 + (rand::random::<f64>() - 0.5) * 20.0);
            }
            _ => {
                data.insert("status".to_string(), if rand::random::<f64>() > 0.1 { 1.0 } else { 0.0 });
            }
        }
        
        Ok(data)
    }
    
    async fn read_opcua_data(&self, device: &IndustrialDevice, endpoint: &str) -> Result<HashMap<String, f64>, String> {
        // Simulate OPC-UA communication
        println!("🔗 Reading OPC-UA data from {} (Endpoint: {})", device.name, endpoint);
        
        let mut data = HashMap::new();
        data.insert("process_value".to_string(), 100.0 + rand::random::<f64>() * 50.0);
        data.insert("setpoint".to_string(), 125.0);
        data.insert("output".to_string(), 75.0 + rand::random::<f64>() * 25.0);
        
        Ok(data)
    }
    
    async fn read_ethernet_data(&self, device: &IndustrialDevice, ip: &str, port: u16) -> Result<HashMap<String, f64>, String> {
        // Simulate Ethernet communication
        println!("🌐 Reading Ethernet data from {} ({}:{})", device.name, ip, port);
        
        let mut data = HashMap::new();
        data.insert("network_status".to_string(), 1.0);
        data.insert("data_rate".to_string(), 1000.0 + rand::random::<f64>() * 100.0);
        
        Ok(data)
    }
    
    async fn read_serial_data(&self, device: &IndustrialDevice, port: &str, baud_rate: u32) -> Result<HashMap<String, f64>, String> {
        // Simulate Serial communication
        println!("📟 Reading Serial data from {} (Port: {}, Baud: {})", device.name, port, baud_rate);
        
        let mut data = HashMap::new();
        data.insert("serial_value".to_string(), rand::random::<f64>() * 100.0);
        
        Ok(data)
    }
    
    pub async fn update_device_parameters(&self, device_id: &str, data: HashMap<String, f64>) -> Result<(), String> {
        let mut devices = self.devices.lock().await;
        
        if let Some(device) = devices.get_mut(device_id) {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            for (param_name, value) in data {
                if let Some(parameter) = device.parameters.get_mut(&param_name) {
                    parameter.value = value;
                    parameter.timestamp = timestamp;
                    
                    // Check for alarms
                    self.check_parameter_alarms(device_id, &param_name, parameter).await;
                    
                    // Store historical data
                    let data_point = ProcessDataPoint {
                        device_id: device_id.to_string(),
                        parameter_name: param_name.clone(),
                        value,
                        timestamp,
                    };
                    
                    let mut historical_data = self.historical_data.lock().await;
                    historical_data.push(data_point);
                    
                    // Keep only last 10000 data points
                    if historical_data.len() > 10000 {
                        historical_data.drain(0..1000);
                    }
                }
            }
            
            device.last_updated = timestamp;
            Ok(())
        } else {
            Err(format!("Device not found: {}", device_id))
        }
    }
    
    async fn check_parameter_alarms(&self, device_id: &str, param_name: &str, parameter: &ProcessParameter) {
        let mut alarms_to_add = Vec::new();
        
        // Check high alarm
        if let Some(alarm_high) = parameter.alarm_high {
            if parameter.value > alarm_high {
                let alarm = Alarm {
                    id: uuid::Uuid::new_v4().to_string(),
                    device_id: device_id.to_string(),
                    parameter_name: param_name.to_string(),
                    alarm_type: AlarmType::High,
                    severity: AlarmSeverity::High,
                    message: format!("{} high alarm: {:.2} > {:.2}", param_name, parameter.value, alarm_high),
                    value: parameter.value,
                    threshold: alarm_high,
                    timestamp: parameter.timestamp,
                    acknowledged: false,
                    acknowledged_by: None,
                };
                alarms_to_add.push(alarm);
            }
        }
        
        // Check low alarm
        if let Some(alarm_low) = parameter.alarm_low {
            if parameter.value < alarm_low {
                let alarm = Alarm {
                    id: uuid::Uuid::new_v4().to_string(),
                    device_id: device_id.to_string(),
                    parameter_name: param_name.to_string(),
                    alarm_type: AlarmType::Low,
                    severity: AlarmSeverity::High,
                    message: format!("{} low alarm: {:.2} < {:.2}", param_name, parameter.value, alarm_low),
                    value: parameter.value,
                    threshold: alarm_low,
                    timestamp: parameter.timestamp,
                    acknowledged: false,
                    acknowledged_by: None,
                };
                alarms_to_add.push(alarm);
            }
        }
        
        if !alarms_to_add.is_empty() {
            let mut alarms = self.alarms.lock().await;
            for alarm in alarms_to_add {
                println!("🚨 ALARM: {}", alarm.message);
                alarms.push(alarm);
            }
        }
    }
    
    pub async fn get_active_alarms(&self) -> Vec<Alarm> {
        let alarms = self.alarms.lock().await;
        alarms.iter().filter(|alarm| !alarm.acknowledged).cloned().collect()
    }
    
    pub async fn acknowledge_alarm(&self, alarm_id: &str, user: &str) -> Result<(), String> {
        let mut alarms = self.alarms.lock().await;
        
        if let Some(alarm) = alarms.iter_mut().find(|a| a.id == alarm_id) {
            alarm.acknowledged = true;
            alarm.acknowledged_by = Some(user.to_string());
            println!("✅ Alarm {} acknowledged by {}", alarm_id, user);
            Ok(())
        } else {
            Err(format!("Alarm not found: {}", alarm_id))
        }
    }
    
    pub async fn start_monitoring(&self) {
        let mut interval = interval(Duration::from_secs(5));
        
        loop {
            interval.tick().await;
            
            let device_ids: Vec<String> = {
                let devices = self.devices.lock().await;
                devices.keys().cloned().collect()
            };
            
            for device_id in device_ids {
                match self.read_device_data(&device_id).await {
                    Ok(data) => {
                        if let Err(e) = self.update_device_parameters(&device_id, data).await {
                            println!("❌ Failed to update device {}: {}", device_id, e);
                        }
                    }
                    Err(e) => {
                        println!("❌ Failed to read device {}: {}", device_id, e);
                    }
                }
            }
            
            // Display active alarms
            let active_alarms = self.get_active_alarms().await;
            if !active_alarms.is_empty() {
                println!("🚨 Active alarms: {}", active_alarms.len());
            }
        }
    }
}

async fn initialize_demo_devices(monitor: &IndustrialIoTMonitor) {
    // Add demo industrial devices
    let devices = vec![
        IndustrialDevice {
            id: "plc_001".to_string(),
            name: "Main Production PLC".to_string(),
            device_type: IndustrialDeviceType::PLC,
            location: "Production Line 1".to_string(),
            protocol: CommunicationProtocol::Modbus { slave_id: 1, baud_rate: 9600 },
            address: "192.168.1.100".to_string(),
            status: DeviceStatus::Online,
            parameters: {
                let mut params = HashMap::new();
                params.insert("input_voltage".to_string(), ProcessParameter {
                    name: "Input Voltage".to_string(),
                    value: 230.0,
                    unit: "V".to_string(),
                    min_value: Some(200.0),
                    max_value: Some(250.0),
                    alarm_high: Some(245.0),
                    alarm_low: Some(210.0),
                    warning_high: Some(240.0),
                    warning_low: Some(215.0),
                    timestamp: 0,
                });
                params.insert("output_current".to_string(), ProcessParameter {
                    name: "Output Current".to_string(),
                    value: 15.5,
                    unit: "A".to_string(),
                    min_value: Some(0.0),
                    max_value: Some(25.0),
                    alarm_high: Some(22.0),
                    alarm_low: Some(2.0),
                    warning_high: Some(20.0),
                    warning_low: Some(5.0),
                    timestamp: 0,
                });
                params
            },
            last_updated: 0,
        },
        IndustrialDevice {
            id: "vfd_001".to_string(),
            name: "Conveyor Motor Drive".to_string(),
            device_type: IndustrialDeviceType::VFD,
            location: "Conveyor System".to_string(),
            protocol: CommunicationProtocol::Modbus { slave_id: 2, baud_rate: 9600 },
            address: "192.168.1.101".to_string(),
            status: DeviceStatus::Online,
            parameters: {
                let mut params = HashMap::new();
                params.insert("motor_speed".to_string(), ProcessParameter {
                    name: "Motor Speed".to_string(),
                    value: 1750.0,
                    unit: "RPM".to_string(),
                    min_value: Some(0.0),
                    max_value: Some(1800.0),
                    alarm_high: Some(1780.0),
                    alarm_low: Some(100.0),
                    warning_high: Some(1750.0),
                    warning_low: Some(200.0),
                    timestamp: 0,
                });
                params
            },
            last_updated: 0,
        },
    ];
    
    for device in devices {
        monitor.add_device(device).await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏭 Industrial IoT Monitor Starting...");
    println!("====================================");
    
    let monitor = IndustrialIoTMonitor::new();
    
    // Initialize demo devices
    initialize_demo_devices(&monitor).await;
    
    println!("✅ Industrial IoT Monitor initialized!");
    println!("📊 Starting process monitoring...");
    
    monitor.start_monitoring().await;
    
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "industrial-iot-monitor"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4"] }
tokio-modbus = "0.7"
opcua = "0.12"
influxdb = "0.7"
rand = "0.8"`,
          language: 'toml',
        }
      }
    },
    // DevOps Templates
    {
      id: 'rust-container-orchestrator',
      name: 'Container Orchestrator',
      description: 'Lightweight container orchestration system built with Rust',
      category: 'DevOps',
      difficulty: 'Expert',
      tags: ['Containers', 'Orchestration', 'Docker', 'Kubernetes', 'Rust'],
      icon: <Container className="w-6 h-6 text-blue-400" />,
      estimatedTime: '5-6 weeks',
      useCase: 'Manage and orchestrate containerized applications at scale',
      techStack: ['Rust', 'Docker', 'etcd', 'gRPC', 'Tokio'],
      features: [
        'Container lifecycle management',
        'Service discovery',
        'Load balancing',
        'Health monitoring',
        'Auto-scaling'
      ],
      files: {
        'main.rs': {
          content: `use tokio::time::{interval, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Container {
    pub id: String,
    pub name: String,
    pub image: String,
    pub status: ContainerStatus,
    pub node_id: String,
    pub ports: Vec<PortMapping>,
    pub environment: HashMap<String, String>,
    pub resources: ResourceRequirements,
    pub health_check: Option<HealthCheck>,
    pub created_at: u64,
    pub started_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerStatus {
    Pending,
    Running,
    Stopped,
    Failed,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    pub container_port: u16,
    pub host_port: u16,
    pub protocol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_limit: f64,      // CPU cores
    pub memory_limit: u64,   // Memory in MB
    pub cpu_request: f64,
    pub memory_request: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub endpoint: String,
    pub interval_seconds: u64,
    pub timeout_seconds: u64,
    pub retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub name: String,
    pub ip_address: String,
    pub status: NodeStatus,
    pub capacity: NodeCapacity,
    pub allocated: NodeCapacity,
    pub labels: HashMap<String, String>,
    pub last_heartbeat: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Ready,
    NotReady,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapacity {
    pub cpu: f64,
    pub memory: u64,
    pub pods: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Service {
    pub id: String,
    pub name: String,
    pub selector: HashMap<String, String>,
    pub ports: Vec<ServicePort>,
    pub service_type: ServiceType,
    pub endpoints: Vec<Endpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePort {
    pub name: String,
    pub port: u16,
    pub target_port: u16,
    pub protocol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceType {
    ClusterIP,
    NodePort,
    LoadBalancer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub ip: String,
    pub port: u16,
    pub ready: bool,
}

#[derive(Debug)]
pub struct ContainerOrchestrator {
    containers: Arc<Mutex<HashMap<String, Container>>>,
    nodes: Arc<Mutex<HashMap<String, Node>>>,
    services: Arc<Mutex<HashMap<String, Service>>>,
    scheduler: Arc<Mutex<Scheduler>>,
}

#[derive(Debug)]
pub struct Scheduler {
    pub strategy: SchedulingStrategy,
}

#[derive(Debug)]
pub enum SchedulingStrategy {
    RoundRobin,
    LeastLoaded,
    ResourceAware,
}

impl ContainerOrchestrator {
    pub fn new() -> Self {
        Self {
            containers: Arc::new(Mutex::new(HashMap::new())),
            nodes: Arc::new(Mutex::new(HashMap::new())),
            services: Arc::new(Mutex::new(HashMap::new())),
            scheduler: Arc::new(Mutex::new(Scheduler {
                strategy: SchedulingStrategy::ResourceAware,
            })),
        }
    }
    
    pub async fn register_node(&self, node: Node) {
        let mut nodes = self.nodes.lock().await;
        println!("🖥️ Registering node: {} ({})", node.name, node.ip_address);
        nodes.insert(node.id.clone(), node);
    }
    
    pub async fn schedule_container(&self, mut container: Container) -> Result<String, String> {
        let nodes = self.nodes.lock().await;
        let scheduler = self.scheduler.lock().await;
        
        let available_nodes: Vec<&Node> = nodes.values()
            .filter(|node| node.status == NodeStatus::Ready)
            .collect();
        
        if available_nodes.is_empty() {
            return Err("No available nodes for scheduling".to_string());
        }
        
        let selected_node = match scheduler.strategy {
            SchedulingStrategy::RoundRobin => {
                // Simple round-robin selection
                &available_nodes[0]
            }
            SchedulingStrategy::LeastLoaded => {
                // Select node with least allocated resources
                available_nodes.iter()
                    .min_by(|a, b| {
                        let a_load = (a.allocated.cpu / a.capacity.cpu) + 
                                   (a.allocated.memory as f64 / a.capacity.memory as f64);
                        let b_load = (b.allocated.cpu / b.capacity.cpu) + 
                                   (b.allocated.memory as f64 / b.capacity.memory as f64);
                        a_load.partial_cmp(&b_load).unwrap()
                    })
                    .unwrap()
            }
            SchedulingStrategy::ResourceAware => {
                // Select node that can accommodate the resource requirements
                available_nodes.iter()
                    .find(|node| {
                        let available_cpu = node.capacity.cpu - node.allocated.cpu;
                        let available_memory = node.capacity.memory - node.allocated.memory;
                        
                        available_cpu >= container.resources.cpu_request &&
                        available_memory >= container.resources.memory_request
                    })
                    .unwrap_or(&available_nodes[0])
            }
        };
        
        container.node_id = selected_node.id.clone();
        container.status = ContainerStatus::Pending;
        
        drop(nodes);
        drop(scheduler);
        
        // Start the container
        self.start_container(container.clone()).await?;
        
        let mut containers = self.containers.lock().await;
        containers.insert(container.id.clone(), container.clone());
        
        println!("📦 Container {} scheduled on node {}", container.name, selected_node.name);
        Ok(container.id)
    }
    
    async fn start_container(&self, mut container: Container) -> Result<(), String> {
        // Simulate container startup
        println!("🚀 Starting container: {} (image: {})", container.name, container.image);
        
        // Simulate Docker/containerd API call
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        container.status = ContainerStatus::Running;
        container.started_at = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        
        // Update node allocation
        let mut nodes = self.nodes.lock().await;
        if let Some(node) = nodes.get_mut(&container.node_id) {
            node.allocated.cpu += container.resources.cpu_request;
            node.allocated.memory += container.resources.memory_request;
        }
        
        println!("✅ Container {} started successfully", container.name);
        Ok(())
    }
    
    pub async fn stop_container(&self, container_id: &str) -> Result<(), String> {
        let mut containers = self.containers.lock().await;
        
        if let Some(container) = containers.get_mut(container_id) {
            println!("🛑 Stopping container: {}", container.name);
            
            // Simulate container stop
            tokio::time::sleep(Duration::from_millis(200)).await;
            
            container.status = ContainerStatus::Stopped;
            
            // Update node allocation
            let mut nodes = self.nodes.lock().await;
            if let Some(node) = nodes.get_mut(&container.node_id) {
                node.allocated.cpu -= container.resources.cpu_request;
                node.allocated.memory -= container.resources.memory_request;
            }
            
            println!("✅ Container {} stopped", container.name);
            Ok(())
        } else {
            Err(format!("Container not found: {}", container_id))
        }
    }
    
    pub async fn create_service(&self, service: Service) {
        let mut services = self.services.lock().await;
        println!("🌐 Creating service: {} (type: {:?})", service.name, service.service_type);
        services.insert(service.id.clone(), service);
    }
    
    pub async fn update_service_endpoints(&self, service_id: &str) -> Result<(), String> {
        let containers = self.containers.lock().await;
        let mut services = self.services.lock().await;
        
        if let Some(service) = services.get_mut(service_id) {
            let mut endpoints = Vec::new();
            
            // Find containers that match the service selector
            for container in containers.values() {
                if container.status == ContainerStatus::Running {
                    // Simple label matching (in real implementation, this would be more sophisticated)
                    let matches = service.selector.iter().all(|(key, value)| {
                        container.environment.get(key).map_or(false, |v| v == value)
                    });
                    
                    if matches {
                        for port_mapping in &container.ports {
                            endpoints.push(Endpoint {
                                ip: "127.0.0.1".to_string(), // In real implementation, get from node
                                port: port_mapping.host_port,
                                ready: true,
                            });
                        }
                    }
                }
            }
            
            service.endpoints = endpoints;
            println!("🔄 Updated service {} endpoints: {} endpoints", service.name, service.endpoints.len());
            Ok(())
        } else {
            Err(format!("Service not found: {}", service_id))
        }
    }
    
    pub async fn health_check_containers(&self) {
        let container_ids: Vec<String> = {
            let containers = self.containers.lock().await;
            containers.keys().cloned().collect()
        };
        
        for container_id in container_ids {
            if let Err(e) = self.check_container_health(&container_id).await {
                println!("❌ Health check failed for container {}: {}", container_id, e);
            }
        }
    }
    
    async fn check_container_health(&self, container_id: &str) -> Result<(), String> {
        let containers = self.containers.lock().await;
        
        if let Some(container) = containers.get(container_id) {
            if let Some(health_check) = &container.health_check {
                // Simulate health check
                let healthy = rand::random::<f64>() > 0.1; // 90% success rate
                
                if !healthy {
                    println!("🏥 Health check failed for container: {}", container.name);
                    return Err("Health check failed".to_string());
                }
            }
        }
        
        Ok(())
    }
    
    pub async fn get_cluster_status(&self) -> ClusterStatus {
        let containers = self.containers.lock().await;
        let nodes = self.nodes.lock().await;
        let services = self.services.lock().await;
        
        let running_containers = containers.values()
            .filter(|c| c.status == ContainerStatus::Running)
            .count();
        
        let ready_nodes = nodes.values()
            .filter(|n| n.status == NodeStatus::Ready)
            .count();
        
        ClusterStatus {
            total_nodes: nodes.len(),
            ready_nodes,
            total_containers: containers.len(),
            running_containers,
            total_services: services.len(),
        }
    }
    
    pub async fn start_control_loop(&self) {
        let mut interval = interval(Duration::from_secs(10));
        
        loop {
            interval.tick().await;
            
            // Perform health checks
            self.health_check_containers().await;
            
            // Update service endpoints
            let service_ids: Vec<String> = {
                let services = self.services.lock().await;
                services.keys().cloned().collect()
            };
            
            for service_id in service_ids {
                if let Err(e) = self.update_service_endpoints(&service_id).await {
                    println!("❌ Failed to update service endpoints: {}", e);
                }
            }
            
            // Display cluster status
            let status = self.get_cluster_status().await;
            println!("📊 Cluster Status: {} nodes ready, {}/{} containers running, {} services", 
                    status.ready_nodes, status.running_containers, status.total_containers, status.total_services);
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub total_nodes: usize,
    pub ready_nodes: usize,
    pub total_containers: usize,
    pub running_containers: usize,
    pub total_services: usize,
}

async fn initialize_demo_cluster(orchestrator: &ContainerOrchestrator) {
    // Register demo nodes
    let nodes = vec![
        Node {
            id: Uuid::new_v4().to_string(),
            name: "worker-node-1".to_string(),
            ip_address: "192.168.1.10".to_string(),
            status: NodeStatus::Ready,
            capacity: NodeCapacity {
                cpu: 4.0,
                memory: 8192,
                pods: 100,
            },
            allocated: NodeCapacity {
                cpu: 0.0,
                memory: 0,
                pods: 0,
            },
            labels: {
                let mut labels = HashMap::new();
                labels.insert("zone".to_string(), "us-west-1a".to_string());
                labels.insert("instance-type".to_string(), "m5.large".to_string());
                labels
            },
            last_heartbeat: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        },
        Node {
            id: Uuid::new_v4().to_string(),
            name: "worker-node-2".to_string(),
            ip_address: "192.168.1.11".to_string(),
            status: NodeStatus::Ready,
            capacity: NodeCapacity {
                cpu: 8.0,
                memory: 16384,
                pods: 200,
            },
            allocated: NodeCapacity {
                cpu: 0.0,
                memory: 0,
                pods: 0,
            },
            labels: {
                let mut labels = HashMap::new();
                labels.insert("zone".to_string(), "us-west-1b".to_string());
                labels.insert("instance-type".to_string(), "m5.xlarge".to_string());
                labels
            },
            last_heartbeat: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        },
    ];
    
    for node in nodes {
        orchestrator.register_node(node).await;
    }
    
    // Schedule demo containers
    let containers = vec![
        Container {
            id: Uuid::new_v4().to_string(),
            name: "web-server-1".to_string(),
            image: "nginx:latest".to_string(),
            status: ContainerStatus::Pending,
            node_id: String::new(),
            ports: vec![
                PortMapping {
                    container_port: 80,
                    host_port: 8080,
                    protocol: "TCP".to_string(),
                }
            ],
            environment: {
                let mut env = HashMap::new();
                env.insert("app".to_string(), "web".to_string());
                env.insert("tier".to_string(), "frontend".to_string());
                env
            },
            resources: ResourceRequirements {
                cpu_limit: 1.0,
                memory_limit: 512,
                cpu_request: 0.5,
                memory_request: 256,
            },
            health_check: Some(HealthCheck {
                endpoint: "/health".to_string(),
                interval_seconds: 30,
                timeout_seconds: 5,
                retries: 3,
            }),
            created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            started_at: None,
        },
        Container {
            id: Uuid::new_v4().to_string(),
            name: "api-server-1".to_string(),
            image: "myapp/api:v1.0".to_string(),
            status: ContainerStatus::Pending,
            node_id: String::new(),
            ports: vec![
                PortMapping {
                    container_port: 3000,
                    host_port: 3000,
                    protocol: "TCP".to_string(),
                }
            ],
            environment: {
                let mut env = HashMap::new();
                env.insert("app".to_string(), "api".to_string());
                env.insert("tier".to_string(), "backend".to_string());
                env
            },
            resources: ResourceRequirements {
                cpu_limit: 2.0,
                memory_limit: 1024,
                cpu_request: 1.0,
                memory_request: 512,
            },
            health_check: Some(HealthCheck {
                endpoint: "/api/health".to_string(),
                interval_seconds: 30,
                timeout_seconds: 5,
                retries: 3,
            }),
            created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            started_at: None,
        },
    ];
    
    for container in containers {
        if let Err(e) = orchestrator.schedule_container(container).await {
            println!("❌ Failed to schedule container: {}", e);
        }
    }
    
    // Create demo service
    let service = Service {
        id: Uuid::new_v4().to_string(),
        name: "web-service".to_string(),
        selector: {
            let mut selector = HashMap::new();
            selector.insert("app".to_string(), "web".to_string());
            selector
        },
        ports: vec![
            ServicePort {
                name: "http".to_string(),
                port: 80,
                target_port: 80,
                protocol: "TCP".to_string(),
            }
        ],
        service_type: ServiceType::LoadBalancer,
        endpoints: Vec::new(),
    };
    
    orchestrator.create_service(service).await;
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🐳 Container Orchestrator Starting...");
    println!("=====================================");
    
    let orchestrator = ContainerOrchestrator::new();
    
    // Initialize demo cluster
    initialize_demo_cluster(&orchestrator).await;
    
    println!("✅ Container orchestrator initialized!");
    println!("🔄 Starting control loop...");
    
    orchestrator.start_control_loop().await;
    
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "container-orchestrator"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4"] }
tonic = "0.10"
prost = "0.12"
etcd-rs = "1.0"
rand = "0.8"`,
          language: 'toml',
        }
      }
    },
    {
      id: 'rust-ci-cd-pipeline',
      name: 'CI/CD Pipeline Engine',
      description: 'High-performance CI/CD pipeline engine with parallel execution and artifact management',
      category: 'DevOps',
      difficulty: 'Advanced',
      tags: ['CI/CD', 'Pipeline', 'Automation', 'Build', 'Rust'],
      icon: <GitBranch className="w-6 h-6 text-green-400" />,
      estimatedTime: '4-5 weeks',
      useCase: 'Automate build, test, and deployment processes with parallel execution',
      techStack: ['Rust', 'Docker', 'Git', 'YAML', 'Tokio'],
      features: [
        'Pipeline as code',
        'Parallel job execution',
        'Artifact management',
        'Multi-environment deployment',
        'Real-time monitoring'
      ],
      files: {
        'main.rs': {
          content: `use tokio::time::{interval, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    pub id: String,
    pub name: String,
    pub repository: Repository,
    pub stages: Vec<Stage>,
    pub environment_variables: HashMap<String, String>,
    pub triggers: Vec<Trigger>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Repository {
    pub url: String,
    pub branch: String,
    pub credentials: Option<Credentials>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    pub username: String,
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    pub name: String,
    pub jobs: Vec<Job>,
    pub depends_on: Vec<String>,
    pub condition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub name: String,
    pub image: String,
    pub commands: Vec<String>,
    pub environment: HashMap<String, String>,
    pub artifacts: Vec<Artifact>,
    pub timeout_minutes: u32,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    pub path: String,
    pub artifact_type: ArtifactType,
    pub retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    Binary,
    TestResults,
    Coverage,
    Documentation,
    Image,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trigger {
    Push { branches: Vec<String> },
    PullRequest { target_branches: Vec<String> },
    Schedule { cron: String },
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRun {
    pub id: String,
    pub pipeline_id: String,
    pub status: RunStatus,
    pub trigger_event: TriggerEvent,
    pub stage_runs: Vec<StageRun>,
    pub started_at: u64,
    pub finished_at: Option<u64>,
    pub duration_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunStatus {
    Pending,
    Running,
    Success,
    Failed,
    Cancelled,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerEvent {
    pub event_type: String,
    pub commit_sha: String,
    pub branch: String,
    pub author: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRun {
    pub stage_name: String,
    pub status: RunStatus,
    pub job_runs: Vec<JobRun>,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRun {
    pub job_name: String,
    pub status: RunStatus,
    pub logs: Vec<String>,
    pub artifacts: Vec<StoredArtifact>,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredArtifact {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub checksum: String,
    pub uploaded_at: u64,
}

#[derive(Debug)]
pub struct CICDEngine {
    pipelines: Arc<Mutex<HashMap<String, Pipeline>>>,
    pipeline_runs: Arc<Mutex<HashMap<String, PipelineRun>>>,
    artifact_store: Arc<Mutex<HashMap<String, StoredArtifact>>>,
    executor: Arc<Mutex<JobExecutor>>,
}

#[derive(Debug)]
pub struct JobExecutor {
    pub max_concurrent_jobs: usize,
    pub running_jobs: HashMap<String, tokio::task::JoinHandle<()>>,
}

impl CICDEngine {
    pub fn new() -> Self {
        Self {
            pipelines: Arc::new(Mutex::new(HashMap::new())),
            pipeline_runs: Arc::new(Mutex::new(HashMap::new())),
            artifact_store: Arc::new(Mutex::new(HashMap::new())),
            executor: Arc::new(Mutex::new(JobExecutor {
                max_concurrent_jobs: 10,
                running_jobs: HashMap::new(),
            })),
        }
    }
    
    pub async fn create_pipeline(&self, pipeline: Pipeline) {
        let mut pipelines = self.pipelines.lock().await;
        println!("📋 Creating pipeline: {}", pipeline.name);
        pipelines.insert(pipeline.id.clone(), pipeline);
    }
    
    pub async fn trigger_pipeline(&self, pipeline_id: &str, event: TriggerEvent) -> Result<String, String> {
        let pipelines = self.pipelines.lock().await;
        
        if let Some(pipeline) = pipelines.get(pipeline_id) {
            let run_id = Uuid::new_v4().to_string();
            
            let pipeline_run = PipelineRun {
                id: run_id.clone(),
                pipeline_id: pipeline_id.to_string(),
                status: RunStatus::Pending,
                trigger_event: event,
                stage_runs: Vec::new(),
                started_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                finished_at: None,
                duration_seconds: None,
            };
            
            drop(pipelines);
            
            let mut pipeline_runs = self.pipeline_runs.lock().await;
            pipeline_runs.insert(run_id.clone(), pipeline_run);
            
            println!("🚀 Pipeline run {} triggered for pipeline {}", run_id, pipeline_id);
            
            // Start pipeline execution
            let engine_clone = Arc::new(self);
            let run_id_clone = run_id.clone();
            let pipeline_id_clone = pipeline_id.to_string();
            
            tokio::spawn(async move {
                if let Err(e) = engine_clone.execute_pipeline_run(&run_id_clone, &pipeline_id_clone).await {
                    println!("❌ Pipeline run {} failed: {}", run_id_clone, e);
                }
            });
            
            Ok(run_id)
        } else {
            Err(format!("Pipeline not found: {}", pipeline_id))
        }
    }
    
    async fn execute_pipeline_run(&self, run_id: &str, pipeline_id: &str) -> Result<(), String> {
        let pipeline = {
            let pipelines = self.pipelines.lock().await;
            pipelines.get(pipeline_id).cloned()
                .ok_or_else(|| format!("Pipeline not found: {}", pipeline_id))?
        };
        
        // Update run status to running
        {
            let mut pipeline_runs = self.pipeline_runs.lock().await;
            if let Some(run) = pipeline_runs.get_mut(run_id) {
                run.status = RunStatus::Running;
            }
        }
        
        println!("▶️ Executing pipeline run: {}", run_id);
        
        // Execute stages in order
        for stage in &pipeline.stages {
            if let Err(e) = self.execute_stage(run_id, stage).await {
                println!("❌ Stage {} failed: {}", stage.name, e);
                
                // Mark pipeline run as failed
                let mut pipeline_runs = self.pipeline_runs.lock().await;
                if let Some(run) = pipeline_runs.get_mut(run_id) {
                    run.status = RunStatus::Failed;
                    run.finished_at = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    );
                }
                
                return Err(e);
            }
        }
        
        // Mark pipeline run as successful
        let mut pipeline_runs = self.pipeline_runs.lock().await;
        if let Some(run) = pipeline_runs.get_mut(run_id) {
            run.status = RunStatus::Success;
            let finished_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            run.finished_at = Some(finished_at);
            run.duration_seconds = Some(finished_at - run.started_at);
        }
        
        println!("✅ Pipeline run {} completed successfully", run_id);
        Ok(())
    }
    
    async fn execute_stage(&self, run_id: &str, stage: &Stage) -> Result<(), String> {
        println!("🎭 Executing stage: {}", stage.name);
        
        let stage_run = StageRun {
            stage_name: stage.name.clone(),
            status: RunStatus::Running,
            job_runs: Vec::new(),
            started_at: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            finished_at: None,
        };
        
        // Add stage run to pipeline run
        {
            let mut pipeline_runs = self.pipeline_runs.lock().await;
            if let Some(run) = pipeline_runs.get_mut(run_id) {
                run.stage_runs.push(stage_run);
            }
        }
        
        // Execute jobs in parallel
        let mut job_handles = Vec::new();
        
        for job in &stage.jobs {
            let job_clone = job.clone();
            let run_id_clone = run_id.to_string();
            let stage_name_clone = stage.name.clone();
            let engine_clone = Arc::new(self);
            
            let handle = tokio::spawn(async move {
                engine_clone.execute_job(&run_id_clone, &stage_name_clone, &job_clone).await
            });
            
            job_handles.push(handle);
        }
        
        // Wait for all jobs to complete
        let mut all_successful = true;
        for handle in job_handles {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    println!("❌ Job failed: {}", e);
                    all_successful = false;
                }
                Err(e) => {
                    println!("❌ Job execution error: {}", e);
                    all_successful = false;
                }
            }
        }
        
        // Update stage status
        {
            let mut pipeline_runs = self.pipeline_runs.lock().await;
            if let Some(run) = pipeline_runs.get_mut(run_id) {
                if let Some(stage_run) = run.stage_runs.iter_mut().find(|s| s.stage_name == stage.name) {
                    stage_run.status = if all_successful { RunStatus::Success } else { RunStatus::Failed };
                    stage_run.finished_at = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    );
                }
            }
        }
        
        if all_successful {
            println!("✅ Stage {} completed successfully", stage.name);
            Ok(())
        } else {
            Err(format!("Stage {} failed", stage.name))
        }
    }
    
    async fn execute_job(&self, run_id: &str, stage_name: &str, job: &Job) -> Result<(), String> {
        println!("🔧 Executing job: {} in stage {}", job.name, stage_name);
        
        let job_run = JobRun {
            job_name: job.name.clone(),
            status: RunStatus::Running,
            logs: Vec::new(),
            artifacts: Vec::new(),
            started_at: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            finished_at: None,
            exit_code: None,
        };
        
        // Add job run to stage run
        {
            let mut pipeline_runs = self.pipeline_runs.lock().await;
            if let Some(run) = pipeline_runs.get_mut(run_id) {
                if let Some(stage_run) = run.stage_runs.iter_mut().find(|s| s.stage_name == stage_name) {
                    stage_run.job_runs.push(job_run);
                }
            }
        }
        
        // Simulate job execution
        for (i, command) in job.commands.iter().enumerate() {
            println!("  📝 Executing command {}: {}", i + 1, command);
            
            // Add log entry
            {
                let mut pipeline_runs = self.pipeline_runs.lock().await;
                if let Some(run) = pipeline_runs.get_mut(run_id) {
                    if let Some(stage_run) = run.stage_runs.iter_mut().find(|s| s.stage_name == stage_name) {
                        if let Some(job_run) = stage_run.job_runs.iter_mut().find(|j| j.job_name == job.name) {
                            job_run.logs.push(format!("Executing: {}", command));
                        }
                    }
                }
            }
            
            // Simulate command execution time
            tokio::time::sleep(Duration::from_millis(500)).await;
            
            // Simulate occasional failures
            if rand::random::<f64>() < 0.05 {
                return Err(format!("Command failed: {}", command));
            }
        }
        
        // Process artifacts
        for artifact in &job.artifacts {
            if let Ok(stored_artifact) = self.store_artifact(artifact).await {
                let mut pipeline_runs = self.pipeline_runs.lock().await;
                if let Some(run) = pipeline_runs.get_mut(run_id) {
                    if let Some(stage_run) = run.stage_runs.iter_mut().find(|s| s.stage_name == stage_name) {
                        if let Some(job_run) = stage_run.job_runs.iter_mut().find(|j| j.job_name == job.name) {
                            job_run.artifacts.push(stored_artifact);
                        }
                    }
                }
            }
        }
        
        // Update job status
        {
            let mut pipeline_runs = self.pipeline_runs.lock().await;
            if let Some(run) = pipeline_runs.get_mut(run_id) {
                if let Some(stage_run) = run.stage_runs.iter_mut().find(|s| s.stage_name == stage_name) {
                    if let Some(job_run) = stage_run.job_runs.iter_mut().find(|j| j.job_name == job.name) {
                        job_run.status = RunStatus::Success;
                        job_run.finished_at = Some(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs()
                        );
                        job_run.exit_code = Some(0);
                    }
                }
            }
        }
        
        println!("✅ Job {} completed successfully", job.name);
        Ok(())
    }
    
    async fn store_artifact(&self, artifact: &Artifact) -> Result<StoredArtifact, String> {
        // Simulate artifact storage
        let stored_artifact = StoredArtifact {
            name: artifact.name.clone(),
            path: format!("/artifacts/{}", artifact.name),
            size_bytes: rand::random::<u64>() % 1000000 + 1000, // Random size between 1KB and 1MB
            checksum: format!("sha256:{}", Uuid::new_v4()),
            uploaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        let mut artifact_store = self.artifact_store.lock().await;
        artifact_store.insert(stored_artifact.name.clone(), stored_artifact.clone());
        
        println!("📦 Artifact stored: {} ({} bytes)", stored_artifact.name, stored_artifact.size_bytes);
        Ok(stored_artifact)
    }
    
    pub async fn get_pipeline_run_status(&self, run_id: &str) -> Option<RunStatus> {
        let pipeline_runs = self.pipeline_runs.lock().await;
        pipeline_runs.get(run_id).map(|run| run.status.clone())
    }
    
    pub async fn get_pipeline_runs(&self, pipeline_id: &str) -> Vec<PipelineRun> {
        let pipeline_runs = self.pipeline_runs.lock().await;
        pipeline_runs.values()
            .filter(|run| run.pipeline_id == pipeline_id)
            .cloned()
            .collect()
    }
    
    pub async fn start_monitoring(&self) {
        let mut interval = interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            let pipeline_runs = self.pipeline_runs.lock().await;
            let running_runs: Vec<&PipelineRun> = pipeline_runs.values()
                .filter(|run| matches!(run.status, RunStatus::Running))
                .collect();
            
            if !running_runs.is_empty() {
                println!("🔄 Active pipeline runs: {}", running_runs.len());
                for run in running_runs {
                    let duration = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs() - run.started_at;
                    println!("  - {} (running for {}s)", run.id, duration);
                }
            }
        }
    }
}

async fn create_demo_pipeline() -> Pipeline {
    Pipeline {
        id: Uuid::new_v4().to_string(),
        name: "Web Application CI/CD".to_string(),
        repository: Repository {
            url: "https://github.com/example/webapp.git".to_string(),
            branch: "main".to_string(),
            credentials: None,
        },
        stages: vec![
            Stage {
                name: "Build".to_string(),
                jobs: vec![
                    Job {
                        name: "compile".to_string(),
                        image: "rust:1.70".to_string(),
                        commands: vec![
                            "cargo build --release".to_string(),
                            "cargo test".to_string(),
                        ],
                        environment: HashMap::new(),
                        artifacts: vec![
                            Artifact {
                                name: "binary".to_string(),
                                path: "target/release/webapp".to_string(),
                                artifact_type: ArtifactType::Binary,
                                retention_days: 30,
                            }
                        ],
                        timeout_minutes: 10,
                        retry_count: 2,
                    }
                ],
                depends_on: Vec::new(),
                condition: None,
            },
            Stage {
                name: "Test".to_string(),
                jobs: vec![
                    Job {
                        name: "unit-tests".to_string(),
                        image: "rust:1.70".to_string(),
                        commands: vec![
                            "cargo test --verbose".to_string(),
                            "cargo tarpaulin --out xml".to_string(),
                        ],
                        environment: HashMap::new(),
                        artifacts: vec![
                            Artifact {
                                name: "test-results".to_string(),
                                path: "test-results.xml".to_string(),
                                artifact_type: ArtifactType::TestResults,
                                retention_days: 7,
                            },
                            Artifact {
                                name: "coverage".to_string(),
                                path: "cobertura.xml".to_string(),
                                artifact_type: ArtifactType::Coverage,
                                retention_days: 7,
                            }
                        ],
                        timeout_minutes: 15,
                        retry_count: 1,
                    },
                    Job {
                        name: "integration-tests".to_string(),
                        image: "rust:1.70".to_string(),
                        commands: vec![
                            "cargo test --test integration".to_string(),
                        ],
                        environment: HashMap::new(),
                        artifacts: Vec::new(),
                        timeout_minutes: 20,
                        retry_count: 1,
                    }
                ],
                depends_on: vec!["Build".to_string()],
                condition: None,
            },
            Stage {
                name: "Deploy".to_string(),
                jobs: vec![
                    Job {
                        name: "deploy-staging".to_string(),
                        image: "alpine/k8s:latest".to_string(),
                        commands: vec![
                            "kubectl apply -f k8s/staging/".to_string(),
                            "kubectl rollout status deployment/webapp-staging".to_string(),
                        ],
                        environment: {
                            let mut env = HashMap::new();
                            env.insert("KUBECONFIG".to_string(), "/etc/kubeconfig".to_string());
                            env
                        },
                        artifacts: Vec::new(),
                        timeout_minutes: 10,
                        retry_count: 2,
                    }
                ],
                depends_on: vec!["Test".to_string()],
                condition: Some("branch == 'main'".to_string()),
            }
        ],
        environment_variables: HashMap::new(),
        triggers: vec![
            Trigger::Push { branches: vec!["main".to_string(), "develop".to_string()] },
            Trigger::PullRequest { target_branches: vec!["main".to_string()] },
        ],
        created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        updated_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 CI/CD Pipeline Engine Starting...");
    println!("====================================");
    
    let engine = CICDEngine::new();
    
    // Create demo pipeline
    let demo_pipeline = create_demo_pipeline().await;
    let pipeline_id = demo_pipeline.id.clone();
    engine.create_pipeline(demo_pipeline).await;
    
    // Start monitoring
    let engine_clone = Arc::new(&engine);
    tokio::spawn(async move {
        engine_clone.start_monitoring().await;
    });
    
    // Trigger demo pipeline run
    let trigger_event = TriggerEvent {
        event_type: "push".to_string(),
        commit_sha: "abc123def456".to_string(),
        branch: "main".to_string(),
        author: "developer@example.com".to_string(),
        message: "Add new feature".to_string(),
    };
    
    match engine.trigger_pipeline(&pipeline_id, trigger_event).await {
        Ok(run_id) => {
            println!("✅ Pipeline run triggered: {}", run_id);
            
            // Wait for completion
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                
                if let Some(status) = engine.get_pipeline_run_status(&run_id).await {
                    match status {
                        RunStatus::Success => {
                            println!("🎉 Pipeline run completed successfully!");
                            break;
                        }
                        RunStatus::Failed => {
                            println!("❌ Pipeline run failed!");
                            break;
                        }
                        RunStatus::Running => {
                            println!("⏳ Pipeline run still in progress...");
                        }
                        _ => {}
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to trigger pipeline: {}", e);
        }
    }
    
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "cicd-pipeline-engine"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
uuid = { version = "1.0", features = ["v4"] }
git2 = "0.18"
docker-api = "0.14"
rand = "0.8"`,
          language: 'toml',
        }
      }
    },
    {
      id: 'rust-monitoring-system',
      name: 'Infrastructure Monitoring',
      description: 'Comprehensive infrastructure monitoring system with metrics collection and alerting',
      category: 'DevOps',
      difficulty: 'Advanced',
      tags: ['Monitoring', 'Metrics', 'Alerting', 'Observability', 'Rust'],
      icon: <Activity className="w-6 h-6 text-red-400" />,
      estimatedTime: '4-5 weeks',
      useCase: 'Monitor infrastructure health, collect metrics, and send alerts',
      techStack: ['Rust', 'Prometheus', 'Grafana', 'InfluxDB', 'SMTP'],
      features: [
        'Multi-source metrics collection',
        'Real-time alerting',
        'Dashboard integration',
        'Historical data storage',
        'Custom metric definitions'
      ],
      files: {
        'main.rs': {
          content: `use tokio::time::{interval, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    pub name: String,
    pub metric_type: MetricType,
    pub description: String,
    pub labels: Vec<String>,
    pub unit: String,
    pub collection_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub metric_name: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub metric_name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub duration: Duration,
    pub severity: AlertSeverity,
    pub notification_channels: Vec<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub rule_id: String,
    pub metric_name: String,
    pub current_value: f64,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub message: String,
    pub started_at: u64,
    pub resolved_at: Option<u64>,
    pub status: AlertStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertStatus {
    Firing,
    Resolved,
    Silenced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringTarget {
    pub id: String,
    pub name: String,
    pub target_type: TargetType,
    pub endpoint: String,
    pub credentials: Option<Credentials>,
    pub metrics: Vec<String>,
    pub collection_interval: Duration,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetType {
    PrometheusExporter,
    HttpEndpoint,
    DatabaseConnection,
    SystemMetrics,
    ApplicationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    pub username: String,
    pub password: String,
}

#[derive(Debug)]
pub struct MonitoringSystem {
    metrics: Arc<Mutex<HashMap<String, Vec<MetricValue>>>>,
    alert_rules: Arc<Mutex<HashMap<String, AlertRule>>>,
    active_alerts: Arc<Mutex<HashMap<String, Alert>>>,
    targets: Arc<Mutex<HashMap<String, MonitoringTarget>>>,
    metric_definitions: Arc<Mutex<HashMap<String, MetricDefinition>>>,
}

impl MonitoringSystem {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
            alert_rules: Arc::new(Mutex::new(HashMap::new())),
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
            targets: Arc::new(Mutex::new(HashMap::new())),
            metric_definitions: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub async fn register_metric(&self, definition: MetricDefinition) {
        let mut metric_definitions = self.metric_definitions.lock().await;
        println!("📊 Registering metric: {} ({})", definition.name, definition.description);
        metric_definitions.insert(definition.name.clone(), definition);
    }
    
    pub async fn add_monitoring_target(&self, target: MonitoringTarget) {
        let mut targets = self.targets.lock().await;
        println!("🎯 Adding monitoring target: {} ({})", target.name, target.endpoint);
        targets.insert(target.id.clone(), target);
    }
    
    pub async fn collect_metrics(&self) {
        let targets = self.targets.lock().await.clone();
        
        for target in targets.values() {
            if !target.enabled {
                continue;
            }
            
            match self.collect_target_metrics(target).await {
                Ok(metrics) => {
                    for metric in metrics {
                        self.store_metric(metric).await;
                    }
                }
                Err(e) => {
                    println!("❌ Failed to collect metrics from {}: {}", target.name, e);
                }
            }
        }
    }
    
    async fn collect_target_metrics(&self, target: &MonitoringTarget) -> Result<Vec<MetricValue>, String> {
        let mut metrics = Vec::new();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        match target.target_type {
            TargetType::SystemMetrics => {
                // Collect system metrics
                metrics.push(MetricValue {
                    metric_name: "system_cpu_usage".to_string(),
                    value: self.get_cpu_usage().await,
                    labels: {
                        let mut labels = HashMap::new();
                        labels.insert("host".to_string(), target.name.clone());
                        labels
                    },
                    timestamp,
                });
                
                metrics.push(MetricValue {
                    metric_name: "system_memory_usage".to_string(),
                    value: self.get_memory_usage().await,
                    labels: {
                        let mut labels = HashMap::new();
                        labels.insert("host".to_string(), target.name.clone());
                        labels
                    },
                    timestamp,
                });
                
                metrics.push(MetricValue {
                    metric_name: "system_disk_usage".to_string(),
                    value: self.get_disk_usage().await,
                    labels: {
                        let mut labels = HashMap::new();
                        labels.insert("host".to_string(), target.name.clone());
                        labels.insert("mount".to_string(), "/".to_string());
                        labels
                    },
                    timestamp,
                });
            }
            TargetType::HttpEndpoint => {
                // Collect HTTP endpoint metrics
                let response_time = self.measure_http_response_time(&target.endpoint).await?;
                metrics.push(MetricValue {
                    metric_name: "http_response_time".to_string(),
                    value: response_time,
                    labels: {
                        let mut labels = HashMap::new();
                        labels.insert("endpoint".to_string(), target.endpoint.clone());
                        labels
                    },
                    timestamp,
                });
                
                let status_code = self.get_http_status_code(&target.endpoint).await?;
                metrics.push(MetricValue {
                    metric_name: "http_status_code".to_string(),
                    value: status_code as f64,
                    labels: {
                        let mut labels = HashMap::new();
                        labels.insert("endpoint".to_string(), target.endpoint.clone());
                        labels
                    },
                    timestamp,
                });
            }
            TargetType::ApplicationMetrics => {
                // Collect application-specific metrics
                metrics.push(MetricValue {
                    metric_name: "app_requests_total".to_string(),
                    value: rand::random::<f64>() * 1000.0,
                    labels: {
                        let mut labels = HashMap::new();
                        labels.insert("app".to_string(), target.name.clone());
                        labels.insert("method".to_string(), "GET".to_string());
                        labels
                    },
                    timestamp,
                });
                
                metrics.push(MetricValue {
                    metric_name: "app_error_rate".to_string(),
                    value: rand::random::<f64>() * 0.05, // 0-5% error rate
                    labels: {
                        let mut labels = HashMap::new();
                        labels.insert("app".to_string(), target.name.clone());
                        labels
                    },
                    timestamp,
                });
            }
            _ => {
                return Err("Unsupported target type".to_string());
            }
        }
        
        Ok(metrics)
    }
    
    async fn get_cpu_usage(&self) -> f64 {
        // Simulate CPU usage (0-100%)
        rand::random::<f64>() * 100.0
    }
    
    async fn get_memory_usage(&self) -> f64 {
        // Simulate memory usage (0-100%)
        50.0 + rand::random::<f64>() * 40.0
    }
    
    async fn get_disk_usage(&self) -> f64 {
        // Simulate disk usage (0-100%)
        30.0 + rand::random::<f64>() * 50.0
    }
    
    async fn measure_http_response_time(&self, endpoint: &str) -> Result<f64, String> {
        // Simulate HTTP response time measurement
        println!("🌐 Measuring response time for {}", endpoint);
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(50.0 + rand::random::<f64>() * 200.0) // 50-250ms
    }
    
    async fn get_http_status_code(&self, endpoint: &str) -> Result<u16, String> {
        // Simulate HTTP status code check
        let status_codes = [200, 200, 200, 200, 404, 500]; // Mostly 200s
        let index = (rand::random::<f64>() * status_codes.len() as f64) as usize;
        Ok(status_codes[index % status_codes.len()])
    }
    
    async fn store_metric(&self, metric: MetricValue) {
        let mut metrics = self.metrics.lock().await;
        
        let metric_history = metrics.entry(metric.metric_name.clone()).or_insert_with(Vec::new);
        metric_history.push(metric.clone());
        
        // Keep only last 1000 data points per metric
        if metric_history.len() > 1000 {
            metric_history.drain(0..100);
        }
        
        // Check alert rules
        drop(metrics);
        self.evaluate_alert_rules(&metric).await;
    }
    
    pub async fn add_alert_rule(&self, rule: AlertRule) {
        let mut alert_rules = self.alert_rules.lock().await;
        println!("🚨 Adding alert rule: {} for metric {}", rule.name, rule.metric_name);
        alert_rules.insert(rule.id.clone(), rule);
    }
    
    async fn evaluate_alert_rules(&self, metric: &MetricValue) {
        let alert_rules = self.alert_rules.lock().await;
        
        for rule in alert_rules.values() {
            if rule.metric_name == metric.metric_name && rule.enabled {
                let should_alert = match rule.condition {
                    AlertCondition::GreaterThan => metric.value > rule.threshold,
                    AlertCondition::LessThan => metric.value < rule.threshold,
                    AlertCondition::Equal => (metric.value - rule.threshold).abs() < f64::EPSILON,
                    AlertCondition::NotEqual => (metric.value - rule.threshold).abs() > f64::EPSILON,
                    AlertCondition::GreaterThanOrEqual => metric.value >= rule.threshold,
                    AlertCondition::LessThanOrEqual => metric.value <= rule.threshold,
                };
                
                if should_alert {
                    self.trigger_alert(rule, metric).await;
                } else {
                    self.resolve_alert(rule).await;
                }
            }
        }
    }
    
    async fn trigger_alert(&self, rule: &AlertRule, metric: &MetricValue) {
        let mut active_alerts = self.active_alerts.lock().await;
        
        // Check if alert is already active
        if active_alerts.contains_key(&rule.id) {
            return;
        }
        
        let alert = Alert {
            id: uuid::Uuid::new_v4().to_string(),
            rule_id: rule.id.clone(),
            metric_name: rule.metric_name.clone(),
            current_value: metric.value,
            threshold: rule.threshold,
            severity: rule.severity.clone(),
            message: format!(
                "Alert: {} - {} is {:.2} (threshold: {:.2})",
                rule.name, rule.metric_name, metric.value, rule.threshold
            ),
            started_at: metric.timestamp,
            resolved_at: None,
            status: AlertStatus::Firing,
        };
        
        println!("🚨 ALERT TRIGGERED: {}", alert.message);
        
        // Send notifications
        for channel in &rule.notification_channels {
            self.send_notification(channel, &alert).await;
        }
        
        active_alerts.insert(rule.id.clone(), alert);
    }
    
    async fn resolve_alert(&self, rule: &AlertRule) {
        let mut active_alerts = self.active_alerts.lock().await;
        
        if let Some(alert) = active_alerts.get_mut(&rule.id) {
            if alert.status == AlertStatus::Firing {
                alert.status = AlertStatus::Resolved;
                alert.resolved_at = Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                );
                
                println!("✅ ALERT RESOLVED: {}", alert.message);
                
                // Send resolution notifications
                for channel in &rule.notification_channels {
                    self.send_resolution_notification(channel, alert).await;
                }
            }
        }
    }
    
    async fn send_notification(&self, channel: &str, alert: &Alert) {
        match channel {
            "email" => {
                println!("📧 Sending email notification for alert: {}", alert.id);
                // Implement email sending logic
            }
            "slack" => {
                println!("💬 Sending Slack notification for alert: {}", alert.id);
                // Implement Slack webhook logic
            }
            "webhook" => {
                println!("🔗 Sending webhook notification for alert: {}", alert.id);
                // Implement webhook logic
            }
            _ => {
                println!("❓ Unknown notification channel: {}", channel);
            }
        }
    }
    
    async fn send_resolution_notification(&self, channel: &str, alert: &Alert) {
        match channel {
            "email" => {
                println!("📧 Sending email resolution notification for alert: {}", alert.id);
            }
            "slack" => {
                println!("💬 Sending Slack resolution notification for alert: {}", alert.id);
            }
            "webhook" => {
                println!("🔗 Sending webhook resolution notification for alert: {}", alert.id);
            }
            _ => {
                println!("❓ Unknown notification channel: {}", channel);
            }
        }
    }
    
    pub async fn get_metric_values(&self, metric_name: &str, limit: usize) -> Vec<MetricValue> {
        let metrics = self.metrics.lock().await;
        
        if let Some(metric_history) = metrics.get(metric_name) {
            metric_history.iter()
                .rev()
                .take(limit)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
    
    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        let active_alerts = self.active_alerts.lock().await;
        active_alerts.values()
            .filter(|alert| alert.status == AlertStatus::Firing)
            .cloned()
            .collect()
    }
    
    pub async fn start_monitoring(&self) {
        let mut collection_interval = interval(Duration::from_secs(30));
        let mut alert_check_interval = interval(Duration::from_secs(10));
        
        loop {
            tokio::select! {
                _ = collection_interval.tick() => {
                    self.collect_metrics().await;
                }
                _ = alert_check_interval.tick() => {
                    let active_alerts = self.get_active_alerts().await;
                    if !active_alerts.is_empty() {
                        println!("🚨 Active alerts: {}", active_alerts.len());
                        for alert in &active_alerts {
                            println!("  - {} ({})", alert.message, format!("{:?}", alert.severity));
                        }
                    }
                }
            }
        }
    }
}

async fn initialize_monitoring_system(system: &MonitoringSystem) {
    // Register metric definitions
    let metric_definitions = vec![
        MetricDefinition {
            name: "system_cpu_usage".to_string(),
            metric_type: MetricType::Gauge,
            description: "CPU usage percentage".to_string(),
            labels: vec!["host".to_string()],
            unit: "percent".to_string(),
            collection_interval: Duration::from_secs(30),
        },
        MetricDefinition {
            name: "system_memory_usage".to_string(),
            metric_type: MetricType::Gauge,
            description: "Memory usage percentage".to_string(),
            labels: vec!["host".to_string()],
            unit: "percent".to_string(),
            collection_interval: Duration::from_secs(30),
        },
        MetricDefinition {
            name: "http_response_time".to_string(),
            metric_type: MetricType::Histogram,
            description: "HTTP response time".to_string(),
            labels: vec!["endpoint".to_string()],
            unit: "milliseconds".to_string(),
            collection_interval: Duration::from_secs(60),
        },
    ];
    
    for definition in metric_definitions {
        system.register_metric(definition).await;
    }
    
    // Add monitoring targets
    let targets = vec![
        MonitoringTarget {
            id: uuid::Uuid::new_v4().to_string(),
            name: "web-server-1".to_string(),
            target_type: TargetType::SystemMetrics,
            endpoint: "localhost".to_string(),
            credentials: None,
            metrics: vec!["system_cpu_usage".to_string(), "system_memory_usage".to_string()],
            collection_interval: Duration::from_secs(30),
            enabled: true,
        },
        MonitoringTarget {
            id: uuid::Uuid::new_v4().to_string(),
            name: "api-endpoint".to_string(),
            target_type: TargetType::HttpEndpoint,
            endpoint: "https://api.example.com/health".to_string(),
            credentials: None,
            metrics: vec!["http_response_time".to_string(), "http_status_code".to_string()],
            collection_interval: Duration::from_secs(60),
            enabled: true,
        },
        MonitoringTarget {
            id: uuid::Uuid::new_v4().to_string(),
            name: "webapp".to_string(),
            target_type: TargetType::ApplicationMetrics,
            endpoint: "localhost:8080".to_string(),
            credentials: None,
            metrics: vec!["app_requests_total".to_string(), "app_error_rate".to_string()],
            collection_interval: Duration::from_secs(30),
            enabled: true,
        },
    ];
    
    for target in targets {
        system.add_monitoring_target(target).await;
    }
    
    // Add alert rules
    let alert_rules = vec![
        AlertRule {
            id: uuid::Uuid::new_v4().to_string(),
            name: "High CPU Usage".to_string(),
            metric_name: "system_cpu_usage".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 80.0,
            duration: Duration::from_secs(300),
            severity: AlertSeverity::Warning,
            notification_channels: vec!["email".to_string(), "slack".to_string()],
            enabled: true,
        },
        AlertRule {
            id: uuid::Uuid::new_v4().to_string(),
            name: "High Memory Usage".to_string(),
            metric_name: "system_memory_usage".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 90.0,
            duration: Duration::from_secs(300),
            severity: AlertSeverity::Critical,
            notification_channels: vec!["email".to_string(), "slack".to_string(), "webhook".to_string()],
            enabled: true,
        },
        AlertRule {
            id: uuid::Uuid::new_v4().to_string(),
            name: "Slow HTTP Response".to_string(),
            metric_name: "http_response_time".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 200.0,
            duration: Duration::from_secs(120),
            severity: AlertSeverity::Warning,
            notification_channels: vec!["slack".to_string()],
            enabled: true,
        },
        AlertRule {
            id: uuid::Uuid::new_v4().to_string(),
            name: "High Error Rate".to_string(),
            metric_name: "app_error_rate".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 0.02, // 2% error rate
            duration: Duration::from_secs(180),
            severity: AlertSeverity::Critical,
            notification_channels: vec!["email".to_string(), "slack".to_string()],
            enabled: true,
        },
    ];
    
    for rule in alert_rules {
        system.add_alert_rule(rule).await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Infrastructure Monitoring System Starting...");
    println!("===============================================");
    
    let monitoring_system = MonitoringSystem::new();
    
    // Initialize the monitoring system
    initialize_monitoring_system(&monitoring_system).await;
    
    println!("✅ Monitoring system initialized!");
    println!("🔍 Starting metric collection and alerting...");
    
    monitoring_system.start_monitoring().await;
    
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "infrastructure-monitoring"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4"] }
reqwest = { version = "0.11", features = ["json"] }
prometheus = "0.13"
influxdb = "0.7"
lettre = "0.11"
rand = "0.8"`,
          language: 'toml',
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
      tags: ['Web API', 'REST', 'Actix', 'Performance', 'Rust'],
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
          content: `use actix_web::{web, App, HttpServer, Result, HttpResponse, middleware::Logger};
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
      tags: ['Game Engine', 'Bevy', '3D Graphics', 'ECS', 'Rust'],
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
    },
    // High-Frequency Trading Templates
    {
      id: 'rust-hft-market-data',
      name: 'HFT Market Data Engine',
      description: 'Ultra-low latency market data processing engine with microsecond precision',
      category: 'HFT',
      difficulty: 'Expert',
      tags: ['HFT', 'Market Data', 'Low Latency', 'Trading', 'Real-time'],
      icon: <TrendingUp className="w-6 h-6 text-green-400" />,
      estimatedTime: '6-8 weeks',
      useCase: 'Process high-frequency market data feeds with minimal latency for algorithmic trading',
      techStack: ['Rust', 'UDP', 'Memory Mapping', 'Lock-free', 'SIMD'],
      features: [
        'Sub-microsecond latency',
        'Lock-free data structures',
        'SIMD optimizations',
        'Memory-mapped I/O',
        'Multi-exchange support'
      ],
      files: {
        'main.rs': {
          content: `use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::UdpSocket;
use crossbeam::queue::SegQueue;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct MarketDataMessage {
    pub timestamp: u64,        // Nanoseconds since epoch
    pub symbol_id: u32,        // Numeric symbol identifier
    pub message_type: u8,      // 1=Quote, 2=Trade, 3=OrderBook
    pub side: u8,              // 0=Buy, 1=Sell
    pub price: u64,            // Price in fixed-point (multiply by 1e-8)
    pub quantity: u64,         // Quantity in fixed-point
    pub sequence_number: u64,  // Message sequence number
}

#[derive(Debug, Clone)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
    pub order_count: u32,
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub last_update: u64,
    pub sequence: u64,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u64,
    pub side: TradeSide,
    pub trade_id: u64,
}

#[derive(Debug, Clone)]
pub enum TradeSide {
    Buy,
    Sell,
    Unknown,
}

pub struct MarketDataEngine {
    // Lock-free queues for different message types
    quote_queue: Arc<SegQueue<MarketDataMessage>>,
    trade_queue: Arc<SegQueue<MarketDataMessage>>,
    orderbook_queue: Arc<SegQueue<MarketDataMessage>>,
    
    // Atomic counters for performance monitoring
    messages_processed: Arc<AtomicU64>,
    bytes_processed: Arc<AtomicU64>,
    
    // Symbol mapping for fast lookups
    symbol_map: HashMap<u32, String>,
    
    // Order books (using RwLock for concurrent access)
    order_books: Arc<std::sync::RwLock<HashMap<String, OrderBook>>>,
}

impl MarketDataEngine {
    pub fn new() -> Self {
        let mut symbol_map = HashMap::new();
        symbol_map.insert(1, "AAPL".to_string());
        symbol_map.insert(2, "GOOGL".to_string());
        symbol_map.insert(3, "MSFT".to_string());
        symbol_map.insert(4, "TSLA".to_string());
        symbol_map.insert(5, "AMZN".to_string());
        
        Self {
            quote_queue: Arc::new(SegQueue::new()),
            trade_queue: Arc::new(SegQueue::new()),
            orderbook_queue: Arc::new(SegQueue::new()),
            messages_processed: Arc::new(AtomicU64::new(0)),
            bytes_processed: Arc::new(AtomicU64::new(0)),
            symbol_map,
            order_books: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn start_udp_listener(&self, bind_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let socket = UdpSocket::bind(bind_addr).await?;
        println!("📡 Market data engine listening on {}", bind_addr);
        
        let mut buffer = vec![0u8; 65536]; // 64KB buffer
        
        loop {
            match socket.recv(&mut buffer).await {
                Ok(size) => {
                    let start_time = self.get_timestamp_nanos();
                    
                    // Process the received data
                    self.process_market_data(&buffer[..size]).await;
                    
                    // Update performance counters
                    self.messages_processed.fetch_add(1, Ordering::Relaxed);
                    self.bytes_processed.fetch_add(size as u64, Ordering::Relaxed);
                    
                    let processing_time = self.get_timestamp_nanos() - start_time;
                    if processing_time > 1000 { // Log if processing takes > 1 microsecond
                        println!("⚠️ Slow processing: {} ns", processing_time);
                    }
                }
                Err(e) => {
                    eprintln!("❌ UDP receive error: {}", e);
                }
            }
        }
    }
    
    async fn process_market_data(&self, data: &[u8]) {
        // Parse binary market data messages
        let message_size = std::mem::size_of::<MarketDataMessage>();
        
        for chunk in data.chunks_exact(message_size) {
            if let Ok(message) = self.parse_message(chunk) {
                match message.message_type {
                    1 => self.quote_queue.push(message),
                    2 => {
                        self.trade_queue.push(message);
                        self.process_trade(message).await;
                    }
                    3 => {
                        self.orderbook_queue.push(message);
                        self.update_order_book(message).await;
                    }
                    _ => {} // Unknown message type
                }
            }
        }
    }
    
    fn parse_message(&self, data: &[u8]) -> Result<MarketDataMessage, &'static str> {
        if data.len() != std::mem::size_of::<MarketDataMessage>() {
            return Err("Invalid message size");
        }
        
        // Safe conversion from bytes to struct
        let message = unsafe {
            std::ptr::read(data.as_ptr() as *const MarketDataMessage)
        };
        
        Ok(message)
    }
    
    async fn process_trade(&self, message: MarketDataMessage) {
        if let Some(symbol) = self.symbol_map.get(&message.symbol_id) {
            let trade = Trade {
                symbol: symbol.clone(),
                price: message.price as f64 * 1e-8,
                quantity: message.quantity as f64 * 1e-8,
                timestamp: message.timestamp,
                side: if message.side == 0 { TradeSide::Buy } else { TradeSide::Sell },
                trade_id: message.sequence_number,
            };
            
            // Process trade for analytics, strategies, etc.
            self.on_trade_received(trade).await;
        }
    }
    
    async fn update_order_book(&self, message: MarketDataMessage) {
        if let Some(symbol) = self.symbol_map.get(&message.symbol_id) {
            let mut order_books = self.order_books.write().unwrap();
            
            let order_book = order_books.entry(symbol.clone()).or_insert_with(|| OrderBook {
                symbol: symbol.clone(),
                bids: Vec::new(),
                asks: Vec::new(),
                last_update: 0,
                sequence: 0,
            });
            
            // Update order book based on message
            let price = message.price as f64 * 1e-8;
            let quantity = message.quantity as f64 * 1e-8;
            
            if message.side == 0 { // Bid
                self.update_book_side(&mut order_book.bids, price, quantity, false);
            } else { // Ask
                self.update_book_side(&mut order_book.asks, price, quantity, true);
            }
            
            order_book.last_update = message.timestamp;
            order_book.sequence = message.sequence_number;
        }
    }
    
    fn update_book_side(&self, levels: &mut Vec<OrderBookLevel>, price: f64, quantity: f64, is_ask: bool) {
        // Find existing level or insert new one
        match levels.binary_search_by(|level| {
            if is_ask {
                level.price.partial_cmp(&price).unwrap()
            } else {
                price.partial_cmp(&level.price).unwrap()
            }
        }) {
            Ok(index) => {
                if quantity == 0.0 {
                    levels.remove(index);
                } else {
                    levels[index].quantity = quantity;
                }
            }
            Err(index) => {
                if quantity > 0.0 {
                    levels.insert(index, OrderBookLevel {
                        price,
                        quantity,
                        order_count: 1,
                    });
                }
            }
        }
    }
    
    async fn on_trade_received(&self, trade: Trade) {
        // This is where trading strategies would be triggered
        println!("💰 Trade: {} {} @ {} ({})", 
                trade.symbol, 
                trade.quantity, 
                trade.price,
                match trade.side {
                    TradeSide::Buy => "BUY",
                    TradeSide::Sell => "SELL",
                    TradeSide::Unknown => "UNKNOWN",
                });
    }
    
    pub fn get_best_bid_ask(&self, symbol: &str) -> Option<(f64, f64)> {
        let order_books = self.order_books.read().unwrap();
        
        if let Some(book) = order_books.get(symbol) {
            let best_bid = book.bids.first().map(|level| level.price);
            let best_ask = book.asks.first().map(|level| level.price);
            
            match (best_bid, best_ask) {
                (Some(bid), Some(ask)) => Some((bid, ask)),
                _ => None,
            }
        } else {
            None
        }
    }
    
    pub fn get_mid_price(&self, symbol: &str) -> Option<f64> {
        self.get_best_bid_ask(symbol)
            .map(|(bid, ask)| (bid + ask) / 2.0)
    }
    
    fn get_timestamp_nanos(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    
    pub fn get_performance_stats(&self) -> (u64, u64) {
        (
            self.messages_processed.load(Ordering::Relaxed),
            self.bytes_processed.load(Ordering::Relaxed),
        )
    }
    
    pub async fn start_performance_monitor(&self) {
        let messages_processed = self.messages_processed.clone();
        let bytes_processed = self.bytes_processed.clone();
        
        tokio::spawn(async move {
            let mut last_messages = 0;
            let mut last_bytes = 0;
            
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                
                let current_messages = messages_processed.load(Ordering::Relaxed);
                let current_bytes = bytes_processed.load(Ordering::Relaxed);
                
                let msg_rate = current_messages - last_messages;
                let byte_rate = current_bytes - last_bytes;
                
                println!("📊 Performance: {} msg/s, {:.2} MB/s", 
                        msg_rate, 
                        byte_rate as f64 / 1_000_000.0);
                
                last_messages = current_messages;
                last_bytes = current_bytes;
            }
        });
    }
}

// Simulate market data for testing
async fn simulate_market_data(target_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    let mut sequence = 0u64;
    
    loop {
        for symbol_id in 1..=5 {
            // Generate random trade
            let trade_message = MarketDataMessage {
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
                symbol_id,
                message_type: 2, // Trade
                side: if rand::random::<bool>() { 0 } else { 1 },
                price: ((100.0 + rand::random::<f64>() * 50.0) * 1e8) as u64,
                quantity: ((100.0 + rand::random::<f64>() * 1000.0) * 1e8) as u64,
                sequence_number: sequence,
            };
            
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &trade_message as *const _ as *const u8,
                    std::mem::size_of::<MarketDataMessage>(),
                )
            };
            
            socket.send_to(bytes, target_addr).await?;
            sequence += 1;
        }
        
        tokio::time::sleep(tokio::time::Duration::from_micros(100)).await; // 10k messages/second
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 HFT Market Data Engine Starting...");
    println!("=====================================");
    
    let engine = MarketDataEngine::new();
    
    // Start performance monitoring
    engine.start_performance_monitor().await;
    
    // Start market data simulation in background
    tokio::spawn(async move {
        if let Err(e) = simulate_market_data("127.0.0.1:9999").await {
            eprintln!("❌ Market data simulation error: {}", e);
        }
    });
    
    // Start the main market data listener
    engine.start_udp_listener("127.0.0.1:9999").await?;
    
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "hft-market-data-engine"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
crossbeam = "0.8"
serde = { version = "1.0", features = ["derive"] }
rand = "0.8"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3`,
          language: 'toml',
        }
      }
    },
    {
      id: 'rust-hft-trading-engine',
      name: 'HFT Trading Engine',
      description: 'Ultra-fast algorithmic trading engine with order management and risk controls',
      category: 'HFT',
      difficulty: 'Expert',
      tags: ['HFT', 'Trading', 'Algorithms', 'Order Management', 'Risk'],
      icon: <DollarSign className="w-6 h-6 text-yellow-400" />,
      estimatedTime: '8-10 weeks',
      useCase: 'Execute high-frequency trading strategies with microsecond order placement',
      techStack: ['Rust', 'FIX Protocol', 'Lock-free', 'FPGA', 'Kernel Bypass'],
      features: [
        'Sub-microsecond order placement',
        'FIX protocol implementation',
        'Real-time risk management',
        'Strategy backtesting',
        'Multi-venue connectivity'
      ],
      files: {
        'main.rs': {
          content: `use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use crossbeam::queue::SegQueue;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    PendingCancel,
    PendingReplace,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeInForce {
    Day,
    GTC,  // Good Till Cancel
    IOC,  // Immediate or Cancel
    FOK,  // Fill or Kill
}

#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: u64,
    pub client_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    pub time_in_force: TimeInForce,
    pub status: OrderStatus,
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub avg_fill_price: f64,
    pub created_time: u64,
    pub last_update_time: u64,
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub fill_id: u64,
    pub order_id: u64,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: u64,
    pub venue: String,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub last_update: u64,
}

#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_position_size: f64,
    pub max_order_size: f64,
    pub max_daily_loss: f64,
    pub max_daily_volume: f64,
    pub position_limit_per_symbol: HashMap<String, f64>,
}

#[derive(Debug)]
pub struct TradingEngine {
    // Order management
    orders: Arc<std::sync::RwLock<HashMap<u64, Order>>>,
    order_id_counter: Arc<AtomicU64>,
    
    // Position tracking
    positions: Arc<std::sync::RwLock<HashMap<String, Position>>>,
    
    // Risk management
    risk_limits: Arc<std::sync::RwLock<RiskLimits>>,
    daily_pnl: Arc<AtomicU64>, // Stored as fixed-point
    daily_volume: Arc<AtomicU64>,
    
    // Order queues (lock-free for performance)
    new_order_queue: Arc<SegQueue<Order>>,
    cancel_order_queue: Arc<SegQueue<u64>>,
    fill_queue: Arc<SegQueue<Fill>>,
    
    // Trading enabled flag
    trading_enabled: Arc<AtomicBool>,
    
    // Performance counters
    orders_sent: Arc<AtomicU64>,
    orders_filled: Arc<AtomicU64>,
    orders_rejected: Arc<AtomicU64>,
}

impl TradingEngine {
    pub fn new() -> Self {
        let risk_limits = RiskLimits {
            max_position_size: 10000.0,
            max_order_size: 1000.0,
            max_daily_loss: 50000.0,
            max_daily_volume: 1000000.0,
            position_limit_per_symbol: {
                let mut limits = HashMap::new();
                limits.insert("AAPL".to_string(), 5000.0);
                limits.insert("GOOGL".to_string(), 1000.0);
                limits.insert("MSFT".to_string(), 5000.0);
                limits.insert("TSLA".to_string(), 2000.0);
                limits
            },
        };
        
        Self {
            orders: Arc::new(std::sync::RwLock::new(HashMap::new())),
            order_id_counter: Arc::new(AtomicU64::new(1)),
            positions: Arc::new(std::sync::RwLock::new(HashMap::new())),
            risk_limits: Arc::new(std::sync::RwLock::new(risk_limits)),
            daily_pnl: Arc::new(AtomicU64::new(0)),
            daily_volume: Arc::new(AtomicU64::new(0)),
            new_order_queue: Arc::new(SegQueue::new()),
            cancel_order_queue: Arc::new(SegQueue::new()),
            fill_queue: Arc::new(SegQueue::new()),
            trading_enabled: Arc::new(AtomicBool::new(true)),
            orders_sent: Arc::new(AtomicU64::new(0)),
            orders_filled: Arc::new(AtomicU64::new(0)),
            orders_rejected: Arc::new(AtomicU64::new(0)),
        }
    }
    
    pub fn submit_order(&self, mut order: Order) -> Result<u64, String> {
        if !self.trading_enabled.load(Ordering::Relaxed) {
            return Err("Trading is disabled".to_string());
        }
        
        // Generate unique order ID
        let order_id = self.order_id_counter.fetch_add(1, Ordering::Relaxed);
        order.order_id = order_id;
        order.created_time = self.get_timestamp_nanos();
        order.last_update_time = order.created_time;
        order.remaining_quantity = order.quantity;
        
        // Pre-trade risk checks
        if let Err(e) = self.check_pre_trade_risk(&order) {
            self.orders_rejected.fetch_add(1, Ordering::Relaxed);
            return Err(e);
        }
        
        // Store order
        {
            let mut orders = self.orders.write().unwrap();
            orders.insert(order_id, order.clone());
        }
        
        // Queue for processing
        self.new_order_queue.push(order);
        self.orders_sent.fetch_add(1, Ordering::Relaxed);
        
        println!("📤 Order submitted: {} {} {} @ {:?}", 
                order_id, 
                order.symbol, 
                order.quantity,
                order.price);
        
        Ok(order_id)
    }
    
    pub fn cancel_order(&self, order_id: u64) -> Result<(), String> {
        {
            let mut orders = self.orders.write().unwrap();
            if let Some(order) = orders.get_mut(&order_id) {
                if order.status == OrderStatus::Filled || order.status == OrderStatus::Cancelled {
                    return Err("Cannot cancel order in current status".to_string());
                }
                order.status = OrderStatus::PendingCancel;
                order.last_update_time = self.get_timestamp_nanos();
            } else {
                return Err("Order not found".to_string());
            }
        }
        
        self.cancel_order_queue.push(order_id);
        println!("❌ Cancel request: {}", order_id);
        
        Ok(())
    }
    
    fn check_pre_trade_risk(&self, order: &Order) -> Result<(), String> {
        let risk_limits = self.risk_limits.read().unwrap();
        
        // Check order size limit
        if order.quantity > risk_limits.max_order_size {
            return Err(format!("Order size {} exceeds limit {}", 
                             order.quantity, risk_limits.max_order_size));
        }
        
        // Check symbol-specific position limit
        if let Some(&symbol_limit) = risk_limits.position_limit_per_symbol.get(&order.symbol) {
            let positions = self.positions.read().unwrap();
            let current_position = positions.get(&order.symbol)
                .map(|p| p.quantity)
                .unwrap_or(0.0);
            
            let new_position = match order.side {
                OrderSide::Buy => current_position + order.quantity,
                OrderSide::Sell => current_position - order.quantity,
            };
            
            if new_position.abs() > symbol_limit {
                return Err(format!("Position limit exceeded for {}: {} > {}", 
                                 order.symbol, new_position.abs(), symbol_limit));
            }
        }
        
        // Check daily loss limit
        let daily_pnl = self.daily_pnl.load(Ordering::Relaxed) as f64 / 1e8;
        if daily_pnl < -risk_limits.max_daily_loss {
            return Err(format!("Daily loss limit exceeded: {}", daily_pnl));
        }
        
        Ok(())
    }
    
    pub fn process_fill(&self, fill: Fill) {
        self.fill_queue.push(fill.clone());
        
        // Update order status
        {
            let mut orders = self.orders.write().unwrap();
            if let Some(order) = orders.get_mut(&fill.order_id) {
                order.filled_quantity += fill.quantity;
                order.remaining_quantity -= fill.quantity;
                order.avg_fill_price = if order.filled_quantity > 0.0 {
                    (order.avg_fill_price * (order.filled_quantity - fill.quantity) + 
                     fill.price * fill.quantity) / order.filled_quantity
                } else {
                    fill.price
                };
                
                order.status = if order.remaining_quantity <= 0.0 {
                    OrderStatus::Filled
                } else {
                    OrderStatus::PartiallyFilled
                };
                
                order.last_update_time = fill.timestamp;
            }
        }
        
        // Update position
        self.update_position(&fill);
        
        // Update performance counters
        self.orders_filled.fetch_add(1, Ordering::Relaxed);
        let volume = (fill.quantity * fill.price * 1e8) as u64;
        self.daily_volume.fetch_add(volume, Ordering::Relaxed);
        
        println!("✅ Fill: {} {} @ {} (Order: {})", 
                fill.quantity, fill.symbol, fill.price, fill.order_id);
    }
    
    fn update_position(&self, fill: &Fill) {
        let mut positions = self.positions.write().unwrap();
        let position = positions.entry(fill.symbol.clone()).or_insert_with(|| Position {
            symbol: fill.symbol.clone(),
            quantity: 0.0,
            avg_price: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            last_update: 0,
        });
        
        let fill_quantity = match fill.side {
            OrderSide::Buy => fill.quantity,
            OrderSide::Sell => -fill.quantity,
        };
        
        // Calculate realized PnL for closing trades
        if (position.quantity > 0.0 && fill_quantity < 0.0) || 
           (position.quantity < 0.0 && fill_quantity > 0.0) {
            let closing_quantity = fill_quantity.abs().min(position.quantity.abs());
            let realized_pnl = match fill.side {
                OrderSide::Sell => (fill.price - position.avg_price) * closing_quantity,
                OrderSide::Buy => (position.avg_price - fill.price) * closing_quantity,
            };
            position.realized_pnl += realized_pnl;
            
            // Update daily PnL
            let daily_pnl_change = (realized_pnl * 1e8) as i64;
            let current_pnl = self.daily_pnl.load(Ordering::Relaxed) as i64;
            self.daily_pnl.store((current_pnl + daily_pnl_change) as u64, Ordering::Relaxed);
        }
        
        // Update position
        if position.quantity == 0.0 {
            position.avg_price = fill.price;
        } else if (position.quantity > 0.0 && fill_quantity > 0.0) || 
                  (position.quantity < 0.0 && fill_quantity < 0.0) {
            // Adding to position
            position.avg_price = (position.avg_price * position.quantity.abs() + 
                                fill.price * fill.quantity) / 
                               (position.quantity.abs() + fill.quantity);
        }
        
        position.quantity += fill_quantity;
        position.last_update = fill.timestamp;
    }
    
    pub fn get_position(&self, symbol: &str) -> Option<Position> {
        let positions = self.positions.read().unwrap();
        positions.get(symbol).cloned()
    }
    
    pub fn get_all_positions(&self) -> Vec<Position> {
        let positions = self.positions.read().unwrap();
        positions.values().cloned().collect()
    }
    
    pub fn update_unrealized_pnl(&self, symbol: &str, current_price: f64) {
        let mut positions = self.positions.write().unwrap();
        if let Some(position) = positions.get_mut(symbol) {
            if position.quantity != 0.0 {
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity;
            }
        }
    }
    
    pub fn get_order_status(&self, order_id: u64) -> Option<OrderStatus> {
        let orders = self.orders.read().unwrap();
        orders.get(&order_id).map(|order| order.status)
    }
    
    pub fn enable_trading(&self) {
        self.trading_enabled.store(true, Ordering::Relaxed);
        println!("✅ Trading enabled");
    }
    
    pub fn disable_trading(&self) {
        self.trading_enabled.store(false, Ordering::Relaxed);
        println!("🛑 Trading disabled");
    }
    
    pub fn emergency_stop(&self) {
        self.disable_trading();
        
        // Cancel all open orders
        let order_ids: Vec<u64> = {
            let orders = self.orders.read().unwrap();
            orders.values()
                .filter(|order| matches!(order.status, OrderStatus::New | OrderStatus::PartiallyFilled))
                .map(|order| order.order_id)
                .collect()
        };
        
        for order_id in order_ids {
            let _ = self.cancel_order(order_id);
        }
        
        println!("🚨 EMERGENCY STOP - All trading halted and orders cancelled");
    }
    
    pub fn get_performance_stats(&self) -> (u64, u64, u64, f64) {
        let orders_sent = self.orders_sent.load(Ordering::Relaxed);
        let orders_filled = self.orders_filled.load(Ordering::Relaxed);
        let orders_rejected = self.orders_rejected.load(Ordering::Relaxed);
        let daily_pnl = self.daily_pnl.load(Ordering::Relaxed) as f64 / 1e8;
        
        (orders_sent, orders_filled, orders_rejected, daily_pnl)
    }
    
    fn get_timestamp_nanos(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    
    pub async fn start_order_processor(&self) {
        let new_order_queue = self.new_order_queue.clone();
        let cancel_order_queue = self.cancel_order_queue.clone();
        let fill_queue = self.fill_queue.clone();
        
        tokio::spawn(async move {
            loop {
                // Process new orders
                while let Some(order) = new_order_queue.pop() {
                    // Simulate order routing to exchange
                    println!("🔄 Processing order: {}", order.order_id);
                    
                    // Simulate random fill after short delay
                    tokio::time::sleep(tokio::time::Duration::from_micros(50)).await;
                    
                    if rand::random::<f64>() > 0.1 { // 90% fill rate
                        let fill = Fill {
                            fill_id: rand::random(),
                            order_id: order.order_id,
                            symbol: order.symbol.clone(),
                            side: order.side,
                            quantity: order.quantity,
                            price: order.price.unwrap_or(100.0 + rand::random::<f64>() * 50.0),
                            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
                            venue: "NASDAQ".to_string(),
                        };
                        fill_queue.push(fill);
                    }
                }
                
                // Process cancellations
                while let Some(order_id) = cancel_order_queue.pop() {
                    println!("🔄 Processing cancel: {}", order_id);
                }
                
                tokio::time::sleep(tokio::time::Duration::from_micros(1)).await;
            }
        });
    }
    
    pub async fn start_fill_processor(&self) {
        let fill_queue = self.fill_queue.clone();
        let engine = Arc::new(self);
        
        tokio::spawn(async move {
            loop {
                while let Some(fill) = fill_queue.pop() {
                    engine.process_fill(fill);
                }
                tokio::time::sleep(tokio::time::Duration::from_micros(1)).await;
            }
        });
    }
}

// Simple market making strategy
pub struct MarketMakingStrategy {
    engine: Arc<TradingEngine>,
    symbols: Vec<String>,
    spread_bps: f64,
    order_size: f64,
}

impl MarketMakingStrategy {
    pub fn new(engine: Arc<TradingEngine>) -> Self {
        Self {
            engine,
            symbols: vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()],
            spread_bps: 5.0, // 5 basis points
            order_size: 100.0,
        }
    }
    
    pub async fn run(&self, mid_price: f64, symbol: &str) {
        let spread = mid_price * self.spread_bps / 10000.0;
        let bid_price = mid_price - spread / 2.0;
        let ask_price = mid_price + spread / 2.0;
        
        // Submit bid order
        let bid_order = Order {
            order_id: 0,
            client_order_id: format!("BID_{}", rand::random::<u32>()),
            symbol: symbol.to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: self.order_size,
            price: Some(bid_price),
            stop_price: None,
            time_in_force: TimeInForce::IOC,
            status: OrderStatus::New,
            filled_quantity: 0.0,
            remaining_quantity: 0.0,
            avg_fill_price: 0.0,
            created_time: 0,
            last_update_time: 0,
        };
        
        // Submit ask order
        let ask_order = Order {
            order_id: 0,
            client_order_id: format!("ASK_{}", rand::random::<u32>()),
            symbol: symbol.to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Limit,
            quantity: self.order_size,
            price: Some(ask_price),
            stop_price: None,
            time_in_force: TimeInForce::IOC,
            status: OrderStatus::New,
            filled_quantity: 0.0,
            remaining_quantity: 0.0,
            avg_fill_price: 0.0,
            created_time: 0,
            last_update_time: 0,
        };
        
        let _ = self.engine.submit_order(bid_order);
        let _ = self.engine.submit_order(ask_order);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 HFT Trading Engine Starting...");
    println!("==================================");
    
    let engine = Arc::new(TradingEngine::new());
    
    // Start order and fill processors
    engine.start_order_processor().await;
    engine.start_fill_processor().await;
    
    // Create market making strategy
    let strategy = MarketMakingStrategy::new(engine.clone());
    
    // Performance monitoring
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            let (sent, filled, rejected, pnl) = engine_clone.get_performance_stats();
            println!("📊 Stats: {} sent, {} filled, {} rejected, PnL: ${:.2}", 
                    sent, filled, rejected, pnl);
            
            // Display positions
            let positions = engine_clone.get_all_positions();
            for position in positions {
                println!("📈 Position {}: {} @ {:.2} (PnL: {:.2})", 
                        position.symbol, 
                        position.quantity, 
                        position.avg_price,
                        position.realized_pnl + position.unrealized_pnl);
            }
        }
    });
    
    // Run market making strategy
    loop {
        for symbol in &["AAPL", "GOOGL", "MSFT"] {
            let mid_price = 100.0 + rand::random::<f64>() * 50.0;
            strategy.run(mid_price, symbol).await;
            
            // Update unrealized PnL
            engine.update_unrealized_pnl(symbol, mid_price);
        }
        
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "hft-trading-engine"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
crossbeam = "0.8"
serde = { version = "1.0", features = ["derive"] }
rand = "0.8"
uuid = { version = "1.0", features = ["v4"] }

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3
target-cpu = "native"`,
          language: 'toml',
        }
      }
    },
    {
      id: 'rust-hft-risk-manager',
      name: 'HFT Risk Management System',
      description: 'Real-time risk management system with position limits and automated controls',
      category: 'HFT',
      difficulty: 'Expert',
      tags: ['Risk Management', 'Real-time', 'Position Limits', 'VaR', 'Compliance'],
      icon: <Shield className="w-6 h-6 text-red-400" />,
      estimatedTime: '6-8 weeks',
      useCase: 'Monitor and control trading risks in real-time with automated circuit breakers',
      techStack: ['Rust', 'Real-time Analytics', 'Statistical Models', 'Alerting'],
      features: [
        'Real-time VaR calculation',
        'Position limit monitoring',
        'Automated circuit breakers',
        'Stress testing',
        'Regulatory reporting'
      ],
      files: {
        'main.rs': {
          content: `use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub portfolio_value: f64,
    pub daily_pnl: f64,
    pub var_95: f64,          // Value at Risk 95%
    pub var_99: f64,          // Value at Risk 99%
    pub expected_shortfall: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub beta: f64,
    pub leverage: f64,
    pub concentration_risk: f64,
    pub last_updated: u64,
}

#[derive(Debug, Clone)]
pub struct PositionRisk {
    pub symbol: String,
    pub position_size: f64,
    pub market_value: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub var_contribution: f64,
    pub concentration_pct: f64,
    pub limit_utilization: f64,
}

#[derive(Debug, Clone)]
pub struct RiskLimit {
    pub name: String,
    pub limit_type: RiskLimitType,
    pub threshold: f64,
    pub current_value: f64,
    pub utilization_pct: f64,
    pub breach_action: BreachAction,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub enum RiskLimitType {
    PositionLimit,
    VaRLimit,
    DailyLossLimit,
    ConcentrationLimit,
    LeverageLimit,
    DrawdownLimit,
    VolumeLimit,
}

#[derive(Debug, Clone)]
pub enum BreachAction {
    Alert,
    ReducePosition,
    StopTrading,
    EmergencyLiquidation,
}

#[derive(Debug, Clone)]
pub struct RiskAlert {
    pub id: u64,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub symbol: Option<String>,
    pub current_value: f64,
    pub threshold: f64,
    pub timestamp: u64,
    pub acknowledged: bool,
}

#[derive(Debug, Clone)]
pub enum AlertType {
    LimitBreach,
    VaRExceedance,
    ConcentrationRisk,
    LiquidityRisk,
    ModelRisk,
    OperationalRisk,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volatility: f64,
    pub correlation_matrix: HashMap<String, f64>,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub current_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

pub struct RiskManager {
    // Risk metrics and limits
    risk_metrics: Arc<RwLock<RiskMetrics>>,
    risk_limits: Arc<RwLock<HashMap<String, RiskLimit>>>,
    position_risks: Arc<RwLock<HashMap<String, PositionRisk>>>,
    
    // Market data and positions
    market_data: Arc<RwLock<HashMap<String, MarketData>>>,
    positions: Arc<RwLock<HashMap<String, Position>>>,
    
    // Historical data for VaR calculation
    price_history: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    pnl_history: Arc<RwLock<Vec<f64>>>,
    
    // Risk alerts
    active_alerts: Arc<RwLock<HashMap<u64, RiskAlert>>>,
    alert_id_counter: Arc<AtomicU64>,
    
    // Circuit breakers
    trading_halted: Arc<AtomicBool>,
    emergency_mode: Arc<AtomicBool>,
    
    // Performance counters
    risk_checks_performed: Arc<AtomicU64>,
    alerts_generated: Arc<AtomicU64>,
}

impl RiskManager {
    pub fn new() -> Self {
        let initial_metrics = RiskMetrics {
            portfolio_value: 0.0,
            daily_pnl: 0.0,
            var_95: 0.0,
            var_99: 0.0,
            expected_shortfall: 0.0,
            max_drawdown: 0.0,
            sharpe_ratio: 0.0,
            beta: 1.0,
            leverage: 1.0,
            concentration_risk: 0.0,
            last_updated: 0,
        };
        
        let mut risk_limits = HashMap::new();
        
        // Initialize default risk limits
        risk_limits.insert("portfolio_var_95".to_string(), RiskLimit {
            name: "Portfolio VaR 95%".to_string(),
            limit_type: RiskLimitType::VaRLimit,
            threshold: 100000.0,
            current_value: 0.0,
            utilization_pct: 0.0,
            breach_action: BreachAction::Alert,
            enabled: true,
        });
        
        risk_limits.insert("daily_loss_limit".to_string(), RiskLimit {
            name: "Daily Loss Limit".to_string(),
            limit_type: RiskLimitType::DailyLossLimit,
            threshold: 50000.0,
            current_value: 0.0,
            utilization_pct: 0.0,
            breach_action: BreachAction::StopTrading,
            enabled: true,
        });
        
        risk_limits.insert("leverage_limit".to_string(), RiskLimit {
            name: "Leverage Limit".to_string(),
            limit_type: RiskLimitType::LeverageLimit,
            threshold: 3.0,
            current_value: 1.0,
            utilization_pct: 33.3,
            breach_action: BreachAction::ReducePosition,
            enabled: true,
        });
        
        Self {
            risk_metrics: Arc::new(RwLock::new(initial_metrics)),
            risk_limits: Arc::new(RwLock::new(risk_limits)),
            position_risks: Arc::new(RwLock::new(HashMap::new())),
            market_data: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            price_history: Arc::new(RwLock::new(HashMap::new())),
            pnl_history: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_id_counter: Arc::new(AtomicU64::new(1)),
            trading_halted: Arc::new(AtomicBool::new(false)),
            emergency_mode: Arc::new(AtomicBool::new(false)),
            risk_checks_performed: Arc::new(AtomicU64::new(0)),
            alerts_generated: Arc::new(AtomicU64::new(0)),
        }
    }
    
    pub async fn update_position(&self, position: Position) {
        let mut positions = self.positions.write().await;
        positions.insert(position.symbol.clone(), position);
        
        // Trigger risk calculation
        drop(positions);
        self.calculate_risk_metrics().await;
    }
    
    pub async fn update_market_data(&self, market_data: MarketData) {
        // Update current market data
        {
            let mut data = self.market_data.write().await;
            data.insert(market_data.symbol.clone(), market_data.clone());
        }
        
        // Update price history for VaR calculation
        {
            let mut price_history = self.price_history.write().await;
            let history = price_history.entry(market_data.symbol.clone()).or_insert_with(Vec::new);
            history.push(market_data.price);
            
            // Keep only last 252 days (1 year of trading days)
            if history.len() > 252 {
                history.drain(0..history.len() - 252);
            }
        }
        
        // Recalculate risk metrics
        self.calculate_risk_metrics().await;
    }
    
    async fn calculate_risk_metrics(&self) {
        let positions = self.positions.read().await;
        let market_data = self.market_data.read().await;
        let price_history = self.price_history.read().await;
        
        let mut portfolio_value = 0.0;
        let mut daily_pnl = 0.0;
        let mut position_risks = HashMap::new();
        
        // Calculate portfolio metrics
        for position in positions.values() {
            portfolio_value += position.market_value;
            daily_pnl += position.unrealized_pnl + position.realized_pnl;
            
            // Calculate position-level risk metrics
            if let Some(data) = market_data.get(&position.symbol) {
                let position_risk = PositionRisk {
                    symbol: position.symbol.clone(),
                    position_size: position.quantity,
                    market_value: position.market_value,
                    delta: position.quantity, // Simplified delta
                    gamma: 0.0, // Would calculate based on options
                    vega: 0.0,  // Would calculate based on options
                    theta: 0.0, // Would calculate based on options
                    var_contribution: self.calculate_position_var(&position.symbol, position.market_value, &price_history).await,
                    concentration_pct: if portfolio_value > 0.0 { 
                        (position.market_value.abs() / portfolio_value) * 100.0 
                    } else { 
                        0.0 
                    },
                    limit_utilization: 0.0, // Would calculate based on position limits
                };
                position_risks.insert(position.symbol.clone(), position_risk);
            }
        }
        
        // Calculate VaR
        let var_95 = self.calculate_portfolio_var(0.95, &price_history, &positions).await;
        let var_99 = self.calculate_portfolio_var(0.99, &price_history, &positions).await;
        
        // Calculate other risk metrics
        let max_drawdown = self.calculate_max_drawdown().await;
        let sharpe_ratio = self.calculate_sharpe_ratio().await;
        let leverage = if portfolio_value > 0.0 {
            positions.values().map(|p| p.market_value.abs()).sum::<f64>() / portfolio_value
        } else {
            1.0
        };
        
        let concentration_risk = position_risks.values()
            .map(|pr| pr.concentration_pct)
            .fold(0.0, f64::max);
        
        // Update risk metrics
        {
            let mut metrics = self.risk_metrics.write().await;
            metrics.portfolio_value = portfolio_value;
            metrics.daily_pnl = daily_pnl;
            metrics.var_95 = var_95;
            metrics.var_99 = var_99;
            metrics.expected_shortfall = var_99 * 1.2; // Simplified ES calculation
            metrics.max_drawdown = max_drawdown;
            metrics.sharpe_ratio = sharpe_ratio;
            metrics.leverage = leverage;
            metrics.concentration_risk = concentration_risk;
            metrics.last_updated = self.get_timestamp_nanos();
        }
        
        // Update position risks
        {
            let mut pos_risks = self.position_risks.write().await;
            *pos_risks = position_risks;
        }
        
        // Check risk limits
        self.check_risk_limits().await;
        
        self.risk_checks_performed.fetch_add(1, Ordering::Relaxed);
    }
    
    async fn calculate_portfolio_var(&self, confidence: f64, price_history: &HashMap<String, Vec<f64>>, positions: &HashMap<String, Position>) -> f64 {
        let mut portfolio_returns = Vec::new();
        
        // Calculate historical portfolio returns
        let max_history = price_history.values().map(|h| h.len()).min().unwrap_or(0);
        if max_history < 2 {
            return 0.0;
        }
        
        for i in 1..max_history {
            let mut portfolio_return = 0.0;
            let mut total_value = 0.0;
            
            for position in positions.values() {
                if let Some(prices) = price_history.get(&position.symbol) {
                    if i < prices.len() {
                        let return_rate = (prices[i] - prices[i-1]) / prices[i-1];
                        portfolio_return += return_rate * position.market_value;
                        total_value += position.market_value.abs();
                    }
                }
            }
            
            if total_value > 0.0 {
                portfolio_returns.push(portfolio_return / total_value);
            }
        }
        
        if portfolio_returns.is_empty() {
            return 0.0;
        }
        
        // Sort returns and calculate VaR
        portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((1.0 - confidence) * portfolio_returns.len() as f64) as usize;
        let var_return = portfolio_returns.get(index).unwrap_or(&0.0);
        
        // Convert to dollar VaR
        let portfolio_value = positions.values().map(|p| p.market_value.abs()).sum::<f64>();
        -var_return * portfolio_value
    }
    
    async fn calculate_position_var(&self, _symbol: &str, position_value: f64, _price_history: &HashMap<String, Vec<f64>>) -> f64 {
        // Simplified position VaR calculation
        position_value * 0.02 // Assume 2% daily VaR
    }
    
    async fn calculate_max_drawdown(&self) -> f64 {
        let pnl_history = self.pnl_history.read().await;
        
        if pnl_history.len() < 2 {
            return 0.0;
        }
        
        let mut max_drawdown = 0.0;
        let mut peak = pnl_history[0];
        
        for &pnl in pnl_history.iter().skip(1) {
            if pnl > peak {
                peak = pnl;
            }
            let drawdown = (peak - pnl) / peak.abs().max(1.0);
            max_drawdown = max_drawdown.max(drawdown);
        }
        
        max_drawdown
    }
    
    async fn calculate_sharpe_ratio(&self) -> f64 {
        let pnl_history = self.pnl_history.read().await;
        
        if pnl_history.len() < 2 {
            return 0.0;
        }
        
        let mean_return = pnl_history.iter().sum::<f64>() / pnl_history.len() as f64;
        let variance = pnl_history.iter()
            .map(|&x| (x - mean_return).powi(2))
            .sum::<f64>() / pnl_history.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 {
            mean_return / std_dev * (252.0_f64).sqrt() // Annualized Sharpe
        } else {
            0.0
        }
    }
    
    async fn check_risk_limits(&self) {
        let metrics = self.risk_metrics.read().await;
        let mut limits = self.risk_limits.write().await;
        
        for (limit_id, limit) in limits.iter_mut() {
            if !limit.enabled {
                continue;
            }
            
            let current_value = match limit.limit_type {
                RiskLimitType::VaRLimit => metrics.var_95,
                RiskLimitType::DailyLossLimit => -metrics.daily_pnl,
                RiskLimitType::LeverageLimit => metrics.leverage,
                RiskLimitType::ConcentrationLimit => metrics.concentration_risk,
                RiskLimitType::DrawdownLimit => metrics.max_drawdown,
                _ => 0.0,
            };
            
            limit.current_value = current_value;
            limit.utilization_pct = if limit.threshold > 0.0 {
                (current_value / limit.threshold) * 100.0
            } else {
                0.0
            };
            
            // Check for breach
            if current_value > limit.threshold {
                self.handle_limit_breach(limit_id, limit).await;
            }
        }
    }
    
    async fn handle_limit_breach(&self, limit_id: &str, limit: &RiskLimit) {
        let alert = RiskAlert {
            id: self.alert_id_counter.fetch_add(1, Ordering::Relaxed),
            alert_type: AlertType::LimitBreach,
            severity: match limit.breach_action {
                BreachAction::Alert => AlertSeverity::Medium,
                BreachAction::ReducePosition => AlertSeverity::High,
                BreachAction::StopTrading => AlertSeverity::Critical,
                BreachAction::EmergencyLiquidation => AlertSeverity::Critical,
            },
            message: format!("Risk limit breach: {} - Current: {:.2}, Limit: {:.2}", 
                           limit.name, limit.current_value, limit.threshold),
            symbol: None,
            current_value: limit.current_value,
            threshold: limit.threshold,
            timestamp: self.get_timestamp_nanos(),
            acknowledged: false,
        };
        
        // Store alert
        {
            let mut alerts = self.active_alerts.write().await;
            alerts.insert(alert.id, alert.clone());
        }
        
        self.alerts_generated.fetch_add(1, Ordering::Relaxed);
        
        println!("🚨 RISK ALERT: {}", alert.message);
        
        // Execute breach action
        match limit.breach_action {
            BreachAction::Alert => {
                // Just log the alert
            }
            BreachAction::ReducePosition => {
                println!("📉 Reducing positions due to risk limit breach");
                // Would implement position reduction logic
            }
            BreachAction::StopTrading => {
                self.halt_trading().await;
            }
            BreachAction::EmergencyLiquidation => {
                self.emergency_liquidation().await;
            }
        }
    }
    
    pub async fn halt_trading(&self) {
        self.trading_halted.store(true, Ordering::Relaxed);
        println!("🛑 TRADING HALTED due to risk limit breach");
    }
    
    pub async fn resume_trading(&self) {
        self.trading_halted.store(false, Ordering::Relaxed);
        println!("✅ Trading resumed");
    }
    
    pub async fn emergency_liquidation(&self) {
        self.emergency_mode.store(true, Ordering::Relaxed);
        self.trading_halted.store(true, Ordering::Relaxed);
        
        println!("🚨 EMERGENCY LIQUIDATION INITIATED");
        
        // Would implement emergency liquidation logic
        let positions = self.positions.read().await;
        for position in positions.values() {
            println!("💥 Emergency liquidating position: {} {}", 
                    position.symbol, position.quantity);
        }
    }
    
    pub async fn get_risk_metrics(&self) -> RiskMetrics {
        self.risk_metrics.read().await.clone()
    }
    
    pub async fn get_active_alerts(&self) -> Vec<RiskAlert> {
        let alerts = self.active_alerts.read().await;
        alerts.values().filter(|alert| !alert.acknowledged).cloned().collect()
    }
    
    pub async fn acknowledge_alert(&self, alert_id: u64) {
        let mut alerts = self.active_alerts.write().await;
        if let Some(alert) = alerts.get_mut(&alert_id) {
            alert.acknowledged = true;
            println!("✅ Alert {} acknowledged", alert_id);
        }
    }
    
    pub fn is_trading_halted(&self) -> bool {
        self.trading_halted.load(Ordering::Relaxed)
    }
    
    pub fn is_emergency_mode(&self) -> bool {
        self.emergency_mode.load(Ordering::Relaxed)
    }
    
    pub fn get_performance_stats(&self) -> (u64, u64) {
        (
            self.risk_checks_performed.load(Ordering::Relaxed),
            self.alerts_generated.load(Ordering::Relaxed),
        )
    }
    
    fn get_timestamp_nanos(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    
    pub async fn start_risk_monitoring(&self) {
        let risk_manager = Arc::new(self);
        
        // Risk calculation loop
        let rm_clone = risk_manager.clone();
        tokio::spawn(async move {
            loop {
                rm_clone.calculate_risk_metrics().await;
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        });
        
        // Performance monitoring
        let rm_clone = risk_manager.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
                
                let metrics = rm_clone.get_risk_metrics().await;
                let (checks, alerts) = rm_clone.get_performance_stats();
                
                println!("📊 Risk Metrics:");
                println!("  Portfolio Value: ${:.2}", metrics.portfolio_value);
                println!("  Daily PnL: ${:.2}", metrics.daily_pnl);
                println!("  VaR 95%: ${:.2}", metrics.var_95);
                println!("  Leverage: {:.2}x", metrics.leverage);
                println!("  Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
                println!("  Checks: {}, Alerts: {}", checks, alerts);
                
                let active_alerts = rm_clone.get_active_alerts().await;
                if !active_alerts.is_empty() {
                    println!("🚨 Active Alerts: {}", active_alerts.len());
                }
            }
        });
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️ HFT Risk Management System Starting...");
    println!("==========================================");
    
    let risk_manager = RiskManager::new();
    
    // Start risk monitoring
    risk_manager.start_risk_monitoring().await;
    
    // Simulate some positions and market data
    let symbols = vec!["AAPL", "GOOGL", "MSFT", "TSLA"];
    
    for symbol in &symbols {
        let position = Position {
            symbol: symbol.to_string(),
            quantity: 1000.0 + rand::random::<f64>() * 2000.0,
            avg_price: 100.0 + rand::random::<f64>() * 50.0,
            current_price: 100.0 + rand::random::<f64>() * 50.0,
            market_value: 0.0, // Will be calculated
            unrealized_pnl: rand::random::<f64>() * 10000.0 - 5000.0,
            realized_pnl: rand::random::<f64>() * 5000.0 - 2500.0,
        };
        
        risk_manager.update_position(position).await;
        
        let market_data = MarketData {
            symbol: symbol.to_string(),
            price: 100.0 + rand::random::<f64>() * 50.0,
            volatility: 0.15 + rand::random::<f64>() * 0.25,
            correlation_matrix: HashMap::new(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
        };
        
        risk_manager.update_market_data(market_data).await;
    }
    
    // Keep the system running
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        
        // Simulate market data updates
        for symbol in &symbols {
            let market_data = MarketData {
                symbol: symbol.to_string(),
                price: 100.0 + rand::random::<f64>() * 50.0,
                volatility: 0.15 + rand::random::<f64>() * 0.25,
                correlation_matrix: HashMap::new(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            };
            
            risk_manager.update_market_data(market_data).await;
        }
    }
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "hft-risk-manager"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
rand = "0.8"
nalgebra = "0.32"
statrs = "0.16"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3`,
          language: 'toml',
        }
      }
    },
    {
      id: 'rust-hft-backtester',
      name: 'HFT Strategy Backtester',
      description: 'High-performance backtesting engine for HFT strategies with tick-level precision',
      category: 'HFT',
      difficulty: 'Advanced',
      tags: ['Backtesting', 'Strategy Testing', 'Performance Analysis', 'Tick Data'],
      icon: <BarChart3 className="w-6 h-6 text-blue-400" />,
      estimatedTime: '5-7 weeks',
      useCase: 'Backtest and optimize high-frequency trading strategies with historical data',
      techStack: ['Rust', 'Time Series', 'Statistics', 'Parallel Processing'],
      features: [
        'Tick-level backtesting',
        'Strategy optimization',
        'Performance analytics',
        'Risk-adjusted returns',
        'Parallel execution'
      ],
      files: {
        'main.rs': {
          content: `use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct TickData {
    pub timestamp: u64,
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub timestamp: u64,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub avg_trade_duration: f64,
    pub avg_trade_pnl: f64,
    pub commission_paid: f64,
    pub start_capital: f64,
    pub end_capital: f64,
    pub daily_returns: Vec<f64>,
    pub equity_curve: Vec<(u64, f64)>,
    pub trades: Vec<Trade>,
}

pub trait Strategy: Send + Sync {
    fn on_tick(&mut self, tick: &TickData, portfolio: &Portfolio) -> Vec<Order>;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct Order {
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub order_type: OrderType,
    pub price: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum OrderType {
    Market,
    Limit,
}

pub struct Portfolio {
    pub cash: f64,
    pub positions: HashMap<String, Position>,
    pub equity_history: Vec<(u64, f64)>,
    pub trades: Vec<Trade>,
    pub commission_rate: f64,
}

impl Portfolio {
    pub fn new(initial_capital: f64, commission_rate: f64) -> Self {
        Self {
            cash: initial_capital,
            positions: HashMap::new(),
            equity_history: Vec::new(),
            trades: Vec::new(),
            commission_rate,
        }
    }
    
    pub fn execute_order(&mut self, order: Order, tick: &TickData) -> Option<Trade> {
        let execution_price = match order.order_type {
            OrderType::Market => {
                match order.side {
                    TradeSide::Buy => tick.ask,
                    TradeSide::Sell => tick.bid,
                }
            }
            OrderType::Limit => {
                if let Some(limit_price) = order.price {
                    // Check if limit order can be executed
                    match order.side {
                        TradeSide::Buy => {
                            if tick.ask <= limit_price {
                                tick.ask
                            } else {
                                return None; // Order not filled
                            }
                        }
                        TradeSide::Sell => {
                            if tick.bid >= limit_price {
                                tick.bid
                            } else {
                                return None; // Order not filled
                            }
                        }
                    }
                } else {
                    return None; // Invalid limit order
                }
            }
        };
        
        let trade_value = order.quantity * execution_price;
        let commission = trade_value * self.commission_rate;
        
        // Check if we have enough cash for buy orders
        if order.side == TradeSide::Buy && self.cash < trade_value + commission {
            return None; // Insufficient funds
        }
        
        // Check if we have enough shares for sell orders
        if order.side == TradeSide::Sell {
            let current_position = self.positions.get(&order.symbol)
                .map(|p| p.quantity)
                .unwrap_or(0.0);
            if current_position < order.quantity {
                return None; // Insufficient shares
            }
        }
        
        // Execute the trade
        let trade = Trade {
            timestamp: tick.timestamp,
            symbol: order.symbol.clone(),
            side: order.side.clone(),
            quantity: order.quantity,
            price: execution_price,
            commission,
        };
        
        // Update cash
        match order.side {
            TradeSide::Buy => {
                self.cash -= trade_value + commission;
            }
            TradeSide::Sell => {
                self.cash += trade_value - commission;
            }
        }
        
        // Update position
        self.update_position(&trade);
        
        // Record trade
        self.trades.push(trade.clone());
        
        Some(trade)
    }
    
    fn update_position(&mut self, trade: &Trade) {
        let position = self.positions.entry(trade.symbol.clone()).or_insert_with(|| Position {
            symbol: trade.symbol.clone(),
            quantity: 0.0,
            avg_price: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        });
        
        match trade.side {
            TradeSide::Buy => {
                if position.quantity >= 0.0 {
                    // Adding to long position
                    position.avg_price = (position.avg_price * position.quantity + 
                                        trade.price * trade.quantity) / 
                                       (position.quantity + trade.quantity);
                    position.quantity += trade.quantity;
                } else {
                    // Covering short position
                    let closing_quantity = trade.quantity.min(-position.quantity);
                    let realized_pnl = (position.avg_price - trade.price) * closing_quantity;
                    position.realized_pnl += realized_pnl;
                    
                    position.quantity += trade.quantity;
                    
                    if position.quantity > 0.0 {
                        position.avg_price = trade.price;
                    }
                }
            }
            TradeSide::Sell => {
                if position.quantity <= 0.0 {
                    // Adding to short position
                    position.avg_price = (position.avg_price * (-position.quantity) + 
                                        trade.price * trade.quantity) / 
                                       (-position.quantity + trade.quantity);
                    position.quantity -= trade.quantity;
                } else {
                    // Closing long position
                    let closing_quantity = trade.quantity.min(position.quantity);
                    let realized_pnl = (trade.price - position.avg_price) * closing_quantity;
                    position.realized_pnl += realized_pnl;
                    
                    position.quantity -= trade.quantity;
                    
                    if position.quantity < 0.0 {
                        position.avg_price = trade.price;
                    }
                }
            }
        }
    }
    
    pub fn update_unrealized_pnl(&mut self, tick: &TickData) {
        if let Some(position) = self.positions.get_mut(&tick.symbol) {
            if position.quantity != 0.0 {
                let current_price = (tick.bid + tick.ask) / 2.0;
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity;
            }
        }
    }
    
    pub fn get_total_equity(&self, market_prices: &HashMap<String, f64>) -> f64 {
        let mut equity = self.cash;
        
        for position in self.positions.values() {
            if let Some(&price) = market_prices.get(&position.symbol) {
                equity += position.quantity * price;
            }
        }
        
        equity
    }
}

pub struct Backtester {
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub start_date: u64,
    pub end_date: u64,
}

impl Backtester {
    pub fn new(initial_capital: f64, commission_rate: f64) -> Self {
        Self {
            initial_capital,
            commission_rate,
            start_date: 0,
            end_date: u64::MAX,
        }
    }
    
    pub fn run_backtest<S: Strategy>(&self, mut strategy: S, tick_data: Vec<TickData>) -> BacktestResults {
        let mut portfolio = Portfolio::new(self.initial_capital, self.commission_rate);
        let mut market_prices = HashMap::new();
        
        println!("🔄 Running backtest for strategy: {}", strategy.name());
        println!("📊 Processing {} ticks...", tick_data.len());
        
        for (i, tick) in tick_data.iter().enumerate() {
            if tick.timestamp < self.start_date || tick.timestamp > self.end_date {
                continue;
            }
            
            // Update market prices
            market_prices.insert(tick.symbol.clone(), (tick.bid + tick.ask) / 2.0);
            
            // Update unrealized PnL
            portfolio.update_unrealized_pnl(tick);
            
            // Get strategy signals
            let orders = strategy.on_tick(tick, &portfolio);
            
            // Execute orders
            for order in orders {
                if let Some(trade) = portfolio.execute_order(order, tick) {
                    println!("💰 Trade: {} {} {} @ {:.4}", 
                            trade.symbol, 
                            if trade.side == TradeSide::Buy { "BUY" } else { "SELL" },
                            trade.quantity, 
                            trade.price);
                }
            }
            
            // Record equity
            let equity = portfolio.get_total_equity(&market_prices);
            portfolio.equity_history.push((tick.timestamp, equity));
            
            // Progress reporting
            if i % 10000 == 0 {
                println!("📈 Progress: {:.1}% - Equity: ${:.2}", 
                        (i as f64 / tick_data.len() as f64) * 100.0, equity);
            }
        }
        
        self.calculate_results(portfolio)
    }
    
    fn calculate_results(&self, portfolio: Portfolio) -> BacktestResults {
        let final_equity = portfolio.equity_history.last()
            .map(|(_, equity)| *equity)
            .unwrap_or(self.initial_capital);
        
        let total_return = (final_equity - self.initial_capital) / self.initial_capital;
        
        // Calculate daily returns
        let mut daily_returns = Vec::new();
        let mut daily_equity = HashMap::new();
        
        for (timestamp, equity) in &portfolio.equity_history {
            let day = timestamp / (24 * 60 * 60 * 1_000_000_000); // Convert to days
            daily_equity.insert(day, *equity);
        }
        
        let mut sorted_days: Vec<_> = daily_equity.keys().collect();
        sorted_days.sort();
        
        for window in sorted_days.windows(2) {
            let prev_equity = daily_equity[window[0]];
            let curr_equity = daily_equity[window[1]];
            let daily_return = (curr_equity - prev_equity) / prev_equity;
            daily_returns.push(daily_return);
        }
        
        // Calculate performance metrics
        let annualized_return = if daily_returns.len() > 0 {
            let avg_daily_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            (1.0 + avg_daily_return).powf(252.0) - 1.0 // 252 trading days
        } else {
            0.0
        };
        
        let sharpe_ratio = if daily_returns.len() > 1 {
            let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            let variance = daily_returns.iter()
                .map(|&r| (r - mean_return).powi(2))
                .sum::<f64>() / (daily_returns.len() - 1) as f64;
            let std_dev = variance.sqrt();
            
            if std_dev > 0.0 {
                (mean_return / std_dev) * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        let max_drawdown = self.calculate_max_drawdown(&portfolio.equity_history);
        
        // Trade statistics
        let winning_trades = portfolio.trades.iter()
            .filter(|trade| {
                // Simplified: assume all trades are profitable if price moved favorably
                true // Would need more sophisticated PnL calculation
            })
            .count();
        
        let win_rate = if portfolio.trades.len() > 0 {
            winning_trades as f64 / portfolio.trades.len() as f64
        } else {
            0.0
        };
        
        let total_commission = portfolio.trades.iter()
            .map(|trade| trade.commission)
            .sum::<f64>();
        
        let avg_trade_pnl = if portfolio.trades.len() > 0 {
            total_return * self.initial_capital / portfolio.trades.len() as f64
        } else {
            0.0
        };
        
        BacktestResults {
            total_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            profit_factor: 1.5, // Simplified calculation
            total_trades: portfolio.trades.len(),
            avg_trade_duration: 0.0, // Would calculate based on entry/exit times
            avg_trade_pnl,
            commission_paid: total_commission,
            start_capital: self.initial_capital,
            end_capital: final_equity,
            daily_returns,
            equity_curve: portfolio.equity_history,
            trades: portfolio.trades,
        }
    }
    
    fn calculate_max_drawdown(&self, equity_curve: &[(u64, f64)]) -> f64 {
        let mut max_drawdown = 0.0;
        let mut peak = 0.0;
        
        for (_, equity) in equity_curve {
            if *equity > peak {
                peak = *equity;
            }
            let drawdown = (peak - equity) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }
        
        max_drawdown
    }
    
    pub fn optimize_strategy<S, F>(&self, strategy_factory: F, tick_data: Vec<TickData>, param_ranges: Vec<(f64, f64, f64)>) -> (Vec<f64>, BacktestResults)
    where
        S: Strategy + 'static,
        F: Fn(&[f64]) -> S + Send + Sync,
    {
        println!("🔧 Starting strategy optimization...");
        
        // Generate parameter combinations
        let mut param_combinations = Vec::new();
        self.generate_param_combinations(&param_ranges, &mut Vec::new(), &mut param_combinations);
        
        println!("🧪 Testing {} parameter combinations...", param_combinations.len());
        
        // Run backtests in parallel
        let results: Vec<_> = param_combinations
            .par_iter()
            .map(|params| {
                let strategy = strategy_factory(params);
                let result = self.run_backtest(strategy, tick_data.clone());
                (params.clone(), result)
            })
            .collect();
        
        // Find best parameters based on Sharpe ratio
        let (best_params, best_result) = results
            .into_iter()
            .max_by(|(_, a), (_, b)| a.sharpe_ratio.partial_cmp(&b.sharpe_ratio).unwrap())
            .unwrap();
        
        println!("✅ Optimization complete!");
        println!("🏆 Best Sharpe ratio: {:.4}", best_result.sharpe_ratio);
        println!("📊 Best parameters: {:?}", best_params);
        
        (best_params, best_result)
    }
    
    fn generate_param_combinations(&self, ranges: &[(f64, f64, f64)], current: &mut Vec<f64>, combinations: &mut Vec<Vec<f64>>) {
        if current.len() == ranges.len() {
            combinations.push(current.clone());
            return;
        }
        
        let (min, max, step) = ranges[current.len()];
        let mut value = min;
        
        while value <= max {
            current.push(value);
            self.generate_param_combinations(ranges, current, combinations);
            current.pop();
            value += step;
        }
    }
}

// Example strategy: Simple Moving Average Crossover
pub struct MovingAverageCrossover {
    short_period: usize,
    long_period: usize,
    short_ma: VecDeque<f64>,
    long_ma: VecDeque<f64>,
    position: f64,
}

impl MovingAverageCrossover {
    pub fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            short_period,
            long_period,
            short_ma: VecDeque::new(),
            long_ma: VecDeque::new(),
            position: 0.0,
        }
    }
}

impl Strategy for MovingAverageCrossover {
    fn on_tick(&mut self, tick: &TickData, portfolio: &Portfolio) -> Vec<Order> {
        let price = (tick.bid + tick.ask) / 2.0;
        
        // Update moving averages
        self.short_ma.push_back(price);
        self.long_ma.push_back(price);
        
        if self.short_ma.len() > self.short_period {
            self.short_ma.pop_front();
        }
        if self.long_ma.len() > self.long_period {
            self.long_ma.pop_front();
        }
        
        // Check if we have enough data
        if self.short_ma.len() < self.short_period || self.long_ma.len() < self.long_period {
            return Vec::new();
        }
        
        // Calculate averages
        let short_avg = self.short_ma.iter().sum::<f64>() / self.short_ma.len() as f64;
        let long_avg = self.long_ma.iter().sum::<f64>() / self.long_ma.len() as f64;
        
        let current_position = portfolio.positions.get(&tick.symbol)
            .map(|p| p.quantity)
            .unwrap_or(0.0);
        
        let mut orders = Vec::new();
        
        // Generate signals
        if short_avg > long_avg && current_position <= 0.0 {
            // Buy signal
            if current_position < 0.0 {
                // Close short position
                orders.push(Order {
                    symbol: tick.symbol.clone(),
                    side: TradeSide::Buy,
                    quantity: -current_position,
                    order_type: OrderType::Market,
                    price: None,
                });
            }
            // Open long position
            let position_size = portfolio.cash * 0.1 / price; // Use 10% of capital
            orders.push(Order {
                symbol: tick.symbol.clone(),
                side: TradeSide::Buy,
                quantity: position_size,
                order_type: OrderType::Market,
                price: None,
            });
        } else if short_avg < long_avg && current_position >= 0.0 {
            // Sell signal
            if current_position > 0.0 {
                // Close long position
                orders.push(Order {
                    symbol: tick.symbol.clone(),
                    side: TradeSide::Sell,
                    quantity: current_position,
                    order_type: OrderType::Market,
                    price: None,
                });
            }
            // Open short position
            let position_size = portfolio.cash * 0.1 / price; // Use 10% of capital
            orders.push(Order {
                symbol: tick.symbol.clone(),
                side: TradeSide::Sell,
                quantity: position_size,
                order_type: OrderType::Market,
                price: None,
            });
        }
        
        orders
    }
    
    fn name(&self) -> &str {
        "Moving Average Crossover"
    }
}

// Generate sample tick data for testing
fn generate_sample_data(symbol: &str, num_ticks: usize) -> Vec<TickData> {
    let mut data = Vec::new();
    let mut price = 100.0;
    let mut timestamp = 1640995200_000_000_000u64; // 2022-01-01 00:00:00 UTC in nanoseconds
    
    for _ in 0..num_ticks {
        // Random walk with slight upward bias
        let change = (rand::random::<f64>() - 0.49) * 0.01;
        price *= 1.0 + change;
        
        let spread = price * 0.001; // 0.1% spread
        let bid = price - spread / 2.0;
        let ask = price + spread / 2.0;
        
        data.push(TickData {
            timestamp,
            symbol: symbol.to_string(),
            price,
            volume: 100.0 + rand::random::<f64>() * 1000.0,
            bid,
            ask,
            bid_size: 100.0 + rand::random::<f64>() * 500.0,
            ask_size: 100.0 + rand::random::<f64>() * 500.0,
        });
        
        timestamp += 1_000_000; // 1ms between ticks
    }
    
    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 HFT Strategy Backtester Starting...");
    println!("=====================================");
    
    // Generate sample data
    let tick_data = generate_sample_data("AAPL", 100_000);
    println!("📊 Generated {} ticks for backtesting", tick_data.len());
    
    // Create backtester
    let backtester = Backtester::new(100_000.0, 0.001); // $100k capital, 0.1% commission
    
    // Create strategy
    let strategy = MovingAverageCrossover::new(10, 30);
    
    // Run backtest
    let results = backtester.run_backtest(strategy, tick_data.clone());
    
    // Display results
    println!("\n📈 Backtest Results:");
    println!("====================");
    println!("Total Return: {:.2}%", results.total_return * 100.0);
    println!("Annualized Return: {:.2}%", results.annualized_return * 100.0);
    println!("Sharpe Ratio: {:.4}", results.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", results.max_drawdown * 100.0);
    println!("Win Rate: {:.2}%", results.win_rate * 100.0);
    println!("Total Trades: {}", results.total_trades);
    println!("Average Trade PnL: ${:.2}", results.avg_trade_pnl);
    println!("Commission Paid: ${:.2}", results.commission_paid);
    println!("Start Capital: ${:.2}", results.start_capital);
    println!("End Capital: ${:.2}", results.end_capital);
    
    // Optimize strategy parameters
    println!("\n🔧 Optimizing strategy parameters...");
    let param_ranges = vec![
        (5.0, 20.0, 5.0),   // Short MA period: 5, 10, 15, 20
        (20.0, 50.0, 10.0), // Long MA period: 20, 30, 40, 50
    ];
    
    let (best_params, best_results) = backtester.optimize_strategy(
        |params| MovingAverageCrossover::new(params[0] as usize, params[1] as usize),
        tick_data,
        param_ranges,
    );
    
    println!("\n🏆 Optimized Results:");
    println!("=====================");
    println!("Best Parameters: Short MA = {}, Long MA = {}", best_params[0], best_params[1]);
    println!("Optimized Sharpe Ratio: {:.4}", best_results.sharpe_ratio);
    println!("Optimized Total Return: {:.2}%", best_results.total_return * 100.0);
    
    Ok(())
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "hft-strategy-backtester"
version = "0.1.0"
edition = "2021"

[dependencies]
rayon = "1.7"
serde = { version = "1.0", features = ["derive"] }
rand = "0.8"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3
target-cpu = "native"`,
          language: 'toml',
        }
      }
    }
  ];

  const categories = ['all', 'AI/ML', 'Mobile', 'Web', 'Gaming', 'Blockchain', 'IoT', 'DevOps'];
  const categories = ['all', 'AI/ML', 'Mobile', 'Web', 'Gaming', 'Blockchain', 'IoT', 'DevOps', 'HFT'];

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