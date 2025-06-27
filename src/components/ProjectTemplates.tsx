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
                let prediction = self.forward(input)?;
                let loss = mse_loss(&prediction, target)?;
                
                // Backpropagation would go here
                total_loss += loss.to_scalar::<f64>()?;
            }
            
            println!("Epoch {}: Loss = {:.6}", epoch + 1, total_loss / data.len() as f64);
        }
        Ok(())
    }
}

fn mse_loss(prediction: &Tensor, target: &Tensor) -> Result<Tensor> {
    let diff = (prediction - target)?;
    let squared = diff.sqr()?;
    squared.mean_all()
}

fn main() -> Result<()> {
    println!("🧠 Neural Network Engine Starting...");
    
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Initialize network
    let vs = VarBuilder::zeros(candle_core::DType::F32, &device);
    let mut network = NeuralNetwork::new(vs, 784, 128, 10)?;
    
    // Generate sample data (MNIST-like)
    let input = Tensor::randn(0f32, 1f32, (100, 784), &device)?;
    let target = Tensor::randn(0f32, 1f32, (100, 10), &device)?;
    
    // Train the network
    let training_data = vec![(input, target)];
    network.train(&training_data, 100, 0.001)?;
    
    println!("✅ Training completed!");
    Ok(())
}`,
          language: 'rust'
        },
        'lib.rs': {
          content: `pub mod neural_network;
pub mod optimizers;
pub mod layers;
pub mod activations;
pub mod loss_functions;
pub mod data_loader;
pub mod metrics;

use candle_core::{Device, Tensor, Result};

pub struct MLFramework {
    device: Device,
    models: Vec<Box<dyn Model>>,
}

pub trait Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn backward(&mut self, gradient: &Tensor) -> Result<()>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn save(&self, path: &str) -> Result<()>;
    fn load(&mut self, path: &str) -> Result<()>;
}

impl MLFramework {
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        Ok(Self {
            device,
            models: Vec::new(),
        })
    }
    
    pub fn add_model(&mut self, model: Box<dyn Model>) {
        self.models.push(model);
    }
    
    pub fn train_distributed(&mut self, data: &[Tensor], epochs: usize) -> Result<()> {
        // Distributed training implementation
        println!("Starting distributed training across {} devices", self.get_device_count());
        Ok(())
    }
    
    fn get_device_count(&self) -> usize {
        // Return number of available GPUs
        1
    }
}`,
          language: 'rust'
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
tch = "0.13"
ndarray = "0.15"
rayon = "1.7"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
clap = { version = "4.0", features = ["derive"] }

[features]
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
mkl = ["candle-core/mkl"]`,
          language: 'toml'
        }
      }
    },
    {
      id: 'computer-vision-pipeline',
      name: 'Computer Vision Pipeline',
      description: 'Real-time computer vision processing with object detection and image classification',
      category: 'AI/ML',
      difficulty: 'Advanced',
      tags: ['Computer Vision', 'Object Detection', 'Real-time', 'OpenCV'],
      icon: <Eye className="w-6 h-6 text-blue-400" />,
      estimatedTime: '2-3 weeks',
      useCase: 'Build applications for autonomous vehicles, security systems, and medical imaging',
      techStack: ['Rust', 'OpenCV', 'YOLO', 'TensorFlow', 'ONNX'],
      features: [
        'Real-time object detection',
        'Image classification',
        'Face recognition',
        'Video stream processing',
        'Edge deployment ready'
      ],
      files: {
        'main.rs': {
          content: `use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    imgcodecs::{imread, imwrite, IMREAD_COLOR},
    imgproc::{rectangle, put_text, FONT_HERSHEY_SIMPLEX, LINE_8},
    objdetect::CascadeClassifier,
    prelude::*,
    videoio::{VideoCapture, CAP_ANY},
    Result,
};
use std::collections::HashMap;

pub struct ComputerVisionPipeline {
    face_cascade: CascadeClassifier,
    object_detector: ObjectDetector,
    classifier: ImageClassifier,
    tracking_objects: HashMap<u32, TrackedObject>,
}

#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub class_id: u32,
    pub confidence: f32,
    pub bbox: Rect,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct TrackedObject {
    pub id: u32,
    pub bbox: Rect,
    pub velocity: Point,
    pub last_seen: u64,
}

impl ComputerVisionPipeline {
    pub fn new() -> Result<Self> {
        let face_cascade = CascadeClassifier::new("haarcascade_frontalface_alt.xml")?;
        let object_detector = ObjectDetector::new("yolov5s.onnx")?;
        let classifier = ImageClassifier::new("resnet50.onnx")?;
        
        Ok(Self {
            face_cascade,
            object_detector,
            classifier,
            tracking_objects: HashMap::new(),
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
        
        // Update tracking
        self.update_tracking(&detections);
        
        Ok(detections)
    }
    
    fn detect_faces(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        let mut gray = Mat::default();
        opencv::imgproc::cvt_color(frame, &mut gray, opencv::imgproc::COLOR_BGR2GRAY, 0)?;
        
        let mut faces = Vector::<Rect>::new();
        self.face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            3,
            0,
            Size::new(30, 30),
            Size::new(0, 0),
        )?;
        
        let mut detections = Vec::new();
        for face in faces.iter() {
            detections.push(DetectedObject {
                class_id: 0,
                confidence: 0.9,
                bbox: face,
                label: "Face".to_string(),
            });
        }
        
        Ok(detections)
    }
    
    fn update_tracking(&mut self, detections: &[DetectedObject]) {
        // Simple tracking algorithm
        for detection in detections {
            let center = Point::new(
                detection.bbox.x + detection.bbox.width / 2,
                detection.bbox.y + detection.bbox.height / 2,
            );
            
            // Find closest existing track or create new one
            let track_id = self.find_or_create_track(center, &detection.bbox);
            
            if let Some(tracked) = self.tracking_objects.get_mut(&track_id) {
                let old_center = Point::new(
                    tracked.bbox.x + tracked.bbox.width / 2,
                    tracked.bbox.y + tracked.bbox.height / 2,
                );
                
                tracked.velocity = Point::new(
                    center.x - old_center.x,
                    center.y - old_center.y,
                );
                tracked.bbox = detection.bbox;
                tracked.last_seen = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
            }
        }
    }
    
    fn find_or_create_track(&mut self, center: Point, bbox: &Rect) -> u32 {
        // Find closest existing track within threshold
        let threshold = 50.0;
        let mut closest_id = None;
        let mut closest_distance = f64::INFINITY;
        
        for (id, tracked) in &self.tracking_objects {
            let tracked_center = Point::new(
                tracked.bbox.x + tracked.bbox.width / 2,
                tracked.bbox.y + tracked.bbox.height / 2,
            );
            
            let distance = ((center.x - tracked_center.x).pow(2) + 
                           (center.y - tracked_center.y).pow(2)) as f64;
            let distance = distance.sqrt();
            
            if distance < threshold && distance < closest_distance {
                closest_distance = distance;
                closest_id = Some(*id);
            }
        }
        
        if let Some(id) = closest_id {
            id
        } else {
            // Create new track
            let new_id = self.tracking_objects.len() as u32;
            self.tracking_objects.insert(new_id, TrackedObject {
                id: new_id,
                bbox: *bbox,
                velocity: Point::new(0, 0),
                last_seen: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            });
            new_id
        }
    }
    
    pub fn draw_detections(&self, frame: &mut Mat, detections: &[DetectedObject]) -> Result<()> {
        for detection in detections {
            // Draw bounding box
            rectangle(
                frame,
                detection.bbox,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                LINE_8,
                0,
            )?;
            
            // Draw label
            let label = format!("{}: {:.2}", detection.label, detection.confidence);
            put_text(
                frame,
                &label,
                Point::new(detection.bbox.x, detection.bbox.y - 10),
                FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                1,
                LINE_8,
                false,
            )?;
        }
        Ok(())
    }
}

// Placeholder structs for ONNX models
pub struct ObjectDetector {
    model_path: String,
}

impl ObjectDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        Ok(Self {
            model_path: model_path.to_string(),
        })
    }
    
    pub fn detect(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        // ONNX inference would go here
        Ok(vec![])
    }
}

pub struct ImageClassifier {
    model_path: String,
}

impl ImageClassifier {
    pub fn new(model_path: &str) -> Result<Self> {
        Ok(Self {
            model_path: model_path.to_string(),
        })
    }
    
    pub fn classify(&self, frame: &Mat) -> Result<(String, f32)> {
        // Classification inference would go here
        Ok(("unknown".to_string(), 0.0))
    }
}

fn main() -> Result<()> {
    println!("👁️ Computer Vision Pipeline Starting...");
    
    let mut pipeline = ComputerVisionPipeline::new()?;
    let mut cap = VideoCapture::new(0, CAP_ANY)?;
    
    if !cap.is_opened()? {
        println!("❌ Cannot open camera");
        return Ok(());
    }
    
    println!("📹 Camera opened successfully");
    
    loop {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;
        
        if frame.empty() {
            break;
        }
        
        // Process frame
        let detections = pipeline.process_frame(&frame)?;
        
        // Draw results
        let mut display_frame = frame.clone();
        pipeline.draw_detections(&mut display_frame, &detections)?;
        
        // Display frame (in real implementation)
        println!("Detected {} objects", detections.len());
        
        // Break on 'q' key (simplified for demo)
        break;
    }
    
    println!("✅ Computer Vision Pipeline completed!");
    Ok(())
}`,
          language: 'rust'
        }
      }
    },
    {
      id: 'reinforcement-learning-agent',
      name: 'Reinforcement Learning Agent',
      description: 'Deep Q-Network (DQN) agent for game playing and decision making',
      category: 'AI/ML',
      difficulty: 'Expert',
      tags: ['Reinforcement Learning', 'DQN', 'Game AI', 'Decision Making'],
      icon: <Target className="w-6 h-6 text-red-400" />,
      estimatedTime: '4-6 weeks',
      useCase: 'Create intelligent agents for games, trading, robotics, and autonomous systems',
      techStack: ['Rust', 'PyTorch', 'OpenAI Gym', 'TensorBoard', 'CUDA'],
      features: [
        'Deep Q-Network implementation',
        'Experience replay buffer',
        'Target network updates',
        'Multi-environment support',
        'Performance visualization'
      ],
      files: {
        'main.rs': {
          content: `use std::collections::VecDeque;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }
    
    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let mut rng = rand::thread_rng();
        let mut batch = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            if !self.buffer.is_empty() {
                let idx = rng.gen_range(0..self.buffer.len());
                batch.push(self.buffer[idx].clone());
            }
        }
        
        batch
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub struct DQNAgent {
    state_size: usize,
    action_size: usize,
    learning_rate: f64,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
    gamma: f64,
    replay_buffer: ReplayBuffer,
    target_update_frequency: usize,
    step_count: usize,
}

impl DQNAgent {
    pub fn new(
        state_size: usize,
        action_size: usize,
        learning_rate: f64,
        buffer_size: usize,
    ) -> Self {
        Self {
            state_size,
            action_size,
            learning_rate,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            gamma: 0.95,
            replay_buffer: ReplayBuffer::new(buffer_size),
            target_update_frequency: 1000,
            step_count: 0,
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
    
    fn predict_action(&self, state: &[f32]) -> usize {
        // Neural network forward pass would go here
        // For now, return random action
        rand::thread_rng().gen_range(0..self.action_size)
    }
    
    pub fn remember(&mut self, experience: Experience) {
        self.replay_buffer.push(experience);
    }
    
    pub fn replay(&mut self, batch_size: usize) -> Result<f64, String> {
        if self.replay_buffer.len() < batch_size {
            return Ok(0.0);
        }
        
        let batch = self.replay_buffer.sample(batch_size);
        let mut total_loss = 0.0;
        
        for experience in batch {
            let target = if experience.done {
                experience.reward
            } else {
                let next_q_values = self.predict_q_values(&experience.next_state);
                experience.reward + self.gamma * next_q_values.iter().fold(0.0f32, |a, &b| a.max(b)) as f64
            };
            
            // Update Q-network (simplified)
            let current_q_values = self.predict_q_values(&experience.state);
            let loss = (target - current_q_values[experience.action] as f64).powi(2);
            total_loss += loss;
        }
        
        // Decay epsilon
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }
        
        // Update target network
        self.step_count += 1;
        if self.step_count % self.target_update_frequency == 0 {
            self.update_target_network();
        }
        
        Ok(total_loss / batch_size as f64)
    }
    
    fn predict_q_values(&self, state: &[f32]) -> Vec<f32> {
        // Neural network forward pass would go here
        vec![0.0; self.action_size]
    }
    
    fn update_target_network(&self) {
        // Copy main network weights to target network
        println!("🎯 Updating target network at step {}", self.step_count);
    }
    
    pub fn save_model(&self, path: &str) -> Result<(), String> {
        // Save model weights
        println!("💾 Saving model to {}", path);
        Ok(())
    }
    
    pub fn load_model(&mut self, path: &str) -> Result<(), String> {
        // Load model weights
        println!("📂 Loading model from {}", path);
        Ok(())
    }
}

pub trait Environment {
    fn reset(&mut self) -> Vec<f32>;
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool);
    fn render(&self);
    fn action_space_size(&self) -> usize;
    fn state_space_size(&self) -> usize;
}

pub struct CartPoleEnvironment {
    position: f32,
    velocity: f32,
    angle: f32,
    angular_velocity: f32,
    steps: usize,
    max_steps: usize,
}

impl CartPoleEnvironment {
    pub fn new() -> Self {
        Self {
            position: 0.0,
            velocity: 0.0,
            angle: 0.0,
            angular_velocity: 0.0,
            steps: 0,
            max_steps: 500,
        }
    }
}

impl Environment for CartPoleEnvironment {
    fn reset(&mut self) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        self.position = rng.gen_range(-0.05..0.05);
        self.velocity = rng.gen_range(-0.05..0.05);
        self.angle = rng.gen_range(-0.05..0.05);
        self.angular_velocity = rng.gen_range(-0.05..0.05);
        self.steps = 0;
        
        vec![self.position, self.velocity, self.angle, self.angular_velocity]
    }
    
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        let force = if action == 0 { -10.0 } else { 10.0 };
        
        // Simplified physics
        let gravity = 9.8;
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let length = 0.5;
        let dt = 0.02;
        
        // Update physics (simplified)
        self.angular_velocity += (gravity * self.angle.sin() + force.cos() * force / (mass_cart + mass_pole)) * dt;
        self.angle += self.angular_velocity * dt;
        self.velocity += force / mass_cart * dt;
        self.position += self.velocity * dt;
        
        self.steps += 1;
        
        let done = self.angle.abs() > 0.2 || self.position.abs() > 2.4 || self.steps >= self.max_steps;
        let reward = if done { 0.0 } else { 1.0 };
        
        let state = vec![self.position, self.velocity, self.angle, self.angular_velocity];
        
        (state, reward, done)
    }
    
    fn render(&self) {
        println!("CartPole - Position: {:.3}, Angle: {:.3}, Steps: {}", 
                self.position, self.angle, self.steps);
    }
    
    fn action_space_size(&self) -> usize {
        2 // Left or Right
    }
    
    fn state_space_size(&self) -> usize {
        4 // Position, Velocity, Angle, Angular Velocity
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 Reinforcement Learning Agent Starting...");
    
    let mut env = CartPoleEnvironment::new();
    let mut agent = DQNAgent::new(
        env.state_space_size(),
        env.action_space_size(),
        0.001,
        10000,
    );
    
    let episodes = 1000;
    let mut scores = Vec::new();
    
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;
        
        loop {
            let action = agent.act(&state);
            let (next_state, reward, done) = env.step(action);
            
            agent.remember(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            state = next_state;
            total_reward += reward;
            steps += 1;
            
            if done {
                break;
            }
        }
        
        scores.push(total_reward);
        
        // Train the agent
        if agent.replay_buffer.len() > 32 {
            let loss = agent.replay(32)?;
            if episode % 100 == 0 {
                println!("Episode {}: Score = {:.1}, Loss = {:.6}, Epsilon = {:.3}", 
                        episode, total_reward, loss, agent.epsilon);
            }
        }
        
        // Save model periodically
        if episode % 500 == 0 && episode > 0 {
            agent.save_model(&format!("dqn_model_{}.pth", episode))?;
        }
    }
    
    let avg_score: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
    println!("🏆 Training completed! Average score: {:.2}", avg_score);
    
    Ok(())
}`,
          language: 'rust'
        }
      }
    },
    {
      id: 'nlp-transformer',
      name: 'NLP Transformer Engine',
      description: 'Custom transformer implementation for natural language processing tasks',
      category: 'AI/ML',
      difficulty: 'Expert',
      tags: ['NLP', 'Transformers', 'BERT', 'GPT', 'Attention'],
      icon: <FileText className="w-6 h-6 text-green-400" />,
      estimatedTime: '5-8 weeks',
      useCase: 'Build chatbots, language models, text classification, and translation systems',
      techStack: ['Rust', 'Transformers', 'Tokenizers', 'ONNX', 'HuggingFace'],
      features: [
        'Multi-head attention mechanism',
        'Positional encoding',
        'Layer normalization',
        'Pre-trained model loading',
        'Fine-tuning capabilities'
      ],
      files: {
        'main.rs': {
          content: `use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub dropout_prob: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            dropout_prob: 0.1,
        }
    }
}

pub struct MultiHeadAttention {
    config: TransformerConfig,
    query_weights: Vec<Vec<f32>>,
    key_weights: Vec<Vec<f32>>,
    value_weights: Vec<Vec<f32>>,
    output_weights: Vec<Vec<f32>>,
}

impl MultiHeadAttention {
    pub fn new(config: TransformerConfig) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        
        Self {
            query_weights: vec![vec![0.0; config.hidden_size]; config.hidden_size],
            key_weights: vec![vec![0.0; config.hidden_size]; config.hidden_size],
            value_weights: vec![vec![0.0; config.hidden_size]; config.hidden_size],
            output_weights: vec![vec![0.0; config.hidden_size]; config.hidden_size],
            config,
        }
    }
    
    pub fn forward(&self, hidden_states: &[Vec<f32>], attention_mask: Option<&[Vec<f32>]>) -> Vec<Vec<f32>> {
        let seq_len = hidden_states.len();
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden_size / num_heads;
        
        // Compute Q, K, V
        let queries = self.linear_transform(hidden_states, &self.query_weights);
        let keys = self.linear_transform(hidden_states, &self.key_weights);
        let values = self.linear_transform(hidden_states, &self.value_weights);
        
        // Reshape for multi-head attention
        let mut attention_output = vec![vec![0.0; hidden_size]; seq_len];
        
        for head in 0..num_heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;
            
            // Extract head-specific Q, K, V
            let head_queries: Vec<Vec<f32>> = queries.iter()
                .map(|q| q[head_start..head_end].to_vec())
                .collect();
            let head_keys: Vec<Vec<f32>> = keys.iter()
                .map(|k| k[head_start..head_end].to_vec())
                .collect();
            let head_values: Vec<Vec<f32>> = values.iter()
                .map(|v| v[head_start..head_end].to_vec())
                .collect();
            
            // Compute attention scores
            let attention_scores = self.compute_attention_scores(&head_queries, &head_keys);
            let attention_probs = self.softmax(&attention_scores, attention_mask);
            
            // Apply attention to values
            let head_output = self.apply_attention(&attention_probs, &head_values);
            
            // Concatenate head output
            for (i, output_row) in attention_output.iter_mut().enumerate() {
                for (j, &value) in head_output[i].iter().enumerate() {
                    output_row[head_start + j] = value;
                }
            }
        }
        
        // Apply output projection
        self.linear_transform(&attention_output, &self.output_weights)
    }
    
    fn linear_transform(&self, input: &[Vec<f32>], weights: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut output = vec![vec![0.0; weights[0].len()]; input.len()];
        
        for (i, input_row) in input.iter().enumerate() {
            for (j, weight_col) in weights.iter().enumerate() {
                for (k, &input_val) in input_row.iter().enumerate() {
                    output[i][j] += input_val * weight_col[k];
                }
            }
        }
        
        output
    }
    
    fn compute_attention_scores(&self, queries: &[Vec<f32>], keys: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = queries.len();
        let mut scores = vec![vec![0.0; seq_len]; seq_len];
        
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut score = 0.0;
                for k in 0..queries[i].len() {
                    score += queries[i][k] * keys[j][k];
                }
                scores[i][j] = score / (queries[i].len() as f32).sqrt();
            }
        }
        
        scores
    }
    
    fn softmax(&self, scores: &[Vec<f32>], mask: Option<&[Vec<f32>]>) -> Vec<Vec<f32>> {
        let mut probs = vec![vec![0.0; scores[0].len()]; scores.len()];
        
        for (i, score_row) in scores.iter().enumerate() {
            let mut max_score = f32::NEG_INFINITY;
            for &score in score_row {
                max_score = max_score.max(score);
            }
            
            let mut sum = 0.0;
            for (j, &score) in score_row.iter().enumerate() {
                let masked_score = if let Some(mask) = mask {
                    if mask[i][j] == 0.0 { f32::NEG_INFINITY } else { score }
                } else { score };
                
                probs[i][j] = (masked_score - max_score).exp();
                sum += probs[i][j];
            }
            
            for prob in &mut probs[i] {
                *prob /= sum;
            }
        }
        
        probs
    }
    
    fn apply_attention(&self, attention_probs: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = attention_probs.len();
        let hidden_dim = values[0].len();
        let mut output = vec![vec![0.0; hidden_dim]; seq_len];
        
        for i in 0..seq_len {
            for j in 0..seq_len {
                let prob = attention_probs[i][j];
                for k in 0..hidden_dim {
                    output[i][k] += prob * values[j][k];
                }
            }
        }
        
        output
    }
}

pub struct TransformerLayer {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

impl TransformerLayer {
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            attention: MultiHeadAttention::new(config.clone()),
            feed_forward: FeedForward::new(config.hidden_size, config.intermediate_size),
            layer_norm1: LayerNorm::new(config.hidden_size),
            layer_norm2: LayerNorm::new(config.hidden_size),
        }
    }
    
    pub fn forward(&self, hidden_states: &[Vec<f32>], attention_mask: Option<&[Vec<f32>]>) -> Vec<Vec<f32>> {
        // Self-attention with residual connection
        let attention_output = self.attention.forward(hidden_states, attention_mask);
        let attention_residual = self.add_residual(hidden_states, &attention_output);
        let attention_normalized = self.layer_norm1.forward(&attention_residual);
        
        // Feed-forward with residual connection
        let ff_output = self.feed_forward.forward(&attention_normalized);
        let ff_residual = self.add_residual(&attention_normalized, &ff_output);
        let output = self.layer_norm2.forward(&ff_residual);
        
        output
    }
    
    fn add_residual(&self, input: &[Vec<f32>], output: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input.iter().zip(output.iter())
            .map(|(inp, out)| {
                inp.iter().zip(out.iter())
                    .map(|(&i, &o)| i + o)
                    .collect()
            })
            .collect()
    }
}

pub struct FeedForward {
    linear1: Vec<Vec<f32>>,
    linear2: Vec<Vec<f32>>,
    bias1: Vec<f32>,
    bias2: Vec<f32>,
}

impl FeedForward {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            linear1: vec![vec![0.0; intermediate_size]; hidden_size],
            linear2: vec![vec![0.0; hidden_size]; intermediate_size],
            bias1: vec![0.0; intermediate_size],
            bias2: vec![0.0; hidden_size],
        }
    }
    
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // First linear transformation
        let mut intermediate = vec![vec![0.0; self.bias1.len()]; input.len()];
        for (i, input_row) in input.iter().enumerate() {
            for (j, &bias) in self.bias1.iter().enumerate() {
                intermediate[i][j] = bias;
                for (k, &input_val) in input_row.iter().enumerate() {
                    intermediate[i][j] += input_val * self.linear1[k][j];
                }
                // GELU activation
                intermediate[i][j] = self.gelu(intermediate[i][j]);
            }
        }
        
        // Second linear transformation
        let mut output = vec![vec![0.0; self.bias2.len()]; input.len()];
        for (i, intermediate_row) in intermediate.iter().enumerate() {
            for (j, &bias) in self.bias2.iter().enumerate() {
                output[i][j] = bias;
                for (k, &intermediate_val) in intermediate_row.iter().enumerate() {
                    output[i][j] += intermediate_val * self.linear2[k][j];
                }
            }
        }
        
        output
    }
    
    fn gelu(&self, x: f32) -> f32 {
        0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
    }
}

pub struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            gamma: vec![1.0; hidden_size],
            beta: vec![0.0; hidden_size],
            eps: 1e-12,
        }
    }
    
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut output = vec![vec![0.0; input[0].len()]; input.len()];
        
        for (i, input_row) in input.iter().enumerate() {
            let mean = input_row.iter().sum::<f32>() / input_row.len() as f32;
            let variance = input_row.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / input_row.len() as f32;
            
            for (j, &input_val) in input_row.iter().enumerate() {
                output[i][j] = self.gamma[j] * (input_val - mean) / (variance + self.eps).sqrt() + self.beta[j];
            }
        }
        
        output
    }
}

pub struct Transformer {
    config: TransformerConfig,
    embeddings: Vec<Vec<f32>>,
    position_embeddings: Vec<Vec<f32>>,
    layers: Vec<TransformerLayer>,
}

impl Transformer {
    pub fn new(config: TransformerConfig) -> Self {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(TransformerLayer::new(config.clone()));
        }
        
        Self {
            embeddings: vec![vec![0.0; config.hidden_size]; config.vocab_size],
            position_embeddings: vec![vec![0.0; config.hidden_size]; config.max_position_embeddings],
            layers,
            config,
        }
    }
    
    pub fn forward(&self, input_ids: &[usize], attention_mask: Option<&[Vec<f32>]>) -> Vec<Vec<f32>> {
        let seq_len = input_ids.len();
        
        // Embedding lookup
        let mut hidden_states = Vec::new();
        for (pos, &token_id) in input_ids.iter().enumerate() {
            let mut embedding = self.embeddings[token_id].clone();
            
            // Add positional embedding
            for (i, &pos_emb) in self.position_embeddings[pos].iter().enumerate() {
                embedding[i] += pos_emb;
            }
            
            hidden_states.push(embedding);
        }
        
        // Pass through transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask);
        }
        
        hidden_states
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 NLP Transformer Engine Starting...");
    
    let config = TransformerConfig::default();
    let transformer = Transformer::new(config);
    
    // Example input: "Hello world"
    let input_ids = vec![101, 7592, 2088, 102]; // [CLS] Hello world [SEP]
    
    println!("Input tokens: {:?}", input_ids);
    
    let output = transformer.forward(&input_ids, None);
    
    println!("Output shape: {} x {}", output.len(), output[0].len());
    println!("First token representation (first 10 dims): {:?}", 
             &output[0][0..10]);
    
    println!("✅ Transformer forward pass completed!");
    
    Ok(())
}`,
          language: 'rust'
        }
      }
    },
    // Existing mobile templates...
    {
      id: 'mobile-game-engine',
      name: 'Mobile Game Engine',
      description: 'Cross-platform 2D/3D game engine with physics and rendering',
      category: 'Mobile',
      difficulty: 'Advanced',
      tags: ['Game Development', 'Graphics', 'Physics', 'Cross-platform'],
      icon: <Gamepad2 className="w-6 h-6 text-purple-400" />,
      estimatedTime: '4-6 weeks',
      useCase: 'Create mobile games for iOS and Android with high performance',
      techStack: ['Rust', 'wgpu', 'winit', 'rapier', 'Flutter'],
      features: [
        '2D/3D rendering pipeline',
        'Physics simulation',
        'Audio system',
        'Input handling',
        'Asset management'
      ],
      files: {
        'main.rs': {
          content: `use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod game_engine;
mod renderer;
mod physics;
mod audio;
mod input;

use game_engine::GameEngine;

fn main() {
    println!("🎮 Mobile Game Engine Starting...");
    
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Mobile Game Engine")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();
    
    let mut game_engine = GameEngine::new(&window);
    
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        game_engine.resize(*physical_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                game_engine.update();
                game_engine.render();
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}`,
          language: 'rust'
        }
      }
    },
    {
      id: 'iot-dashboard',
      name: 'IoT Control Dashboard',
      description: 'Real-time IoT device monitoring and control system',
      category: 'IoT',
      difficulty: 'Intermediate',
      tags: ['IoT', 'Real-time', 'MQTT', 'Dashboard'],
      icon: <Wifi className="w-6 h-6 text-blue-400" />,
      estimatedTime: '2-3 weeks',
      useCase: 'Monitor and control IoT devices in smart homes and industrial settings',
      techStack: ['Rust', 'MQTT', 'WebSocket', 'InfluxDB', 'Grafana'],
      features: [
        'Real-time device monitoring',
        'MQTT communication',
        'Data visualization',
        'Alert system',
        'Device control interface'
      ],
      files: {
        'main.rs': {
          content: `use tokio;
use rumqttc::{MqttOptions, Client, QoS};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize)]
struct SensorData {
    device_id: String,
    sensor_type: String,
    value: f64,
    timestamp: u64,
    unit: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeviceCommand {
    device_id: String,
    command: String,
    parameters: serde_json::Value,
}

struct IoTDashboard {
    mqtt_client: Client,
    devices: std::collections::HashMap<String, DeviceStatus>,
}

#[derive(Debug, Clone)]
struct DeviceStatus {
    id: String,
    name: String,
    device_type: String,
    online: bool,
    last_seen: u64,
    sensors: Vec<SensorReading>,
}

#[derive(Debug, Clone)]
struct SensorReading {
    sensor_type: String,
    value: f64,
    unit: String,
    timestamp: u64,
}

impl IoTDashboard {
    fn new() -> Self {
        let mut mqttoptions = MqttOptions::new("iot-dashboard", "localhost", 1883);
        mqttoptions.set_keep_alive(Duration::from_secs(5));
        
        let (client, _connection) = Client::new(mqttoptions, 10);
        
        Self {
            mqtt_client: client,
            devices: std::collections::HashMap::new(),
        }
    }
    
    async fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🏠 IoT Dashboard Starting...");
        
        // Subscribe to device topics
        self.mqtt_client.subscribe("devices/+/sensors/+", QoS::AtMostOnce).await?;
        self.mqtt_client.subscribe("devices/+/status", QoS::AtMostOnce).await?;
        
        // Simulate some devices
        self.simulate_devices().await?;
        
        Ok(())
    }
    
    async fn simulate_devices(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Simulate temperature sensor
        let temp_data = SensorData {
            device_id: "living-room-sensor".to_string(),
            sensor_type: "temperature".to_string(),
            value: 22.5,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            unit: "°C".to_string(),
        };
        
        let payload = serde_json::to_string(&temp_data)?;
        self.mqtt_client.publish(
            "devices/living-room-sensor/sensors/temperature",
            QoS::AtMostOnce,
            false,
            payload,
        ).await?;
        
        // Simulate humidity sensor
        let humidity_data = SensorData {
            device_id: "living-room-sensor".to_string(),
            sensor_type: "humidity".to_string(),
            value: 45.2,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            unit: "%".to_string(),
        };
        
        let payload = serde_json::to_string(&humidity_data)?;
        self.mqtt_client.publish(
            "devices/living-room-sensor/sensors/humidity",
            QoS::AtMostOnce,
            false,
            payload,
        ).await?;
        
        println!("📊 Simulated sensor data published");
        Ok(())
    }
    
    async fn send_command(&mut self, device_id: &str, command: &str, parameters: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        let cmd = DeviceCommand {
            device_id: device_id.to_string(),
            command: command.to_string(),
            parameters,
        };
        
        let payload = serde_json::to_string(&cmd)?;
        let topic = format!("devices/{}/commands", device_id);
        
        self.mqtt_client.publish(
            topic,
            QoS::AtMostOnce,
            false,
            payload,
        ).await?;
        
        println!("📤 Command sent to {}: {}", device_id, command);
        Ok(())
    }
    
    fn process_sensor_data(&mut self, data: SensorData) {
        let device = self.devices.entry(data.device_id.clone()).or_insert_with(|| {
            DeviceStatus {
                id: data.device_id.clone(),
                name: format!("Device {}", data.device_id),
                device_type: "Sensor".to_string(),
                online: true,
                last_seen: data.timestamp,
                sensors: Vec::new(),
            }
        });
        
        device.last_seen = data.timestamp;
        device.online = true;
        
        // Update or add sensor reading
        if let Some(sensor) = device.sensors.iter_mut().find(|s| s.sensor_type == data.sensor_type) {
            sensor.value = data.value;
            sensor.timestamp = data.timestamp;
        } else {
            device.sensors.push(SensorReading {
                sensor_type: data.sensor_type,
                value: data.value,
                unit: data.unit,
                timestamp: data.timestamp,
            });
        }
        
        // Check for alerts
        self.check_alerts(&data);
    }
    
    fn check_alerts(&self, data: &SensorData) {
        match data.sensor_type.as_str() {
            "temperature" => {
                if data.value > 30.0 {
                    println!("🚨 ALERT: High temperature detected: {}°C", data.value);
                } else if data.value < 10.0 {
                    println!("🚨 ALERT: Low temperature detected: {}°C", data.value);
                }
            }
            "humidity" => {
                if data.value > 70.0 {
                    println!("🚨 ALERT: High humidity detected: {}%", data.value);
                } else if data.value < 20.0 {
                    println!("🚨 ALERT: Low humidity detected: {}%", data.value);
                }
            }
            _ => {}
        }
    }
    
    fn display_dashboard(&self) {
        println!("\n📊 IoT Dashboard Status:");
        println!("========================");
        
        for (id, device) in &self.devices {
            let status = if device.online { "🟢 Online" } else { "🔴 Offline" };
            println!("Device: {} - {}", device.name, status);
            
            for sensor in &device.sensors {
                println!("  📈 {}: {:.1} {}", 
                    sensor.sensor_type, sensor.value, sensor.unit);
            }
            println!();
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dashboard = IoTDashboard::new();
    dashboard.start().await?;
    
    // Simulate dashboard operation
    tokio::time::sleep(Duration::from_secs(1)).await;
    dashboard.display_dashboard();
    
    // Send a command to a device
    let params = serde_json::json!({
        "target_temperature": 24.0
    });
    dashboard.send_command("living-room-thermostat", "set_temperature", params).await?;
    
    println!("✅ IoT Dashboard running successfully!");
    
    Ok(())
}`,
          language: 'rust'
        }
      }
    },
    {
      id: 'crypto-wallet',
      name: 'Cryptocurrency Wallet',
      description: 'Secure multi-currency wallet with DeFi integration',
      category: 'Blockchain',
      difficulty: 'Advanced',
      tags: ['Blockchain', 'Cryptocurrency', 'DeFi', 'Security'],
      icon: <DollarSign className="w-6 h-6 text-yellow-400" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Manage cryptocurrencies and interact with DeFi protocols',
      techStack: ['Rust', 'Web3', 'Ethereum', 'Bitcoin', 'Solana'],
      features: [
        'Multi-currency support',
        'Hardware wallet integration',
        'DeFi protocol interaction',
        'Transaction history',
        'Portfolio tracking'
      ],
      files: {
        'main.rs': {
          content: `use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wallet {
    pub id: String,
    pub name: String,
    pub accounts: HashMap<String, Account>,
    pub total_balance_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub currency: String,
    pub address: String,
    pub balance: f64,
    pub private_key_encrypted: String,
    pub transactions: Vec<Transaction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: String,
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub fee: f64,
    pub timestamp: u64,
    pub status: TransactionStatus,
    pub block_number: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,
    Confirmed,
    Failed,
}

pub struct CryptoWallet {
    wallets: HashMap<String, Wallet>,
    exchange_rates: HashMap<String, f64>,
}

impl CryptoWallet {
    pub fn new() -> Self {
        Self {
            wallets: HashMap::new(),
            exchange_rates: HashMap::new(),
        }
    }
    
    pub fn create_wallet(&mut self, name: String) -> Result<String, String> {
        let wallet_id = format!("wallet_{}", uuid::Uuid::new_v4());
        
        let mut accounts = HashMap::new();
        
        // Create Bitcoin account
        let btc_account = Account {
            currency: "BTC".to_string(),
            address: self.generate_btc_address()?,
            balance: 0.0,
            private_key_encrypted: self.encrypt_private_key(&self.generate_private_key())?,
            transactions: Vec::new(),
        };
        accounts.insert("BTC".to_string(), btc_account);
        
        // Create Ethereum account
        let eth_account = Account {
            currency: "ETH".to_string(),
            address: self.generate_eth_address()?,
            balance: 0.0,
            private_key_encrypted: self.encrypt_private_key(&self.generate_private_key())?,
            transactions: Vec::new(),
        };
        accounts.insert("ETH".to_string(), eth_account);
        
        // Create Solana account
        let sol_account = Account {
            currency: "SOL".to_string(),
            address: self.generate_sol_address()?,
            balance: 0.0,
            private_key_encrypted: self.encrypt_private_key(&self.generate_private_key())?,
            transactions: Vec::new(),
        };
        accounts.insert("SOL".to_string(), sol_account);
        
        let wallet = Wallet {
            id: wallet_id.clone(),
            name,
            accounts,
            total_balance_usd: 0.0,
        };
        
        self.wallets.insert(wallet_id.clone(), wallet);
        
        println!("✅ Wallet '{}' created successfully!", wallet_id);
        Ok(wallet_id)
    }
    
    pub async fn send_transaction(
        &mut self,
        wallet_id: &str,
        currency: &str,
        to_address: &str,
        amount: f64,
    ) -> Result<String, String> {
        let wallet = self.wallets.get_mut(wallet_id)
            .ok_or("Wallet not found")?;
        
        let account = wallet.accounts.get_mut(currency)
            .ok_or("Currency not supported in this wallet")?;
        
        if account.balance < amount {
            return Err("Insufficient balance".to_string());
        }
        
        // Calculate fee (simplified)
        let fee = match currency {
            "BTC" => 0.0001,
            "ETH" => 0.002,
            "SOL" => 0.000005,
            _ => 0.001,
        };
        
        if account.balance < amount + fee {
            return Err("Insufficient balance for transaction + fee".to_string());
        }
        
        // Create transaction
        let tx_hash = self.generate_transaction_hash();
        let transaction = Transaction {
            hash: tx_hash.clone(),
            from: account.address.clone(),
            to: to_address.to_string(),
            amount,
            fee,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            status: TransactionStatus::Pending,
            block_number: None,
        };
        
        // Update balance
        account.balance -= amount + fee;
        account.transactions.push(transaction);
        
        // Simulate blockchain submission
        self.submit_to_blockchain(currency, &tx_hash).await?;
        
        println!("📤 Transaction sent: {} {} to {}", amount, currency, to_address);
        println!("Transaction hash: {}", tx_hash);
        
        Ok(tx_hash)
    }
    
    pub async fn update_balances(&mut self) -> Result<(), String> {
        println!("🔄 Updating wallet balances...");
        
        for wallet in self.wallets.values_mut() {
            let mut total_usd = 0.0;
            
            for account in wallet.accounts.values_mut() {
                // Simulate balance fetching from blockchain
                let new_balance = self.fetch_balance_from_blockchain(&account.address, &account.currency).await?;
                account.balance = new_balance;
                
                // Convert to USD
                let rate = self.exchange_rates.get(&account.currency).unwrap_or(&0.0);
                total_usd += account.balance * rate;
                
                console.log(\`  ${account.currency} Balance: ${account.balance.toFixed(8)} ${account.currency} ($${(account.balance * rate).toFixed(2)})\`);
            
            wallet.total_balance_usd = total_usd;
        }
        
        Ok(())
    }
    
    pub async fn update_exchange_rates(&mut self) -> Result<(), String> {
        println!("💱 Updating exchange rates...");
        
        // Simulate fetching from price API
        self.exchange_rates.insert("BTC".to_string(), 45000.0);
        self.exchange_rates.insert("ETH".to_string(), 2800.0);
        self.exchange_rates.insert("SOL".to_string(), 95.0);
        
        println!("  BTC: $45,000");
        println!("  ETH: $2,800");
        println!("  SOL: $95");
        
        Ok(())
    }
    
    pub fn get_portfolio_summary(&self, wallet_id: &str) -> Result<(), String> {
        let wallet = self.wallets.get(wallet_id)
            .ok_or("Wallet not found")?;
        
        println!("\n💼 Portfolio Summary for '{}'", wallet.name);
        println!("=====================================");
        println!("Total Value: \\${:.2}", wallet.total_balance_usd);
        println!();
        
        for (currency, account) in &wallet.accounts {
            let rate = self.exchange_rates.get(currency).unwrap_or(&0.0);
            let usd_value = account.balance * rate;
            let percentage = if wallet.total_balance_usd > 0.0 {
                (usd_value / wallet.total_balance_usd) * 100.0
            } else {
                0.0
            };
            
            println!("{}: {:.8} (\\${:.2}) - {:.1}%", 
                currency, account.balance, usd_value, percentage);
            println!("  Address: {}", account.address);
            println!("  Transactions: {}", account.transactions.len());
            println!();
        }
        
        Ok(())
    }
    
    // Helper methods (simplified implementations)
    fn generate_btc_address(&self) -> Result<String, String> {
        Ok(format!("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa{}", rand::random::<u32>()))
    }
    
    fn generate_eth_address(&self) -> Result<String, String> {
        Ok(format!("0x742d35Cc6634C0532925a3b8D4{:x}", rand::random::<u64>()))
    }
    
    fn generate_sol_address(&self) -> Result<String, String> {
        Ok(format!("{}1111111QLbz7VNegYGGEUdRaLqmRpxMHJfvKdBZRd", 
            (0..32).map(|_| char::from(rand::random::<u8>() % 26 + b'A')).collect::<String>()))
    }
    
    fn generate_private_key(&self) -> String {
        format!("{:064x}", rand::random::<u64>())
    }
    
    fn encrypt_private_key(&self, private_key: &str) -> Result<String, String> {
        // In real implementation, use proper encryption
        Ok(format!("encrypted_{}", private_key))
    }
    
    fn generate_transaction_hash(&self) -> String {
        format!("0x{:064x}", rand::random::<u64>())
    }
    
    async fn submit_to_blockchain(&self, currency: &str, tx_hash: &str) -> Result<(), String> {
        // Simulate blockchain submission delay
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        println!("⛓️ Transaction {} submitted to {} blockchain", tx_hash, currency);
        Ok(())
    }
    
    async fn fetch_balance_from_blockchain(&self, address: &str, currency: &str) -> Result<f64, String> {
        // Simulate API call delay
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        
        // Return simulated balance
        match currency {
            "BTC" => Ok(0.05 + rand::random::<f64>() * 0.1),
            "ETH" => Ok(1.2 + rand::random::<f64>() * 2.0),
            "SOL" => Ok(10.0 + rand::random::<f64>() * 20.0),
            _ => Ok(0.0),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💰 Cryptocurrency Wallet Starting...");
    
    let mut wallet_manager = CryptoWallet::new();
    
    // Update exchange rates
    wallet_manager.update_exchange_rates().await?;
    
    // Create a new wallet
    let wallet_id = wallet_manager.create_wallet("My Crypto Wallet".to_string())?;
    
    // Update balances
    wallet_manager.update_balances().await?;
    
    // Show portfolio
    wallet_manager.get_portfolio_summary(&wallet_id)?;
    
    // Send a transaction
    match wallet_manager.send_transaction(&wallet_id, "ETH", "0x742d35Cc6634C0532925a3b8D4123456789", 0.1).await {
        Ok(tx_hash) => println!("✅ Transaction successful: {}", tx_hash),
        Err(e) => println!("❌ Transaction failed: {}", e),
    }
    
    println!("✅ Crypto wallet demo completed!");
    
    Ok(())
}`,
          language: 'rust'
        }
      }
    },
    {
      id: 'web-scraper',
      name: 'Intelligent Web Scraper',
      description: 'AI-powered web scraping with data extraction and analysis',
      category: 'Data',
      difficulty: 'Intermediate',
      tags: ['Web Scraping', 'Data Extraction', 'AI', 'Automation'],
      icon: <Globe className="w-6 h-6 text-green-400" />,
      estimatedTime: '2-3 weeks',
      useCase: 'Extract and analyze data from websites for market research and monitoring',
      techStack: ['Rust', 'Reqwest', 'Scraper', 'Tokio', 'Serde'],
      features: [
        'Intelligent content extraction',
        'Rate limiting and respect for robots.txt',
        'Data cleaning and normalization',
        'Export to multiple formats',
        'Scheduled scraping'
      ],
      files: {
        'main.rs': {
          content: `use reqwest;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{sleep, Duration};

#[derive(Debug, Serialize, Deserialize)]
pub struct ScrapedData {
    pub url: String,
    pub title: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: u64,
    pub links: Vec<String>,
    pub images: Vec<String>,
}

#[derive(Debug)]
pub struct WebScraper {
    client: reqwest::Client,
    rate_limit: Duration,
    user_agent: String,
}

impl WebScraper {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            client,
            rate_limit: Duration::from_millis(1000),
            user_agent: "Mozilla/5.0 (compatible; RustScraper/1.0)".to_string(),
        }
    }
    
    pub async fn scrape_url(&self, url: &str) -> Result<ScrapedData, Box<dyn std::error::Error>> {
        println!("🕷️ Scraping: {}", url);
        
        // Respect rate limiting
        sleep(self.rate_limit).await;
        
        let response = self.client
            .get(url)
            .header("User-Agent", &self.user_agent)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()).into());
        }
        
        let html_content = response.text().await?;
        let document = Html::parse_document(&html_content);
        
        // Extract title
        let title_selector = Selector::parse("title").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>())
            .unwrap_or_else(|| "No title found".to_string());
        
        // Extract main content
        let content = self.extract_main_content(&document);
        
        // Extract metadata
        let metadata = self.extract_metadata(&document);
        
        // Extract links
        let links = self.extract_links(&document, url);
        
        // Extract images
        let images = self.extract_images(&document, url);
        
        let scraped_data = ScrapedData {
            url: url.to_string(),
            title: title.trim().to_string(),
            content,
            metadata,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            links,
            images,
        };
        
        println!("✅ Successfully scraped: {}", url);
        Ok(scraped_data)
    }
    
    fn extract_main_content(&self, document: &Html) -> String {
        // Try multiple selectors for main content
        let content_selectors = [
            "article",
            "main",
            ".content",
            "#content",
            ".post-content",
            ".entry-content",
            "p",
        ];
        
        for selector_str in &content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                let content: String = document
                    .select(&selector)
                    .map(|el| el.text().collect::<String>())
                    .collect::<Vec<_>>()
                    .join(" ");
                
                if !content.trim().is_empty() && content.len() > 100 {
                    return self.clean_text(&content);
                }
            }
        }
        
        "No content extracted".to_string()
    }
    
    fn extract_metadata(&self, document: &Html) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        // Extract meta tags
        let meta_selector = Selector::parse("meta").unwrap();
        for element in document.select(&meta_selector) {
            if let Some(name) = element.value().attr("name") {
                if let Some(content) = element.value().attr("content") {
                    metadata.insert(name.to_string(), content.to_string());
                }
            }
            
            if let Some(property) = element.value().attr("property") {
                if let Some(content) = element.value().attr("content") {
                    metadata.insert(property.to_string(), content.to_string());
                }
            }
        }
        
        // Extract description
        if let Ok(desc_selector) = Selector::parse("meta[name='description']") {
            if let Some(desc_element) = document.select(&desc_selector).next() {
                if let Some(content) = desc_element.value().attr("content") {
                    metadata.insert("description".to_string(), content.to_string());
                }
            }
        }
        
        metadata
    }
    
    fn extract_links(&self, document: &Html, base_url: &str) -> Vec<String> {
        let link_selector = Selector::parse("a[href]").unwrap();
        let mut links = Vec::new();
        
        for element in document.select(&link_selector) {
            if let Some(href) = element.value().attr("href") {
                if let Ok(absolute_url) = self.resolve_url(base_url, href) {
                    links.push(absolute_url);
                }
            }
        }
        
        // Remove duplicates
        links.sort();
        links.dedup();
        links
    }
    
    fn extract_images(&self, document: &Html, base_url: &str) -> Vec<String> {
        let img_selector = Selector::parse("img[src]").unwrap();
        let mut images = Vec::new();
        
        for element in document.select(&img_selector) {
            if let Some(src) = element.value().attr("src") {
                if let Ok(absolute_url) = self.resolve_url(base_url, src) {
                    images.push(absolute_url);
                }
            }
        }
        
        images
    }
    
    fn resolve_url(&self, base: &str, relative: &str) -> Result<String, url::ParseError> {
        let base_url = url::Url::parse(base)?;
        let resolved = base_url.join(relative)?;
        Ok(resolved.to_string())
    }
    
    fn clean_text(&self, text: &str) -> String {
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string()
    }
    
    pub async fn scrape_multiple(&self, urls: Vec<&str>) -> Vec<Result<ScrapedData, Box<dyn std::error::Error>>> {
        let mut results = Vec::new();
        
        for url in urls {
            let result = self.scrape_url(url).await;
            results.push(result);
        }
        
        results
    }
    
    pub fn export_to_json(&self, data: &[ScrapedData], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(data)?;
        std::fs::write(filename, json)?;
        println!("📄 Data exported to {}", filename);
        Ok(())
    }
    
    pub fn export_to_csv(&self, data: &[ScrapedData], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut csv_content = String::new();
        csv_content.push_str("URL,Title,Content Length,Timestamp,Links Count,Images Count\\n");
        
        for item in data {
            csv_content.push_str(&format!(
                "\\"{}\\",\\"{}\\",{},{},{},{}\\n",
                item.url.replace('"', '""'),
                item.title.replace('"', '""'),
                item.content.len(),
                item.timestamp,
                item.links.len(),
                item.images.len()
            ));
        }
        
        std::fs::write(filename, csv_content)?;
        println!("📊 Data exported to {}", filename);
        Ok(())
    }
    
    pub fn analyze_data(&self, data: &[ScrapedData]) {
        println!("\\n📈 Data Analysis Report");
        println!("=======================");
        println!("Total pages scraped: {}", data.len());
        
        let total_content_length: usize = data.iter().map(|d| d.content.len()).sum();
        let avg_content_length = if !data.is_empty() {
            total_content_length / data.len()
        } else {
            0
        };
        
        let total_links: usize = data.iter().map(|d| d.links.len()).sum();
        let total_images: usize = data.iter().map(|d| d.images.len()).sum();
        
        println!("Average content length: {} characters", avg_content_length);
        println!("Total links found: {}", total_links);
        println!("Total images found: {}", total_images);
        
        // Find most common domains
        let mut domains = HashMap::new();
        for item in data {
            if let Ok(url) = url::Url::parse(&item.url) {
                if let Some(domain) = url.host_str() {
                    *domains.entry(domain.to_string()).or_insert(0) += 1;
                }
            }
        }
        
        println!("\\nDomains scraped:");
        for (domain, count) in domains {
            println!("  {}: {} pages", domain, count);
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🕷️ Intelligent Web Scraper Starting...");
    
    let scraper = WebScraper::new();
    
    // URLs to scrape (using example URLs)
    let urls = vec![
        "https://httpbin.org/html",
        "https://httpbin.org/json",
    ];
    
    println!("🎯 Scraping {} URLs...", urls.len());
    
    let results = scraper.scrape_multiple(urls).await;
    
    let mut successful_data = Vec::new();
    let mut failed_count = 0;
    
    for result in results {
        match result {
            Ok(data) => {
                println!("✅ Successfully scraped: {}", data.url);
                successful_data.push(data);
            }
            Err(e) => {
                println!("❌ Failed to scrape: {}", e);
                failed_count += 1;
            }
        }
    }
    
    if !successful_data.is_empty() {
        // Analyze the scraped data
        scraper.analyze_data(&successful_data);
        
        // Export data
        scraper.export_to_json(&successful_data, "scraped_data.json")?;
        scraper.export_to_csv(&successful_data, "scraped_data.csv")?;
    }
    
    println!("\\n📊 Scraping Summary:");
    println!("Successful: {}", successful_data.len());
    println!("Failed: {}", failed_count);
    
    println!("✅ Web scraping completed!");
    
    Ok(())
}`,
          language: 'rust'
        }
      }
    },
    {
      id: 'chat-application',
      name: 'Real-time Chat Application',
      description: 'Multi-platform chat app with end-to-end encryption',
      category: 'Social',
      difficulty: 'Intermediate',
      tags: ['Real-time', 'WebSocket', 'Encryption', 'Chat'],
      icon: <MessageSquare className="w-6 h-6 text-blue-400" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Build secure messaging applications for teams and communities',
      techStack: ['Rust', 'WebSocket', 'React', 'PostgreSQL', 'Redis'],
      features: [
        'Real-time messaging',
        'End-to-end encryption',
        'File sharing',
        'Group chats',
        'Message history'
      ],
      files: {
        'main.rs': {
          content: `use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub user_id: String,
    pub username: String,
    pub content: String,
    pub timestamp: u64,
    pub message_type: MessageType,
    pub room_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Text,
    Image,
    File,
    System,
}

#[derive(Debug, Clone)]
pub struct User {
    pub id: String,
    pub username: String,
    pub avatar: Option<String>,
    pub status: UserStatus,
}

#[derive(Debug, Clone)]
pub enum UserStatus {
    Online,
    Away,
    Busy,
    Offline,
}

#[derive(Debug, Clone)]
pub struct ChatRoom {
    pub id: String,
    pub name: String,
    pub description: String,
    pub members: Vec<String>,
    pub created_at: u64,
    pub is_private: bool,
}

pub struct ChatServer {
    rooms: Arc<RwLock<HashMap<String, ChatRoom>>>,
    users: Arc<RwLock<HashMap<String, User>>>,
    connections: Arc<RwLock<HashMap<String, broadcast::Sender<ChatMessage>>>>,
    message_history: Arc<RwLock<HashMap<String, Vec<ChatMessage>>>>,
}

impl ChatServer {
    pub fn new() -> Self {
        let mut rooms = HashMap::new();
        
        // Create default general room
        let general_room = ChatRoom {
            id: "general".to_string(),
            name: "General".to_string(),
            description: "General discussion room".to_string(),
            members: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            is_private: false,
        };
        rooms.insert("general".to_string(), general_room);
        
        Self {
            rooms: Arc::new(RwLock::new(rooms)),
            users: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(HashMap::new())),
            message_history: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn add_user(&self, username: String) -> String {
        let user_id = Uuid::new_v4().to_string();
        let user = User {
            id: user_id.clone(),
            username: username.clone(),
            avatar: None,
            status: UserStatus::Online,
        };
        
        self.users.write().await.insert(user_id.clone(), user);
        
        // Create broadcast channel for this user
        let (tx, _) = broadcast::channel(100);
        self.connections.write().await.insert(user_id.clone(), tx);
        
        // Add user to general room
        self.join_room(&user_id, "general").await.ok();
        
        // Send welcome message
        let welcome_message = ChatMessage {
            id: Uuid::new_v4().to_string(),
            user_id: "system".to_string(),
            username: "System".to_string(),
            content: format!("{} joined the chat!", username),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            message_type: MessageType::System,
            room_id: "general".to_string(),
        };
        
        self.broadcast_message("general", welcome_message).await.ok();
        
        println!("👤 User {} ({}) connected", username, user_id);
        user_id
    }
    
    pub async fn remove_user(&self, user_id: &str) {
        if let Some(user) = self.users.write().await.remove(user_id) {
            self.connections.write().await.remove(user_id);
            
            // Send goodbye message
            let goodbye_message = ChatMessage {
                id: Uuid::new_v4().to_string(),
                user_id: "system".to_string(),
                username: "System".to_string(),
                content: format!("{} left the chat", user.username),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                message_type: MessageType::System,
                room_id: "general".to_string(),
            };
            
            self.broadcast_message("general", goodbye_message).await.ok();
            println!("👤 User {} disconnected", user.username);
        }
    }
    
    pub async fn send_message(&self, message: ChatMessage) -> Result<(), String> {
        // Validate user exists
        if !self.users.read().await.contains_key(&message.user_id) {
            return Err("User not found".to_string());
        }
        
        // Validate room exists
        if !self.rooms.read().await.contains_key(&message.room_id) {
            return Err("Room not found".to_string());
        }
        
        // Store message in history
        self.message_history
            .write()
            .await
            .entry(message.room_id.clone())
            .or_insert_with(Vec::new)
            .push(message.clone());
        
        // Broadcast message to room
        self.broadcast_message(&message.room_id, message).await?;
        
        Ok(())
    }
    
    async fn broadcast_message(&self, room_id: &str, message: ChatMessage) -> Result<(), String> {
        let room = self.rooms.read().await;
        let room = room.get(room_id).ok_or("Room not found")?;
        
        let connections = self.connections.read().await;
        
        for member_id in &room.members {
            if let Some(sender) = connections.get(member_id) {
                if sender.send(message.clone()).is_err() {
                    println!("Failed to send message to user {}", member_id);
                }
            }
        }
        
        println!("📢 Message broadcast to room {}: {}", room_id, message.content);
        Ok(())
    }
    
    pub async fn join_room(&self, user_id: &str, room_id: &str) -> Result<(), String> {
        let mut rooms = self.rooms.write().await;
        let room = rooms.get_mut(room_id).ok_or("Room not found")?;
        
        if !room.members.contains(&user_id.to_string()) {
            room.members.push(user_id.to_string());
            
            if let Some(user) = self.users.read().await.get(user_id) {
                let join_message = ChatMessage {
                    id: Uuid::new_v4().to_string(),
                    user_id: "system".to_string(),
                    username: "System".to_string(),
                    content: format!("{} joined the room", user.username),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    message_type: MessageType::System,
                    room_id: room_id.to_string(),
                };
                
                drop(rooms); // Release the lock before broadcasting
                println!("{}: {:.8} (\\\\${:.2}) - {:.1}%", 
                self.broadcast_message(room_id, join_message).await.ok();
            }
        }
        
        Ok(())
    }
    
    pub async fn leave_room(&self, user_id: &str, room_id: &str) -> Result<(), String> {
        let mut rooms = self.rooms.write().await;
        let room = rooms.get_mut(room_id).ok_or("Room not found")?;
        
        room.members.retain(|id| id != user_id);
        
        if let Some(user) = self.users.read().await.get(user_id) {
            let leave_message = ChatMessage {
                id: Uuid::new_v4().to_string(),
                user_id: "system".to_string(),
                username: "System".to_string(),
                content: format!("{} left the room", user.username),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                message_type: MessageType::System,
                room_id: room_id.to_string(),
            };
            
            drop(rooms); // Release the lock before broadcasting
            self.broadcast_message(room_id, leave_message).await.ok();
        }
        
        Ok(())
    }
    
    pub async fn create_room(&self, name: String, description: String, creator_id: &str, is_private: bool) -> Result<String, String> {
        let room_id = Uuid::new_v4().to_string();
        let room = ChatRoom {
            id: room_id.clone(),
            name: name.clone(),
            description,
            members: vec![creator_id.to_string()],
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            is_private,
        };
        
        self.rooms.write().await.insert(room_id.clone(), room);
        
        if let Some(user) = self.users.read().await.get(creator_id) {
            let create_message = ChatMessage {
                id: Uuid::new_v4().to_string(),
                user_id: "system".to_string(),
                username: "System".to_string(),
                content: format!("{} created room '{}'", user.username, name),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                message_type: MessageType::System,
                room_id: room_id.clone(),
            };
            
            self.broadcast_message(&room_id, create_message).await.ok();
        }
        
        println!("🏠 Room '{}' created by user {}", name, creator_id);
        Ok(room_id)
    }
    
    pub async fn get_room_history(&self, room_id: &str, limit: usize) -> Vec<ChatMessage> {
        let history = self.message_history.read().await;
        if let Some(messages) = history.get(room_id) {
            let start = if messages.len() > limit {
                messages.len() - limit
            } else {
                0
            };
            messages[start..].to_vec()
        } else {
            Vec::new()
        }
    }
    
    pub async fn get_online_users(&self) -> Vec<User> {
        self.users
            .read()
            .await
            .values()
            .filter(|user| matches!(user.status, UserStatus::Online))
            .cloned()
            .collect()
    }
    
    pub async fn get_rooms(&self) -> Vec<ChatRoom> {
        self.rooms.read().await.values().cloned().collect()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💬 Real-time Chat Application Starting...");
    
    let chat_server = Arc::new(ChatServer::new());
    
    // Simulate some users and activity
    let user1_id = chat_server.add_user("Alice".to_string()).await;
    let user2_id = chat_server.add_user("Bob".to_string()).await;
    let user3_id = chat_server.add_user("Charlie".to_string()).await;
    
    // Create a private room
    let private_room_id = chat_server
        .create_room(
            "Development Team".to_string(),
            "Private room for development discussions".to_string(),
            &user1_id,
            true,
        )
        .await?;
    
    // Add users to private room
    chat_server.join_room(&user2_id, &private_room_id).await?;
    chat_server.join_room(&user3_id, &private_room_id).await?;
    
    // Send some messages
    let message1 = ChatMessage {
        id: Uuid::new_v4().to_string(),
        user_id: user1_id.clone(),
        username: "Alice".to_string(),
        content: "Hello everyone! 👋".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        message_type: MessageType::Text,
        room_id: "general".to_string(),
    };
    
    let message2 = ChatMessage {
        id: Uuid::new_v4().to_string(),
        user_id: user2_id.clone(),
        username: "Bob".to_string(),
        content: "Hey Alice! How's the project going?".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        message_type: MessageType::Text,
        room_id: "general".to_string(),
    };
    
    let message3 = ChatMessage {
        id: Uuid::new_v4().to_string(),
        user_id: user3_id.clone(),
        username: "Charlie".to_string(),
        content: "Let's discuss the new features in the dev room".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        message_type: MessageType::Text,
        room_id: private_room_id.clone(),
    };
    
    chat_server.send_message(message1).await?;
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    chat_server.send_message(message2).await?;
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    chat_server.send_message(message3).await?;
    
    // Display chat statistics
    println!("\\n📊 Chat Server Statistics:");
    println!("==========================");
    
    let online_users = chat_server.get_online_users().await;
    println!("Online users: {}", online_users.len());
    for user in online_users {
        println!("  - {}", user.username);
    }
    
    let rooms = chat_server.get_rooms().await;
    println!("\\nActive rooms: {}", rooms.len());
    for room in rooms {
        println!("  - {} ({} members)", room.name, room.members.len());
        
        let history = chat_server.get_room_history(&room.id, 5).await;
        if !history.is_empty() {
            println!("    Recent messages:");
            for msg in history.iter().take(3) {
                println!("      {}: {}", msg.username, msg.content);
            }
        }
    }
    
    println!("\\n✅ Chat application demo completed!");
    
    Ok(())
}`,
          language: 'rust'
        }
      }
    }
  ];

  const categories = ['all', 'AI/ML', 'Mobile', 'IoT', 'Blockchain', 'Data', 'Social'];

  const filteredTemplates = templates.filter(template => {
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
    const matchesSearch = searchQuery === '' || 
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    return matchesCategory && matchesSearch;
  });

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'text-green-400 bg-green-900/20';
      case 'Intermediate': return 'text-yellow-400 bg-yellow-900/20';
      case 'Advanced': return 'text-orange-400 bg-orange-900/20';
      case 'Expert': return 'text-red-400 bg-red-900/20';
      default: return 'text-gray-400 bg-gray-900/20';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'AI/ML': return <Brain className="w-5 h-5" />;
      case 'Mobile': return <Smartphone className="w-5 h-5" />;
      case 'IoT': return <Wifi className="w-5 h-5" />;
      case 'Blockchain': return <DollarSign className="w-5 h-5" />;
      case 'Data': return <Database className="w-5 h-5" />;
      case 'Social': return <MessageSquare className="w-5 h-5" />;
      default: return <Rocket className="w-5 h-5" />;
    }
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-lg max-w-7xl w-full h-[90vh] flex flex-col border border-gray-700">
        {/* Header */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-orange-600 rounded-lg">
                <Rocket className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Project Templates</h2>
                <p className="text-gray-400">Start with production-ready templates for AI, mobile, and more</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors text-xl"
            >
              ✕
            </button>
          </div>

          {/* Search and Filters */}
          <div className="flex items-center space-x-4">
            <div className="flex-1 relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search templates..."
                className="w-full pl-10 pr-4 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-orange-500 focus:outline-none"
              />
              <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
                <div className="w-4 h-4 text-gray-400" />
              </div>
            </div>
            
            <div className="flex space-x-2">
              {categories.map(category => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm transition-colors ${
                    selectedCategory === category
                      ? 'bg-orange-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {category !== 'all' && getCategoryIcon(category)}
                  <span>{category === 'all' ? 'All' : category}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Templates Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredTemplates.map(template => (
              <div
                key={template.id}
                className="bg-gray-700 rounded-lg p-6 border border-gray-600 hover:border-orange-500 transition-all duration-200 cursor-pointer group"
                onClick={() => onSelectTemplate(template)}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-gray-600 rounded-lg group-hover:bg-orange-600 transition-colors">
                      {template.icon}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white group-hover:text-orange-300 transition-colors">
                        {template.name}
                      </h3>
                      <div className="flex items-center space-x-2 mt-1">
                        <span className={`px-2 py-1 rounded text-xs ${getDifficultyColor(template.difficulty)}`}>
                          {template.difficulty}
                        </span>
                        <span className="text-gray-400 text-xs">{template.estimatedTime}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <p className="text-gray-300 text-sm mb-4 line-clamp-2">
                  {template.description}
                </p>

                <div className="mb-4">
                  <div className="text-xs text-gray-400 mb-2">Use Case:</div>
                  <div className="text-sm text-gray-200">{template.useCase}</div>
                </div>

                <div className="mb-4">
                  <div className="text-xs text-gray-400 mb-2">Tech Stack:</div>
                  <div className="flex flex-wrap gap-1">
                    {template.techStack.slice(0, 4).map(tech => (
                      <span key={tech} className="px-2 py-1 bg-gray-600 text-gray-200 rounded text-xs">
                        {tech}
                      </span>
                    ))}
                    {template.techStack.length > 4 && (
                      <span className="px-2 py-1 bg-gray-600 text-gray-400 rounded text-xs">
                        +{template.techStack.length - 4}
                      </span>
                    )}
                  </div>
                </div>

                <div className="mb-4">
                  <div className="text-xs text-gray-400 mb-2">Key Features:</div>
                  <ul className="text-sm text-gray-300 space-y-1">
                    {template.features.slice(0, 3).map((feature, index) => (
                      <li key={index} className="flex items-center space-x-2">
                        <div className="w-1 h-1 bg-orange-400 rounded-full" />
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="flex flex-wrap gap-1 mb-4">
                  {template.tags.slice(0, 3).map(tag => (
                    <span key={tag} className="px-2 py-1 bg-blue-900/30 text-blue-300 rounded text-xs">
                      {tag}
                    </span>
                  ))}
                </div>

                <button className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors group-hover:bg-orange-500">
                  <Rocket className="w-4 h-4" />
                  <span>Use Template</span>
                </button>
              </div>
            ))}
          </div>

          {filteredTemplates.length === 0 && (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">🔍</div>
              <h3 className="text-xl font-semibold text-white mb-2">No templates found</h3>
              <p className="text-gray-400">Try adjusting your search or category filter</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-700 bg-gray-900">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-400">
              {filteredTemplates.length} template{filteredTemplates.length !== 1 ? 's' : ''} available
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-400">
              <div className="flex items-center space-x-1">
                <Brain className="w-4 h-4 text-purple-400" />
                <span>AI/ML Ready</span>
              </div>
              <div className="flex items-center space-x-1">
                <Smartphone className="w-4 h-4 text-blue-400" />
                <span>Mobile First</span>
              </div>
              <div className="flex items-center space-x-1">
                <Shield className="w-4 h-4 text-green-400" />
                <span>Production Ready</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectTemplates;