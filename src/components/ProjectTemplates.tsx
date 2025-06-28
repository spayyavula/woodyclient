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
    // HFT Templates
    {
      id: 'rust-hft-engine',
      name: 'High-Frequency Trading Engine',
      description: 'Ultra-low latency trading engine built with Rust for microsecond-level performance',
      category: 'HFT',
      difficulty: 'Expert',
      tags: ['HFT', 'Low Latency', 'Trading', 'Market Data', 'Rust'],
      icon: <TrendingUp className="w-6 h-6 text-green-400" />,
      estimatedTime: '6-8 weeks',
      useCase: 'Build ultra-fast trading systems for algorithmic trading and market making',
      techStack: ['Rust', 'DPDK', 'Kernel Bypass', 'FIX Protocol', 'Market Data'],
      features: [
        'Sub-microsecond latency',
        'Lock-free data structures',
        'Kernel bypass networking',
        'Real-time risk management',
        'Multi-venue connectivity'
      ],
      files: {
        'main.rs': {
          content: `use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time;

#[derive(Debug, Clone)]
pub struct Order {
    pub id: u64,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: u64,
    pub price: f64,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: u64,
    pub ask_size: u64,
    pub timestamp: Instant,
}

pub struct TradingEngine {
    orders_sent: AtomicU64,
    orders_filled: AtomicU64,
    orders_rejected: AtomicU64,
    total_pnl: AtomicU64, // Store as integer cents to avoid floating point
}

impl TradingEngine {
    pub fn new() -> Self {
        Self {
            orders_sent: AtomicU64::new(0),
            orders_filled: AtomicU64::new(0),
            orders_rejected: AtomicU64::new(0),
            total_pnl: AtomicU64::new(0),
        }
    }
    
    pub async fn process_market_data(&self, market_data: MarketData) {
        let start = Instant::now();
        
        // Ultra-fast market data processing
        let spread = market_data.ask_price - market_data.bid_price;
        
        // Simple market making strategy
        if spread > 0.01 {
            self.send_market_making_orders(&market_data).await;
        }
        
        let latency = start.elapsed();
        if latency > Duration::from_nanos(500) {
            println!("⚠️ High latency detected: {:?}", latency);
        }
    }
    
    async fn send_market_making_orders(&self, market_data: &MarketData) {
        let order_id = self.orders_sent.fetch_add(1, Ordering::Relaxed);
        
        // Send buy order slightly below bid
        let buy_order = Order {
            id: order_id,
            symbol: market_data.symbol.clone(),
            side: OrderSide::Buy,
            quantity: 100,
            price: market_data.bid_price - 0.01,
            timestamp: Instant::now(),
        };
        
        // Send sell order slightly above ask
        let sell_order = Order {
            id: order_id + 1,
            symbol: market_data.symbol.clone(),
            side: OrderSide::Sell,
            quantity: 100,
            price: market_data.ask_price + 0.01,
            timestamp: Instant::now(),
        };
        
        // Simulate order sending (in real implementation, this would use
        // kernel bypass networking for minimal latency)
        self.send_order_to_exchange(buy_order).await;
        self.send_order_to_exchange(sell_order).await;
        
        self.orders_sent.fetch_add(2, Ordering::Relaxed);
    }
    
    async fn send_order_to_exchange(&self, order: Order) {
        // In production, this would use:
        // - DPDK for kernel bypass
        // - Custom FIX protocol implementation
        // - Direct market access (DMA)
        // - Hardware timestamping
        
        println!("📤 Sending order: {:?}", order);
        
        // Simulate network latency (in real HFT, this would be <10 microseconds)
        time::sleep(Duration::from_micros(5)).await;
    }
    
    pub fn get_performance_stats(&self) -> (u64, u64, u64, f64) {
        let sent = self.orders_sent.load(Ordering::Relaxed);
        let filled = self.orders_filled.load(Ordering::Relaxed);
        let rejected = self.orders_rejected.load(Ordering::Relaxed);
        let pnl = self.total_pnl.load(Ordering::Relaxed) as f64 / 100.0; // Convert cents to dollars
        
        (sent, filled, rejected, pnl)
    }
}

#[tokio::main]
async fn main() {
    println!("🚀 High-Frequency Trading Engine");
    println!("=================================");
    
    let engine = Arc::new(TradingEngine::new());
    let engine_clone = Arc::clone(&engine);
    
    // Simulate market data feed
    let market_data_task = tokio::spawn(async move {
        let mut counter = 0;
        loop {
            let market_data = MarketData {
                symbol: "AAPL".to_string(),
                bid_price: 150.00 + (counter as f64 * 0.01),
                ask_price: 150.01 + (counter as f64 * 0.01),
                bid_size: 1000,
                ask_size: 1000,
                timestamp: Instant::now(),
            };
            
            engine.process_market_data(market_data).await;
            
            // High-frequency updates (1000 times per second)
            time::sleep(Duration::from_millis(1)).await;
            counter += 1;
            
            if counter > 10000 {
                break;
            }
        }
    });
    
    // Performance monitoring task
    let monitoring_task = tokio::spawn(async move {
        loop {
            time::sleep(Duration::from_secs(5)).await;
            let (sent, filled, rejected, pnl) = engine_clone.get_performance_stats();
            println!("📊 Stats: {} sent, {} filled, {} rejected, PnL: \${:.2}",
                    sent, filled, rejected, pnl);
        }
    });
    
    // Wait for tasks to complete
    let _ = tokio::join!(market_data_task, monitoring_task);
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
serde = { version = "1.0", features = ["derive"] }
crossbeam = "0.8"
lockfree = "0.5"
dpdk-sys = "0.1"  # For kernel bypass networking
fix-rs = "0.1"    # FIX protocol implementation

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
      id: 'market-data-processor',
      name: 'Real-time Market Data Processor',
      description: 'High-throughput market data processing system with microsecond latency',
      category: 'HFT',
      difficulty: 'Advanced',
      tags: ['Market Data', 'Real-time', 'Low Latency', 'Processing', 'Rust'],
      icon: <BarChart3 className="w-6 h-6 text-blue-400" />,
      estimatedTime: '4-5 weeks',
      useCase: 'Process millions of market data messages per second for trading algorithms',
      techStack: ['Rust', 'Lock-free', 'SIMD', 'Memory Mapping', 'Binary Protocols'],
      features: [
        'Lock-free message processing',
        'SIMD optimizations',
        'Zero-copy deserialization',
        'Real-time analytics',
        'Multi-exchange support'
      ],
      files: {
        'main.rs': {
          content: `use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use crossbeam::channel::{bounded, Receiver, Sender};
use lockfree::queue::Queue;

#[derive(Debug, Clone)]
pub struct Tick {
    pub symbol: u32,        // Symbol ID for faster processing
    pub price: u64,         // Price in fixed-point (avoid floating point)
    pub size: u32,
    pub timestamp: u64,     // Nanoseconds since epoch
    pub exchange_id: u8,
    pub message_type: MessageType,
}

#[derive(Debug, Clone)]
pub enum MessageType {
    Trade,
    Quote,
    BookUpdate,
}

pub struct MarketDataProcessor {
    tick_queue: Arc<Queue<Tick>>,
    processed_count: AtomicU64,
    total_latency: AtomicU64,
    max_latency: AtomicU64,
}

impl MarketDataProcessor {
    pub fn new() -> Self {
        Self {
            tick_queue: Arc::new(Queue::new()),
            processed_count: AtomicU64::new(0),
            total_latency: AtomicU64::new(0),
            max_latency: AtomicU64::new(0),
        }
    }
    
    pub fn process_tick(&self, tick: Tick) {
        let start = Instant::now();
        
        // Ultra-fast tick processing
        self.update_order_book(&tick);
        self.calculate_indicators(&tick);
        self.check_trading_signals(&tick);
        
        let latency = start.elapsed().as_nanos() as u64;
        self.update_latency_stats(latency);
        
        self.processed_count.fetch_add(1, Ordering::Relaxed);
    }
    
    fn update_order_book(&self, tick: &Tick) {
        // Lock-free order book update
        // In production, this would use custom lock-free data structures
        match tick.message_type {
            MessageType::Quote => {
                // Update bid/ask
            },
            MessageType::Trade => {
                // Update last trade price
            },
            MessageType::BookUpdate => {
                // Update order book levels
            },
        }
    }
    
    fn calculate_indicators(&self, tick: &Tick) {
        // SIMD-optimized technical indicators
        // - Moving averages
        // - VWAP
        // - Momentum indicators
        // - Volatility measures
    }
    
    fn check_trading_signals(&self, tick: &Tick) {
        // Ultra-fast signal detection
        let price = tick.price as f64 / 10000.0; // Convert from fixed-point
        
        // Example: Simple momentum signal
        if price > 150.0 {
            // Signal detected - would trigger trading logic
        }
    }
    
    fn update_latency_stats(&self, latency_ns: u64) {
        self.total_latency.fetch_add(latency_ns, Ordering::Relaxed);
        
        // Update max latency using compare-and-swap
        let mut current_max = self.max_latency.load(Ordering::Relaxed);
        while latency_ns > current_max {
            match self.max_latency.compare_exchange_weak(
                current_max,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }
    
    pub fn get_stats(&self) -> ProcessingStats {
        let count = self.processed_count.load(Ordering::Relaxed);
        let total_latency = self.total_latency.load(Ordering::Relaxed);
        let max_latency = self.max_latency.load(Ordering::Relaxed);
        
        ProcessingStats {
            processed_count: count,
            avg_latency_ns: if count > 0 { total_latency / count } else { 0 },
            max_latency_ns: max_latency,
        }
    }
}

#[derive(Debug)]
pub struct ProcessingStats {
    pub processed_count: u64,
    pub avg_latency_ns: u64,
    pub max_latency_ns: u64,
}

pub struct MarketDataFeed {
    sender: Sender<Tick>,
}

impl MarketDataFeed {
    pub fn new(sender: Sender<Tick>) -> Self {
        Self { sender }
    }
    
    pub async fn start_feed(&self) {
        let mut counter = 0u32;
        
        loop {
            // Simulate high-frequency market data
            let tick = Tick {
                symbol: 1, // AAPL
                price: 1500000 + (counter % 1000) as u64, // $150.00 + small variations
                size: 100 + (counter % 500),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                exchange_id: 1, // NYSE
                message_type: if counter % 3 == 0 {
                    MessageType::Trade
                } else {
                    MessageType::Quote
                },
            };
            
            if self.sender.try_send(tick).is_err() {
                // Channel full - in production, this would trigger backpressure handling
                break;
            }
            
            counter += 1;
            
            // Simulate 1 million messages per second
            if counter % 1000 == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            
            if counter > 1_000_000 {
                break;
            }
        }
    }
}

#[tokio::main]
async fn main() {
    println!("📊 Real-time Market Data Processor");
    println!("==================================");
    
    let processor = Arc::new(MarketDataProcessor::new());
    let (sender, receiver) = bounded(100_000);
    
    // Start market data feed
    let feed = MarketDataFeed::new(sender);
    let feed_task = tokio::spawn(async move {
        feed.start_feed().await;
    });
    
    // Start processing task
    let processor_clone = Arc::clone(&processor);
    let processing_task = tokio::spawn(async move {
        while let Ok(tick) = receiver.recv() {
            processor_clone.process_tick(tick);
        }
    });
    
    // Statistics monitoring
    let stats_processor = Arc::clone(&processor);
    let stats_task = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let stats = stats_processor.get_stats();
            println!(
                "📈 Processed: {} ticks, Avg Latency: {}ns, Max Latency: {}ns",
                stats.processed_count,
                stats.avg_latency_ns,
                stats.max_latency_ns
            );
        }
    });
    
    // Wait for completion
    let _ = tokio::join!(feed_task, processing_task, stats_task);
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "market-data-processor"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
crossbeam = "0.8"
lockfree = "0.5"
serde = { version = "1.0", features = ["derive"] }
byteorder = "1.4"

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
      id: 'algorithmic-trading-system',
      name: 'Algorithmic Trading System',
      description: 'Complete algorithmic trading platform with strategy backtesting and live execution',
      category: 'HFT',
      difficulty: 'Expert',
      tags: ['Algorithmic Trading', 'Backtesting', 'Strategies', 'Risk Management', 'Rust'],
      icon: <Bot className="w-6 h-6 text-purple-400" />,
      estimatedTime: '8-10 weeks',
      useCase: 'Build sophisticated trading algorithms with backtesting and risk management',
      techStack: ['Rust', 'Time Series', 'Statistics', 'Machine Learning', 'Risk Management'],
      features: [
        'Strategy backtesting engine',
        'Real-time strategy execution',
        'Risk management system',
        'Performance analytics',
        'Multi-asset support'
      ],
      files: {
        'main.rs': {
          content: `use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: i64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: u64,
    pub price: f64,
    pub timestamp: Instant,
    pub strategy_id: String,
}

#[derive(Debug, Clone)]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: u64,
    pub timestamp: Instant,
}

pub trait TradingStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn on_market_data(&mut self, data: &MarketData) -> Vec<Trade>;
    fn on_trade_fill(&mut self, trade: &Trade);
    fn get_positions(&self) -> &HashMap<String, Position>;
}

pub struct MomentumStrategy {
    name: String,
    positions: HashMap<String, Position>,
    price_history: HashMap<String, Vec<f64>>,
    lookback_period: usize,
    threshold: f64,
}

impl MomentumStrategy {
    pub fn new(lookback_period: usize, threshold: f64) -> Self {
        Self {
            name: "Momentum Strategy".to_string(),
            positions: HashMap::new(),
            price_history: HashMap::new(),
            lookback_period,
            threshold,
        }
    }
    
    fn calculate_momentum(&self, symbol: &str) -> Option<f64> {
        let prices = self.price_history.get(symbol)?;
        if prices.len() < self.lookback_period {
            return None;
        }
        
        let recent_prices = &prices[prices.len() - self.lookback_period..];
        let current_price = *recent_prices.last()?;
        let old_price = *recent_prices.first()?;
        
        Some((current_price - old_price) / old_price)
    }
}

impl TradingStrategy for MomentumStrategy {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn on_market_data(&mut self, data: &MarketData) -> Vec<Trade> {
        let mut trades = Vec::new();
        
        // Update price history
        let prices = self.price_history.entry(data.symbol.clone()).or_insert_with(Vec::new);
        prices.push(data.price);
        
        // Keep only recent prices
        if prices.len() > self.lookback_period * 2 {
            prices.drain(0..self.lookback_period);
        }
        
        // Calculate momentum signal
        if let Some(momentum) = self.calculate_momentum(&data.symbol) {
            let current_position = self.positions.get(&data.symbol)
                .map(|p| p.quantity)
                .unwrap_or(0);
            
            // Generate trading signals
            if momentum > self.threshold && current_position <= 0 {
                // Strong upward momentum - buy signal
                trades.push(Trade {
                    symbol: data.symbol.clone(),
                    side: TradeSide::Buy,
                    quantity: 100,
                    price: data.price,
                    timestamp: data.timestamp,
                    strategy_id: self.name.clone(),
                });
            } else if momentum < -self.threshold && current_position >= 0 {
                // Strong downward momentum - sell signal
                trades.push(Trade {
                    symbol: data.symbol.clone(),
                    side: TradeSide::Sell,
                    quantity: 100,
                    price: data.price,
                    timestamp: data.timestamp,
                    strategy_id: self.name.clone(),
                });
            }
        }
        
        trades
    }
    
    fn on_trade_fill(&mut self, trade: &Trade) {
        let position = self.positions.entry(trade.symbol.clone()).or_insert(Position {
            symbol: trade.symbol.clone(),
            quantity: 0,
            avg_price: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        });
        
        match trade.side {
            TradeSide::Buy => {
                position.quantity += trade.quantity as i64;
                position.avg_price = (position.avg_price * (position.quantity - trade.quantity as i64) as f64 
                    + trade.price * trade.quantity as f64) / position.quantity as f64;
            },
            TradeSide::Sell => {
                position.quantity -= trade.quantity as i64;
                // Calculate realized PnL
                position.realized_pnl += (trade.price - position.avg_price) * trade.quantity as f64;
            },
        }
    }
    
    fn get_positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }
}

pub struct TradingEngine {
    strategies: Vec<Box<dyn TradingStrategy>>,
    risk_manager: RiskManager,
    total_pnl: f64,
    trade_count: u64,
}

impl TradingEngine {
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            risk_manager: RiskManager::new(),
            total_pnl: 0.0,
            trade_count: 0,
        }
    }
    
    pub fn add_strategy(&mut self, strategy: Box<dyn TradingStrategy>) {
        self.strategies.push(strategy);
    }
    
    pub fn process_market_data(&mut self, data: MarketData) {
        for strategy in &mut self.strategies {
            let trades = strategy.on_market_data(&data);
            
            for trade in trades {
                if self.risk_manager.validate_trade(&trade) {
                    self.execute_trade(trade, strategy.as_mut());
                }
            }
        }
    }
    
    fn execute_trade(&mut self, trade: Trade, strategy: &mut dyn TradingStrategy) {
        println!("🔄 Executing trade: {:?}", trade);
        
        // Simulate trade execution
        strategy.on_trade_fill(&trade);
        self.trade_count += 1;
        
        // Update total PnL (simplified)
        match trade.side {
            TradeSide::Buy => self.total_pnl -= trade.price * trade.quantity as f64,
            TradeSide::Sell => self.total_pnl += trade.price * trade.quantity as f64,
        }
    }
    
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            total_pnl: self.total_pnl,
            trade_count: self.trade_count,
            strategy_count: self.strategies.len(),
        }
    }
}

pub struct RiskManager {
    max_position_size: u64,
    max_daily_loss: f64,
    current_daily_loss: f64,
}

impl RiskManager {
    pub fn new() -> Self {
        Self {
            max_position_size: 1000,
            max_daily_loss: 10000.0,
            current_daily_loss: 0.0,
        }
    }
    
    pub fn validate_trade(&self, trade: &Trade) -> bool {
        // Position size check
        if trade.quantity > self.max_position_size {
            println!("❌ Trade rejected: Position size too large");
            return false;
        }
        
        // Daily loss limit check
        if self.current_daily_loss > self.max_daily_loss {
            println!("❌ Trade rejected: Daily loss limit exceeded");
            return false;
        }
        
        true
    }
}

#[derive(Debug)]
pub struct PerformanceSummary {
    pub total_pnl: f64,
    pub trade_count: u64,
    pub strategy_count: usize,
}

#[tokio::main]
async fn main() {
    println!("🤖 Algorithmic Trading System");
    println!("=============================");
    
    let mut engine = TradingEngine::new();
    
    // Add momentum strategy
    let momentum_strategy = Box::new(MomentumStrategy::new(20, 0.02));
    engine.add_strategy(momentum_strategy);
    
    // Simulate market data feed
    let mut price = 150.0;
    for i in 0..1000 {
        // Simulate price movement
        price += (rand::random::<f64>() - 0.5) * 2.0;
        
        let market_data = MarketData {
            symbol: "AAPL".to_string(),
            price,
            volume: 1000,
            timestamp: Instant::now(),
        };
        
        engine.process_market_data(market_data);
        
        // Print performance every 100 iterations
        if i % 100 == 0 {
            let summary = engine.get_performance_summary();
            println!("📊 Performance: PnL: \${:.2}, Trades: {}", 
                    summary.total_pnl, summary.trade_count);
        }
        
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    let final_summary = engine.get_performance_summary();
    println!("🏁 Final Performance: PnL: \${:.2}, Total Trades: {}", 
            final_summary.total_pnl, final_summary.trade_count);
}

// Simple random number generation for demo
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    static SEED: AtomicU64 = AtomicU64::new(1);
    
    pub fn random<T>() -> f64 {
        let seed = SEED.fetch_add(1, Ordering::Relaxed);
        ((seed.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31)) as f64 / (1 << 31) as f64
    }
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "algorithmic-trading-system"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.0", features = ["v4"] }

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
      id: 'risk-management-system',
      name: 'Real-time Risk Management',
      description: 'Advanced risk management system for trading operations with real-time monitoring',
      category: 'HFT',
      difficulty: 'Advanced',
      tags: ['Risk Management', 'Real-time', 'Monitoring', 'Compliance', 'Rust'],
      icon: <Shield className="w-6 h-6 text-red-400" />,
      estimatedTime: '5-6 weeks',
      useCase: 'Monitor and control trading risks in real-time with automated safeguards',
      techStack: ['Rust', 'Real-time Systems', 'Statistics', 'Monitoring', 'Alerting'],
      features: [
        'Real-time position monitoring',
        'Automated risk limits',
        'VaR calculations',
        'Stress testing',
        'Compliance reporting'
      ],
      files: {
        'main.rs': {
          content: `use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position_value: f64,
    pub max_daily_loss: f64,
    pub max_leverage: f64,
    pub max_concentration: f64, // Max % of portfolio in single asset
    pub var_limit: f64,         // Value at Risk limit
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: i64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
}

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub total_portfolio_value: f64,
    pub total_pnl: f64,
    pub var_1d: f64,
    pub var_10d: f64,
    pub max_drawdown: f64,
    pub leverage: f64,
    pub largest_position_pct: f64,
}

#[derive(Debug, Clone)]
pub enum RiskAlert {
    PositionLimitExceeded { symbol: String, current: f64, limit: f64 },
    DailyLossLimitExceeded { current: f64, limit: f64 },
    VarLimitExceeded { current: f64, limit: f64 },
    ConcentrationRiskHigh { symbol: String, percentage: f64 },
    LeverageExceeded { current: f64, limit: f64 },
}

pub struct RiskManager {
    limits: RiskLimits,
    positions: Arc<RwLock<HashMap<String, Position>>>,
    price_history: HashMap<String, Vec<f64>>,
    portfolio_history: Vec<f64>,
    alerts: Vec<RiskAlert>,
    start_of_day_value: f64,
}

impl RiskManager {
    pub fn new(limits: RiskLimits) -> Self {
        Self {
            limits,
            positions: Arc::new(RwLock::new(HashMap::new())),
            price_history: HashMap::new(),
            portfolio_history: Vec::new(),
            alerts: Vec::new(),
            start_of_day_value: 1_000_000.0, // $1M starting portfolio
        }
    }
    
    pub fn update_position(&mut self, symbol: String, position: Position) {
        let mut positions = self.positions.write().unwrap();
        positions.insert(symbol, position);
    }
    
    pub fn update_market_data(&mut self, symbol: String, price: f64) {
        // Update price history for VaR calculations
        let prices = self.price_history.entry(symbol.clone()).or_insert_with(Vec::new);
        prices.push(price);
        
        // Keep only last 252 days (1 year) for calculations
        if prices.len() > 252 {
            prices.remove(0);
        }
        
        // Update position market values
        if let Ok(mut positions) = self.positions.write() {
            if let Some(position) = positions.get_mut(&symbol) {
                position.market_value = position.quantity as f64 * price;
                // Update unrealized PnL (simplified)
                position.unrealized_pnl = position.market_value - (position.quantity as f64 * 150.0); // Assume $150 cost basis
            }
        }
    }
    
    pub fn calculate_risk_metrics(&mut self) -> RiskMetrics {
        let positions = self.positions.read().unwrap();
        
        let total_portfolio_value: f64 = positions.values()
            .map(|p| p.market_value)
            .sum();
        
        let total_pnl: f64 = positions.values()
            .map(|p| p.unrealized_pnl)
            .sum();
        
        // Calculate Value at Risk (simplified historical simulation)
        let var_1d = self.calculate_var(1);
        let var_10d = self.calculate_var(10);
        
        // Calculate maximum drawdown
        self.portfolio_history.push(total_portfolio_value);
        let max_drawdown = self.calculate_max_drawdown();
        
        // Calculate leverage
        let leverage = total_portfolio_value / self.start_of_day_value;
        
        // Find largest position percentage
        let largest_position_pct = positions.values()
            .map(|p| p.market_value.abs() / total_portfolio_value)
            .fold(0.0, f64::max);
        
        RiskMetrics {
            total_portfolio_value,
            total_pnl,
            var_1d,
            var_10d,
            max_drawdown,
            leverage,
            largest_position_pct,
        }
    }
    
    fn calculate_var(&self, days: usize) -> f64 {
        // Simplified VaR calculation using historical simulation
        if self.portfolio_history.len() < days + 1 {
            return 0.0;
        }
        
        let mut returns = Vec::new();
        for i in days..self.portfolio_history.len() {
            let ret = (self.portfolio_history[i] - self.portfolio_history[i - days]) / self.portfolio_history[i - days];
            returns.push(ret);
        }
        
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // 95% VaR (5th percentile)
        let index = (returns.len() as f64 * 0.05) as usize;
        if index < returns.len() {
            -returns[index] * self.start_of_day_value
        } else {
            0.0
        }
    }
    
    fn calculate_max_drawdown(&self) -> f64 {
        if self.portfolio_history.len() < 2 {
            return 0.0;
        }
        
        let mut max_value = self.portfolio_history[0];
        let mut max_drawdown = 0.0;
        
        for &value in &self.portfolio_history[1..] {
            if value > max_value {
                max_value = value;
            } else {
                let drawdown = (max_value - value) / max_value;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }
        }
        
        max_drawdown
    }
    
    pub fn check_risk_limits(&mut self, metrics: &RiskMetrics) -> Vec<RiskAlert> {
        let mut alerts = Vec::new();
        
        // Check daily loss limit
        let daily_pnl = metrics.total_portfolio_value - self.start_of_day_value;
        if daily_pnl < -self.limits.max_daily_loss {
            alerts.push(RiskAlert::DailyLossLimitExceeded {
                current: -daily_pnl,
                limit: self.limits.max_daily_loss,
            });
        }
        
        // Check VaR limit
        if metrics.var_1d > self.limits.var_limit {
            alerts.push(RiskAlert::VarLimitExceeded {
                current: metrics.var_1d,
                limit: self.limits.var_limit,
            });
        }
        
        // Check leverage limit
        if metrics.leverage > self.limits.max_leverage {
            alerts.push(RiskAlert::LeverageExceeded {
                current: metrics.leverage,
                limit: self.limits.max_leverage,
            });
        }
        
        // Check concentration risk
        if metrics.largest_position_pct > self.limits.max_concentration {
            alerts.push(RiskAlert::ConcentrationRiskHigh {
                symbol: "LARGEST_POSITION".to_string(),
                percentage: metrics.largest_position_pct * 100.0,
            });
        }
        
        // Check individual position limits
        let positions = self.positions.read().unwrap();
        for (symbol, position) in positions.iter() {
            if position.market_value.abs() > self.limits.max_position_value {
                alerts.push(RiskAlert::PositionLimitExceeded {
                    symbol: symbol.clone(),
                    current: position.market_value.abs(),
                    limit: self.limits.max_position_value,
                });
            }
        }
        
        self.alerts.extend(alerts.clone());
        alerts
    }
    
    pub fn generate_risk_report(&self, metrics: &RiskMetrics) -> String {
        format!(
            "📊 RISK MANAGEMENT REPORT
============================
Portfolio Value: ${:.2}
Total P&L: ${:.2}
1-Day VaR (95%): ${:.2}
10-Day VaR (95%): ${:.2}
Max Drawdown: {:.2}%
Leverage: {:.2}x
Largest Position: {:.1}%

🚨 Active Alerts: {}
",
            metrics.total_portfolio_value,
            metrics.total_pnl,
            metrics.var_1d,
            metrics.var_10d,
            metrics.max_drawdown * 100.0,
            metrics.leverage,
            metrics.largest_position_pct * 100.0,
            self.alerts.len()
        )
    }
}

#[tokio::main]
async fn main() {
    println!("🛡️ Real-time Risk Management System");
    println!("===================================");
    
    let limits = RiskLimits {
        max_position_value: 100_000.0,
        max_daily_loss: 50_000.0,
        max_leverage: 3.0,
        max_concentration: 0.2, // 20%
        var_limit: 25_000.0,
    };
    
    let mut risk_manager = RiskManager::new(limits);
    
    // Simulate trading activity
    for i in 0..100 {
        // Update market data
        let price = 150.0 + (i as f64 * 0.1) + (rand::random::<f64>() - 0.5) * 10.0;
        risk_manager.update_market_data("AAPL".to_string(), price);
        
        // Update position
        let position = Position {
            symbol: "AAPL".to_string(),
            quantity: 1000 + (i * 10) as i64,
            market_value: (1000 + (i * 10)) as f64 * price,
            unrealized_pnl: 0.0,
            delta: 1.0,
            gamma: 0.0,
            vega: 0.0,
        };
        risk_manager.update_position("AAPL".to_string(), position);
        
        // Calculate risk metrics
        let metrics = risk_manager.calculate_risk_metrics();
        
        // Check for risk limit violations
        let alerts = risk_manager.check_risk_limits(&metrics);
        
        // Print alerts
        for alert in alerts {
            println!("🚨 RISK ALERT: {:?}", alert);
        }
        
        // Print periodic risk report
        if i % 20 == 0 {
            let report = risk_manager.generate_risk_report(&metrics);
            println!("{}", report);
        }
        
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// Simple random number generation for demo
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    static SEED: AtomicU64 = AtomicU64::new(1);
    
    pub fn random<T>() -> f64 {
        let seed = SEED.fetch_add(1, Ordering::Relaxed);
        ((seed.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31)) as f64 / (1 << 31) as f64
    }
}`,
          language: 'rust',
        },
        'Cargo.toml': {
          content: `[package]
name = "risk-management-system"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3`,
          language: 'toml',
        }
      }
    },
    // Mobile Development Templates
    {
      id: 'flutter-rust-mobile',
      name: 'Flutter + Rust Mobile App',
      description: 'Cross-platform mobile app with Rust backend for performance-critical operations',
      category: 'Mobile',
      difficulty: 'Advanced',
      tags: ['Flutter', 'Rust', 'FFI', 'Cross-platform', 'Mobile'],
      icon: <Smartphone className="w-6 h-6 text-blue-400" />,
      estimatedTime: '3-4 weeks',
      useCase: 'Build high-performance mobile apps with Rust for compute-intensive tasks',
      techStack: ['Flutter', 'Rust', 'FFI', 'Dart'],
      features: [
        'Cross-platform deployment',
        'Native performance',
        'Rust FFI integration',
        'State management',
        'Platform channels'
      ],
      files: {
        'lib/main.dart': {
          content: `import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ffi';
import 'dart:io';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter + Rust Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(title: 'Flutter + Rust Integration'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key? key, required this.title}) : super(key: key);
  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  static const platform = MethodChannel('rust_integration');
  String _result = 'No calculation yet';

  Future<void> _performHeavyCalculation() async {
    try {
      final int result = await platform.invokeMethod('heavyCalculation', {'input': 1000000});
      setState(() {
        _result = 'Rust calculation result: $result';
      });
    } on PlatformException catch (e) {
      setState(() {
        _result = 'Error: ${e.message}';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'Tap the button to run Rust computation:',
            ),
            Text(
              _result,
              style: Theme.of(context).textTheme.headlineMedium,
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _performHeavyCalculation,
        tooltip: 'Calculate',
        child: Icon(Icons.calculate),
      ),
    );
  }
}`,
          language: 'dart',
        },
        'rust/src/lib.rs': {
          content: `use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn heavy_calculation(input: i64) -> i64 {
    // Simulate heavy computation
    let mut result = 0;
    for i in 0..input {
        result += fibonacci(i % 40);
    }
    result
}

fn fibonacci(n: i64) -> i64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[no_mangle]
pub extern "C" fn process_string(input: *const c_char) -> *mut c_char {
    let c_str = unsafe { CStr::from_ptr(input) };
    let input_str = c_str.to_str().unwrap();
    
    let processed = format!("Processed by Rust: {}", input_str.to_uppercase());
    let c_string = CString::new(processed).unwrap();
    c_string.into_raw()
}

#[no_mangle]
pub extern "C" fn free_string(ptr: *mut c_char) {
    unsafe {
        if ptr.is_null() {
            return;
        }
        CString::from_raw(ptr)
    };
}`,
          language: 'rust',
        }
      }
    }
  ];

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