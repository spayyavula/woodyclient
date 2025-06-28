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