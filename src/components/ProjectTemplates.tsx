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
            let mut total_loss =
        }
      }
    }
  ]
}