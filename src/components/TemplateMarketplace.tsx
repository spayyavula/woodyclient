import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  Smartphone, 
  Zap, 
  Star, 
  Download, 
  ChevronDown, 
  ChevronUp, 
  Code, 
  DollarSign,
  Tag,
  Clock,
  Users,
  ArrowRight,
  PlusCircle,
  Heart,
  TrendingUp,
  BarChart,
  Cpu,
  Database,
  Globe,
  Layers,
  Lock,
  RefreshCw,
  Rocket
} from 'lucide-react';
import PaymentButton from './PaymentButton';
import { stripeProducts } from '../stripe-config';

interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  subcategory: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
  tags: string[];
  price: number | null;
  isPremium: boolean;
  author: {
    name: string;
    avatar: string;
    rating: number;
  };
  downloads: number;
  rating: number;
  reviewCount: number;
  lastUpdated: string;
  estimatedTime: string;
  features: string[];
  techStack: string[];
  icon: React.ReactNode;
}

interface TemplateMarketplaceProps {
  onClose?: () => void;
  onSelectTemplate?: (template: Template) => void;
}

const TemplateMarketplace: React.FC<TemplateMarketplaceProps> = ({ onClose, onSelectTemplate }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedSubcategory, setSelectedSubcategory] = useState('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string[]>([]);
  const [priceFilter, setPriceFilter] = useState<'all' | 'free' | 'premium'>('all');
  const [sortBy, setSortBy] = useState<'popular' | 'newest' | 'rating'>('popular');
  const [expandedTemplate, setExpandedTemplate] = useState<string | null>(null);
  const [showFeatureRequestForm, setShowFeatureRequestForm] = useState(false);
  const [featureRequestData, setFeatureRequestData] = useState({
    title: '',
    description: '',
    budget: '',
    deadline: '',
    contactEmail: ''
  });

  const categories = [
    { id: 'all', name: 'All Templates' },
    { id: 'mobile', name: 'Mobile Development' },
    { id: 'hft', name: 'High-Frequency Trading' },
    { id: 'web', name: 'Web Development' },
    { id: 'ai', name: 'AI/ML' },
    { id: 'blockchain', name: 'Blockchain' }
  ];

  const subcategories: Record<string, { id: string, name: string }[]> = {
    all: [{ id: 'all', name: 'All Subcategories' }],
    mobile: [
      { id: 'all', name: 'All Mobile' },
      { id: 'android', name: 'Android' },
      { id: 'ios', name: 'iOS' },
      { id: 'flutter', name: 'Flutter' },
      { id: 'react-native', name: 'React Native' },
      { id: 'kotlin', name: 'Kotlin' },
      { id: 'swift', name: 'Swift' }
    ],
    hft: [
      { id: 'all', name: 'All HFT' },
      { id: 'market-data', name: 'Market Data' },
      { id: 'order-execution', name: 'Order Execution' },
      { id: 'strategy', name: 'Trading Strategy' },
      { id: 'backtesting', name: 'Backtesting' },
      { id: 'risk-management', name: 'Risk Management' }
    ],
    web: [
      { id: 'all', name: 'All Web' },
      { id: 'frontend', name: 'Frontend' },
      { id: 'backend', name: 'Backend' },
      { id: 'fullstack', name: 'Full Stack' }
    ],
    ai: [
      { id: 'all', name: 'All AI/ML' },
      { id: 'nlp', name: 'NLP' },
      { id: 'computer-vision', name: 'Computer Vision' },
      { id: 'reinforcement', name: 'Reinforcement Learning' }
    ],
    blockchain: [
      { id: 'all', name: 'All Blockchain' },
      { id: 'smart-contracts', name: 'Smart Contracts' },
      { id: 'defi', name: 'DeFi' },
      { id: 'nft', name: 'NFT' }
    ]
  };

  const templates: Template[] = [
    // Mobile Development Templates
    {
      id: 'android-hello-world',
      name: 'Android Hello World',
      description: 'A simple Android Hello World app with complete deployment pipeline to Google Play Store',
      category: 'mobile',
      subcategory: 'android',
      difficulty: 'Beginner',
      tags: ['Android', 'Kotlin', 'Deployment', 'Google Play'],
      price: null,
      isPremium: false,
      author: {
        name: 'Alex Chen',
        avatar: '👨‍💻',
        rating: 4.8
      },
      downloads: 1245,
      rating: 4.7,
      reviewCount: 89,
      lastUpdated: '2025-05-15',
      estimatedTime: '30 minutes',
      features: [
        'Complete Android app with Kotlin',
        'Automated keystore generation',
        'Secure signing configuration',
        'Google Play Store deployment pipeline',
        'CI/CD integration with GitHub Actions',
        'Step-by-step deployment guide'
      ],
      techStack: ['Android', 'Kotlin', 'Gradle', 'GitHub Actions', 'Fastlane'],
      icon: <Smartphone className="w-8 h-8 text-green-400" />
    },
    {
      id: 'ios-swift-starter',
      name: 'iOS Swift Starter',
      description: 'Modern iOS app template with Swift UI, Core Data integration, and App Store deployment',
      category: 'mobile',
      subcategory: 'ios',
      difficulty: 'Beginner',
      tags: ['iOS', 'Swift', 'SwiftUI', 'App Store'],
      price: null,
      isPremium: false,
      author: {
        name: 'Sarah Johnson',
        avatar: '👩‍💻',
        rating: 4.9
      },
      downloads: 987,
      rating: 4.8,
      reviewCount: 76,
      lastUpdated: '2025-06-02',
      estimatedTime: '45 minutes',
      features: [
        'Modern SwiftUI interface',
        'Core Data persistence',
        'User authentication',
        'Push notifications setup',
        'App Store deployment pipeline',
        'TestFlight integration'
      ],
      techStack: ['Swift', 'SwiftUI', 'Core Data', 'XCode Cloud', 'Fastlane'],
      icon: <Smartphone className="w-8 h-8 text-blue-400" />
    },
    {
      id: 'flutter-cross-platform',
      name: 'Flutter Cross-Platform App',
      description: 'Complete Flutter template for building beautiful cross-platform mobile applications',
      category: 'mobile',
      subcategory: 'flutter',
      difficulty: 'Intermediate',
      tags: ['Flutter', 'Dart', 'Cross-Platform', 'Material Design'],
      price: null,
      isPremium: false,
      author: {
        name: 'Raj Patel',
        avatar: '👨‍🚀',
        rating: 4.7
      },
      downloads: 1876,
      rating: 4.6,
      reviewCount: 124,
      lastUpdated: '2025-05-28',
      estimatedTime: '1 hour',
      features: [
        'Cross-platform (iOS & Android)',
        'Material Design components',
        'State management with Provider',
        'Firebase integration',
        'Local storage with Hive',
        'Responsive layouts'
      ],
      techStack: ['Flutter', 'Dart', 'Firebase', 'Provider', 'Hive'],
      icon: <Smartphone className="w-8 h-8 text-cyan-400" />
    },
    {
      id: 'react-native-ecommerce',
      name: 'React Native E-Commerce',
      description: 'Full-featured e-commerce mobile app with React Native, Redux, and payment integration',
      category: 'mobile',
      subcategory: 'react-native',
      difficulty: 'Advanced',
      tags: ['React Native', 'E-Commerce', 'Redux', 'Stripe'],
      price: 49.99,
      isPremium: true,
      author: {
        name: 'Maria Rodriguez',
        avatar: '👩‍🎨',
        rating: 4.9
      },
      downloads: 756,
      rating: 4.9,
      reviewCount: 68,
      lastUpdated: '2025-06-10',
      estimatedTime: '2-3 hours',
      features: [
        'Complete e-commerce functionality',
        'Product catalog with search',
        'Shopping cart and checkout',
        'Stripe payment integration',
        'User authentication',
        'Order history and tracking',
        'Push notifications'
      ],
      techStack: ['React Native', 'Redux', 'Stripe API', 'Firebase', 'Expo'],
      icon: <Smartphone className="w-8 h-8 text-purple-400" />
    },
    {
      id: 'kotlin-multiplatform',
      name: 'Kotlin Multiplatform Mobile',
      description: 'Share code between Android and iOS with Kotlin Multiplatform Mobile (KMM)',
      category: 'mobile',
      subcategory: 'kotlin',
      difficulty: 'Advanced',
      tags: ['Kotlin', 'KMM', 'Android', 'iOS', 'Shared Code'],
      price: 39.99,
      isPremium: true,
      author: {
        name: 'Hiroshi Tanaka',
        avatar: '👨‍💻',
        rating: 4.8
      },
      downloads: 432,
      rating: 4.7,
      reviewCount: 41,
      lastUpdated: '2025-05-20',
      estimatedTime: '3-4 hours',
      features: [
        'Shared business logic between platforms',
        'Platform-specific UI implementations',
        'Networking with Ktor',
        'Data persistence with SQLDelight',
        'Dependency injection with Koin',
        'CI/CD setup for both platforms'
      ],
      techStack: ['Kotlin', 'KMM', 'Ktor', 'SQLDelight', 'Koin'],
      icon: <Smartphone className="w-8 h-8 text-orange-400" />
    },
    {
      id: 'swift-ar-template',
      name: 'Swift AR Experience',
      description: 'iOS augmented reality template with ARKit, SceneKit, and RealityKit',
      category: 'mobile',
      subcategory: 'swift',
      difficulty: 'Expert',
      tags: ['iOS', 'Swift', 'AR', 'ARKit', '3D'],
      price: 59.99,
      isPremium: true,
      author: {
        name: 'James Wilson',
        avatar: '👨‍🔬',
        rating: 4.9
      },
      downloads: 287,
      rating: 4.8,
      reviewCount: 32,
      lastUpdated: '2025-06-05',
      estimatedTime: '4-5 hours',
      features: [
        'ARKit integration',
        '3D object placement',
        'Surface detection',
        'Gesture interactions',
        'Physics simulation',
        'Lighting and shadows',
        'Object occlusion'
      ],
      techStack: ['Swift', 'ARKit', 'SceneKit', 'RealityKit', 'Metal'],
      icon: <Smartphone className="w-8 h-8 text-blue-400" />
    },
    
    // High-Frequency Trading Templates
    {
      id: 'hft-market-data-engine',
      name: 'HFT Market Data Engine',
      description: 'High-performance market data processing engine for algorithmic trading',
      category: 'hft',
      subcategory: 'market-data',
      difficulty: 'Expert',
      tags: ['Rust', 'HFT', 'Low Latency', 'Market Data'],
      price: 199.99,
      isPremium: true,
      author: {
        name: 'Michael Chang',
        avatar: '👨‍💼',
        rating: 4.9
      },
      downloads: 156,
      rating: 4.9,
      reviewCount: 28,
      lastUpdated: '2025-06-01',
      estimatedTime: '5-7 hours',
      features: [
        'Sub-microsecond latency',
        'Lock-free data structures',
        'Memory-mapped file I/O',
        'FPGA integration support',
        'Multi-exchange connectivity',
        'Order book reconstruction',
        'Real-time analytics'
      ],
      techStack: ['Rust', 'C++', 'FPGA', 'Linux kernel tuning', 'ZeroMQ'],
      icon: <Zap className="w-8 h-8 text-yellow-400" />
    },
    {
      id: 'hft-order-execution',
      name: 'HFT Order Execution System',
      description: 'Ultra-low latency order execution system for high-frequency trading',
      category: 'hft',
      subcategory: 'order-execution',
      difficulty: 'Expert',
      tags: ['C++', 'HFT', 'Order Execution', 'Low Latency'],
      price: 249.99,
      isPremium: true,
      author: {
        name: 'David Kim',
        avatar: '👨‍🔧',
        rating: 5.0
      },
      downloads: 98,
      rating: 4.9,
      reviewCount: 17,
      lastUpdated: '2025-05-25',
      estimatedTime: '6-8 hours',
      features: [
        'Nanosecond-level execution',
        'Smart order routing',
        'Anti-gaming strategies',
        'Colocation optimizations',
        'Exchange connectivity adapters',
        'Risk management controls',
        'Performance monitoring'
      ],
      techStack: ['C++', 'Rust', 'FPGA', 'Kernel bypass networking', 'Custom TCP/IP stack'],
      icon: <Zap className="w-8 h-8 text-red-400" />
    },
    {
      id: 'hft-strategy-framework',
      name: 'HFT Strategy Framework',
      description: 'Framework for developing and backtesting high-frequency trading strategies',
      category: 'hft',
      subcategory: 'strategy',
      difficulty: 'Advanced',
      tags: ['Python', 'Rust', 'HFT', 'Strategy', 'Backtesting'],
      price: 179.99,
      isPremium: true,
      author: {
        name: 'Elena Petrova',
        avatar: '👩‍🔬',
        rating: 4.8
      },
      downloads: 215,
      rating: 4.7,
      reviewCount: 34,
      lastUpdated: '2025-06-08',
      estimatedTime: '4-6 hours',
      features: [
        'Strategy development framework',
        'Real-time signal generation',
        'Alpha model integration',
        'Risk and position management',
        'Performance analytics',
        'Strategy optimization',
        'Multi-asset support'
      ],
      techStack: ['Python', 'Rust', 'NumPy', 'pandas', 'TA-Lib', 'PyTorch'],
      icon: <TrendingUp className="w-8 h-8 text-green-400" />
    },
    {
      id: 'hft-backtesting-engine',
      name: 'HFT Backtesting Engine',
      description: 'High-performance backtesting engine for HFT strategies with tick-level precision',
      category: 'hft',
      subcategory: 'backtesting',
      difficulty: 'Advanced',
      tags: ['Python', 'C++', 'Backtesting', 'HFT', 'Simulation'],
      price: 149.99,
      isPremium: true,
      author: {
        name: 'Thomas Weber',
        avatar: '👨‍🎓',
        rating: 4.7
      },
      downloads: 187,
      rating: 4.6,
      reviewCount: 29,
      lastUpdated: '2025-05-30',
      estimatedTime: '3-5 hours',
      features: [
        'Tick-by-tick simulation',
        'Market microstructure modeling',
        'Exchange latency simulation',
        'Order book reconstruction',
        'Performance metrics',
        'Monte Carlo analysis',
        'Visualization tools'
      ],
      techStack: ['Python', 'C++', 'pandas', 'NumPy', 'Matplotlib', 'Plotly'],
      icon: <BarChart className="w-8 h-8 text-blue-400" />
    },
    {
      id: 'hft-risk-management',
      name: 'HFT Risk Management System',
      description: 'Real-time risk management system for high-frequency trading operations',
      category: 'hft',
      subcategory: 'risk-management',
      difficulty: 'Expert',
      tags: ['Java', 'Risk Management', 'HFT', 'Real-time'],
      price: 199.99,
      isPremium: true,
      author: {
        name: 'Sophia Chen',
        avatar: '👩‍🏫',
        rating: 4.9
      },
      downloads: 112,
      rating: 4.8,
      reviewCount: 21,
      lastUpdated: '2025-06-12',
      estimatedTime: '5-7 hours',
      features: [
        'Real-time position monitoring',
        'Pre-trade risk checks',
        'Exposure limits enforcement',
        'Circuit breakers',
        'Stress testing',
        'Regulatory compliance',
        'Reporting dashboard'
      ],
      techStack: ['Java', 'Spring Boot', 'Hazelcast', 'InfluxDB', 'Grafana'],
      icon: <Shield className="w-8 h-8 text-purple-400" />
    },
    
    // Additional Mobile Templates
    {
      id: 'android-mvvm-template',
      name: 'Android MVVM Architecture',
      description: 'Clean architecture Android template with MVVM, Jetpack Compose, and Kotlin Coroutines',
      category: 'mobile',
      subcategory: 'android',
      difficulty: 'Intermediate',
      tags: ['Android', 'MVVM', 'Jetpack Compose', 'Kotlin'],
      price: 29.99,
      isPremium: true,
      author: {
        name: 'Marcus Johnson',
        avatar: '👨‍🚀',
        rating: 4.8
      },
      downloads: 876,
      rating: 4.7,
      reviewCount: 72,
      lastUpdated: '2025-06-07',
      estimatedTime: '2-3 hours',
      features: [
        'MVVM architecture pattern',
        'Jetpack Compose UI',
        'Kotlin Coroutines for async operations',
        'Dependency injection with Hilt',
        'Room database integration',
        'Unit and UI testing setup',
        'CI/CD configuration'
      ],
      techStack: ['Kotlin', 'Jetpack Compose', 'Coroutines', 'Hilt', 'Room', 'Retrofit'],
      icon: <Smartphone className="w-8 h-8 text-green-400" />
    },
    {
      id: 'flutter-state-management',
      name: 'Flutter State Management',
      description: 'Flutter template showcasing different state management approaches',
      category: 'mobile',
      subcategory: 'flutter',
      difficulty: 'Intermediate',
      tags: ['Flutter', 'State Management', 'Provider', 'Riverpod', 'Bloc'],
      price: 19.99,
      isPremium: true,
      author: {
        name: 'Priya Sharma',
        avatar: '👩‍🎨',
        rating: 4.6
      },
      downloads: 654,
      rating: 4.5,
      reviewCount: 58,
      lastUpdated: '2025-05-22',
      estimatedTime: '2 hours',
      features: [
        'Multiple state management implementations',
        'Provider pattern',
        'Riverpod implementation',
        'BLoC pattern with flutter_bloc',
        'GetX implementation',
        'Performance comparisons',
        'Best practices guide'
      ],
      techStack: ['Flutter', 'Dart', 'Provider', 'Riverpod', 'flutter_bloc', 'GetX'],
      icon: <Smartphone className="w-8 h-8 text-cyan-400" />
    },
    
    // More HFT Templates
    {
      id: 'hft-exchange-connectivity',
      name: 'HFT Exchange Connectivity',
      description: 'Low-latency exchange connectivity adapters for major global exchanges',
      category: 'hft',
      subcategory: 'market-data',
      difficulty: 'Expert',
      tags: ['C++', 'FIX Protocol', 'Exchange API', 'Low Latency'],
      price: 299.99,
      isPremium: true,
      author: {
        name: 'Robert Zhang',
        avatar: '👨‍💻',
        rating: 5.0
      },
      downloads: 87,
      rating: 4.9,
      reviewCount: 15,
      lastUpdated: '2025-06-15',
      estimatedTime: '8-10 hours',
      features: [
        'FIX protocol implementation',
        'Binary protocol adapters',
        'Colocation optimizations',
        'Connection failover',
        'Message normalization',
        'Latency monitoring',
        'Support for 15+ major exchanges'
      ],
      techStack: ['C++', 'Rust', 'FIX Protocol', 'Custom TCP/IP', 'FPGA'],
      icon: <Globe className="w-8 h-8 text-blue-400" />
    },
    {
      id: 'hft-ml-prediction',
      name: 'HFT Machine Learning Prediction',
      description: 'ML-based market prediction system optimized for high-frequency trading',
      category: 'hft',
      subcategory: 'strategy',
      difficulty: 'Expert',
      tags: ['Python', 'Machine Learning', 'HFT', 'Prediction'],
      price: 349.99,
      isPremium: true,
      author: {
        name: 'Dr. Alan Wong',
        avatar: '👨‍🔬',
        rating: 4.9
      },
      downloads: 76,
      rating: 4.8,
      reviewCount: 14,
      lastUpdated: '2025-06-18',
      estimatedTime: '10-12 hours',
      features: [
        'Real-time feature engineering',
        'Low-latency inference pipeline',
        'Ensemble model architecture',
        'Adaptive learning system',
        'Feature importance analysis',
        'Model performance monitoring',
        'Deployment optimization'
      ],
      techStack: ['Python', 'PyTorch', 'CUDA', 'C++ extensions', 'Ray', 'MLflow'],
      icon: <Cpu className="w-8 h-8 text-purple-400" />
    }
  ];

  // Add more templates to reach 50+ for mobile development
  const mobileTemplateNames = [
    'iOS Core Data Manager',
    'Android Jetpack Navigation',
    'Flutter Responsive UI',
    'React Native Authentication',
    'Kotlin Multiplatform Mobile',
    'Swift UI Animations',
    'Flutter BLoC Pattern',
    'Android Material Design 3',
    'iOS SwiftUI Components',
    'React Native Maps & Location',
    'Flutter Firebase Integration',
    'Android Room Database',
    'iOS CloudKit Sync',
    'React Native Animations',
    'Flutter State Management',
    'Android Camera & Gallery',
    'iOS ARKit Starter',
    'React Native E-Commerce',
    'Flutter Social Network',
    'Android Wear OS',
    'iOS HealthKit Integration',
    'React Native Push Notifications',
    'Flutter Chat Application',
    'Android ML Kit Vision',
    'iOS Core ML Implementation',
    'React Native Offline First',
    'Flutter Internationalization',
    'Android Biometric Auth',
    'iOS StoreKit In-App Purchases',
    'React Native TV App',
    'Flutter Desktop App',
    'Android Instant App',
    'iOS App Clips',
    'React Native Web',
    'Flutter Game Development',
    'Android Audio Processing',
    'iOS Vision Framework',
    'React Native Gestures',
    'Flutter Custom Painters',
    'Android Jetpack Compose',
    'iOS Combine Framework',
    'React Native Performance',
    'Flutter Testing Suite',
    'Android Security Patterns',
    'iOS Accessibility'
  ];

  // Generate additional mobile templates
  const additionalMobileTemplates: Template[] = mobileTemplateNames.map((name, index) => {
    const subcategories = ['android', 'ios', 'flutter', 'react-native', 'kotlin', 'swift'];
    const subcategory = subcategories[index % subcategories.length];
    const difficulty = ['Beginner', 'Intermediate', 'Advanced', 'Expert'][Math.floor(Math.random() * 4)] as 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
    const isPremium = Math.random() > 0.6;
    
    let icon;
    if (subcategory === 'android' || subcategory === 'kotlin') {
      icon = <Smartphone className="w-8 h-8 text-green-400" />;
    } else if (subcategory === 'ios' || subcategory === 'swift') {
      icon = <Smartphone className="w-8 h-8 text-blue-400" />;
    } else if (subcategory === 'flutter') {
      icon = <Smartphone className="w-8 h-8 text-cyan-400" />;
    } else {
      icon = <Smartphone className="w-8 h-8 text-purple-400" />;
    }
    
    return {
      id: `mobile-template-${index + 1}`,
      name,
      description: `Professional ${name} template for modern mobile development`,
      category: 'mobile',
      subcategory,
      difficulty,
      tags: [subcategory, 'Mobile', name.split(' ')[0]],
      price: isPremium ? 19.99 + Math.floor(Math.random() * 30) : null,
      isPremium,
      author: {
        name: ['Alex Chen', 'Sarah Johnson', 'Raj Patel', 'Maria Rodriguez', 'James Wilson'][Math.floor(Math.random() * 5)],
        avatar: ['👨‍💻', '👩‍💻', '👨‍🚀', '👩‍🎨', '👨‍🔬'][Math.floor(Math.random() * 5)],
        rating: 4.5 + Math.random() * 0.5
      },
      downloads: Math.floor(Math.random() * 1000) + 100,
      rating: 4.0 + Math.random(),
      reviewCount: Math.floor(Math.random() * 100) + 10,
      lastUpdated: `2025-${Math.floor(Math.random() * 6) + 1}-${Math.floor(Math.random() * 28) + 1}`,
      estimatedTime: `${Math.floor(Math.random() * 5) + 1} hours`,
      features: [
        'Professional UI/UX design',
        'Clean architecture implementation',
        'Comprehensive documentation',
        'Unit and integration tests',
        'Performance optimizations',
        'Accessibility support'
      ],
      techStack: ['Kotlin', 'Swift', 'Flutter', 'React Native', 'Jetpack Compose', 'SwiftUI'].filter(() => Math.random() > 0.5),
      icon
    };
  });

  // Combine all templates
  const allTemplates = [...templates, ...additionalMobileTemplates];

  // Filter templates based on search and filters
  const filteredTemplates = allTemplates.filter(template => {
    // Filter by search query
    const matchesSearch = 
      searchQuery === '' || 
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    // Filter by category
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
    
    // Filter by subcategory
    const matchesSubcategory = selectedSubcategory === 'all' || template.subcategory === selectedSubcategory;
    
    // Filter by difficulty
    const matchesDifficulty = selectedDifficulty.length === 0 || selectedDifficulty.includes(template.difficulty);
    
    // Filter by price
    const matchesPrice = 
      priceFilter === 'all' || 
      (priceFilter === 'free' && !template.isPremium) || 
      (priceFilter === 'premium' && template.isPremium);
    
    return matchesSearch && matchesCategory && matchesSubcategory && matchesDifficulty && matchesPrice;
  });

  // Sort templates
  const sortedTemplates = [...filteredTemplates].sort((a, b) => {
    if (sortBy === 'popular') {
      return b.downloads - a.downloads;
    } else if (sortBy === 'newest') {
      return new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime();
    } else {
      return b.rating - a.rating;
    }
  });

  const handleDifficultyChange = (difficulty: string) => {
    if (selectedDifficulty.includes(difficulty)) {
      setSelectedDifficulty(selectedDifficulty.filter(d => d !== difficulty));
    } else {
      setSelectedDifficulty([...selectedDifficulty, difficulty]);
    }
  };

  const handleCategoryChange = (category: string) => {
    setSelectedCategory(category);
    setSelectedSubcategory('all');
  };

  const handleFeatureRequestSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // In a real app, this would submit to a backend
    console.log('Feature request submitted:', featureRequestData);
    alert('Thank you for your feature request! We will contact you soon.');
    setShowFeatureRequestForm(false);
    setFeatureRequestData({
      title: '',
      description: '',
      budget: '',
      deadline: '',
      contactEmail: ''
    });
  };

  const handleTemplateSelect = (template: Template) => {
    if (onSelectTemplate) {
      onSelectTemplate(template);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-gray-800 rounded-xl border border-gray-700 w-full max-w-7xl max-h-[95vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-gradient-to-br from-orange-500 to-red-500 rounded-xl shadow-lg">
              <Code className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Template Marketplace</h2>
              <p className="text-gray-400">Browse 50+ professional templates for mobile development and HFT</p>
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors text-xl"
            >
              ✕
            </button>
          )}
        </div>

        {showFeatureRequestForm ? (
          <div className="p-6 overflow-y-auto max-h-[80vh]">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Request Custom Feature or Template</h3>
              <button
                onClick={() => setShowFeatureRequestForm(false)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
            
            <form onSubmit={handleFeatureRequestSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Feature/Template Title
                </label>
                <input
                  type="text"
                  value={featureRequestData.title}
                  onChange={(e) => setFeatureRequestData({...featureRequestData, title: e.target.value})}
                  required
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  placeholder="E.g., Android ML Kit Integration Template"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Detailed Description
                </label>
                <textarea
                  value={featureRequestData.description}
                  onChange={(e) => setFeatureRequestData({...featureRequestData, description: e.target.value})}
                  required
                  rows={5}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  placeholder="Describe the feature or template you need in detail..."
                />
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Budget (USD)
                  </label>
                  <input
                    type="number"
                    value={featureRequestData.budget}
                    onChange={(e) => setFeatureRequestData({...featureRequestData, budget: e.target.value})}
                    required
                    min="1"
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                    placeholder="Your budget for this request"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Deadline
                  </label>
                  <input
                    type="date"
                    value={featureRequestData.deadline}
                    onChange={(e) => setFeatureRequestData({...featureRequestData, deadline: e.target.value})}
                    required
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Contact Email
                </label>
                <input
                  type="email"
                  value={featureRequestData.contactEmail}
                  onChange={(e) => setFeatureRequestData({...featureRequestData, contactEmail: e.target.value})}
                  required
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  placeholder="Your email address"
                />
              </div>
              
              <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                <h4 className="font-medium text-white mb-2">How It Works</h4>
                <ol className="space-y-2 text-gray-300 list-decimal list-inside">
                  <li>Submit your feature or template request with details</li>
                  <li>Our team will review and provide a quote within 24 hours</li>
                  <li>Once approved, we'll start development immediately</li>
                  <li>You'll receive regular updates on progress</li>
                  <li>Payment is only processed when you're satisfied with the result</li>
                </ol>
              </div>
              
              <div className="flex justify-end space-x-4">
                <button
                  type="button"
                  onClick={() => setShowFeatureRequestForm(false)}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors"
                >
                  Submit Request
                </button>
              </div>
            </form>
          </div>
        ) : (
          <>
            {/* Search and Filters */}
            <div className="p-6 border-b border-gray-700 bg-gray-800">
              <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search templates..."
                    className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                
                <div className="flex space-x-2">
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="popular">Most Popular</option>
                    <option value="newest">Newest</option>
                    <option value="rating">Highest Rated</option>
                  </select>
                  
                  <button
                    onClick={() => setShowFeatureRequestForm(true)}
                    className="flex items-center space-x-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors"
                  >
                    <PlusCircle className="w-4 h-4" />
                    <span>Request Feature</span>
                  </button>
                </div>
              </div>
            </div>

            <div className="flex h-[calc(95vh-12rem)]">
              {/* Filters Sidebar */}
              <div className="w-64 bg-gray-900 border-r border-gray-700 p-6 overflow-y-auto">
                <div className="space-y-6">
                  {/* Categories */}
                  <div>
                    <h3 className="text-sm font-medium text-gray-300 mb-3">Categories</h3>
                    <div className="space-y-2">
                      {categories.map((category) => (
                        <button
                          key={category.id}
                          onClick={() => handleCategoryChange(category.id)}
                          className={`flex items-center justify-between w-full px-3 py-2 rounded-lg transition-colors ${
                            selectedCategory === category.id
                              ? 'bg-orange-600 text-white'
                              : 'text-gray-300 hover:bg-gray-700'
                          }`}
                        >
                          <span>{category.name}</span>
                          <span className="text-xs bg-gray-800 px-2 py-1 rounded-full">
                            {allTemplates.filter(t => category.id === 'all' || t.category === category.id).length}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  {/* Subcategories */}
                  {selectedCategory !== 'all' && (
                    <div>
                      <h3 className="text-sm font-medium text-gray-300 mb-3">Subcategories</h3>
                      <div className="space-y-2">
                        {subcategories[selectedCategory].map((subcategory) => (
                          <button
                            key={subcategory.id}
                            onClick={() => setSelectedSubcategory(subcategory.id)}
                            className={`flex items-center justify-between w-full px-3 py-2 rounded-lg transition-colors ${
                              selectedSubcategory === subcategory.id
                                ? 'bg-blue-600 text-white'
                                : 'text-gray-300 hover:bg-gray-700'
                            }`}
                          >
                            <span>{subcategory.name}</span>
                            <span className="text-xs bg-gray-800 px-2 py-1 rounded-full">
                              {allTemplates.filter(t => 
                                (subcategory.id === 'all' && t.category === selectedCategory) || 
                                t.subcategory === subcategory.id
                              ).length}
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Difficulty */}
                  <div>
                    <h3 className="text-sm font-medium text-gray-300 mb-3">Difficulty</h3>
                    <div className="space-y-2">
                      {['Beginner', 'Intermediate', 'Advanced', 'Expert'].map((difficulty) => (
                        <label key={difficulty} className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={selectedDifficulty.includes(difficulty)}
                            onChange={() => handleDifficultyChange(difficulty)}
                            className="rounded border-gray-600 text-orange-600 focus:ring-orange-500"
                          />
                          <span className="text-gray-300">{difficulty}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  {/* Price */}
                  <div>
                    <h3 className="text-sm font-medium text-gray-300 mb-3">Price</h3>
                    <div className="space-y-2">
                      <button
                        onClick={() => setPriceFilter('all')}
                        className={`flex items-center justify-between w-full px-3 py-2 rounded-lg transition-colors ${
                          priceFilter === 'all'
                            ? 'bg-orange-600 text-white'
                            : 'text-gray-300 hover:bg-gray-700'
                        }`}
                      >
                        <span>All</span>
                      </button>
                      <button
                        onClick={() => setPriceFilter('free')}
                        className={`flex items-center justify-between w-full px-3 py-2 rounded-lg transition-colors ${
                          priceFilter === 'free'
                            ? 'bg-orange-600 text-white'
                            : 'text-gray-300 hover:bg-gray-700'
                        }`}
                      >
                        <span>Free</span>
                      </button>
                      <button
                        onClick={() => setPriceFilter('premium')}
                        className={`flex items-center justify-between w-full px-3 py-2 rounded-lg transition-colors ${
                          priceFilter === 'premium'
                            ? 'bg-orange-600 text-white'
                            : 'text-gray-300 hover:bg-gray-700'
                        }`}
                      >
                        <span>Premium</span>
                      </button>
                    </div>
                  </div>
                  
                  {/* Stats */}
                  <div className="pt-4 border-t border-gray-700">
                    <div className="text-sm text-gray-400 mb-3">Template Stats</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-gray-800 p-2 rounded-lg text-center">
                        <div className="text-lg font-bold text-white">{allTemplates.length}</div>
                        <div className="text-xs text-gray-400">Total</div>
                      </div>
                      <div className="bg-gray-800 p-2 rounded-lg text-center">
                        <div className="text-lg font-bold text-white">{allTemplates.filter(t => t.category === 'mobile').length}</div>
                        <div className="text-xs text-gray-400">Mobile</div>
                      </div>
                      <div className="bg-gray-800 p-2 rounded-lg text-center">
                        <div className="text-lg font-bold text-white">{allTemplates.filter(t => t.category === 'hft').length}</div>
                        <div className="text-xs text-gray-400">HFT</div>
                      </div>
                      <div className="bg-gray-800 p-2 rounded-lg text-center">
                        <div className="text-lg font-bold text-white">{allTemplates.filter(t => t.isPremium).length}</div>
                        <div className="text-xs text-gray-400">Premium</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Templates Grid */}
              <div className="flex-1 p-6 overflow-y-auto">
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-white">
                    {sortedTemplates.length} templates found
                  </h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {sortedTemplates.map((template) => (
                    <div
                      key={template.id}
                      className="bg-gray-700 rounded-lg border border-gray-600 hover:border-orange-500 transition-colors overflow-hidden flex flex-col"
                    >
                      <div className="p-6">
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex items-center space-x-3">
                            <div className="p-3 bg-gray-800 rounded-lg">
                              {template.icon}
                            </div>
                            <div>
                              <h3 className="text-lg font-semibold text-white">{template.name}</h3>
                              <div className="flex items-center space-x-2 mt-1">
                                <div className="flex items-center">
                                  <Star className="w-3 h-3 text-yellow-400 fill-current" />
                                  <span className="text-xs text-gray-300 ml-1">{template.rating.toFixed(1)}</span>
                                </div>
                                <span className="text-xs text-gray-400">({template.reviewCount} reviews)</span>
                              </div>
                            </div>
                          </div>
                          {template.isPremium && (
                            <div className="bg-orange-600 text-white text-xs px-2 py-1 rounded-full">
                              Premium
                            </div>
                          )}
                        </div>
                        
                        <p className="text-gray-300 text-sm mb-4 line-clamp-2">
                          {template.description}
                        </p>
                        
                        <div className="flex flex-wrap gap-2 mb-4">
                          {template.tags.slice(0, 3).map((tag) => (
                            <span key={tag} className="px-2 py-1 bg-gray-800 text-gray-300 rounded text-xs">
                              {tag}
                            </span>
                          ))}
                          {template.tags.length > 3 && (
                            <span className="px-2 py-1 bg-gray-800 text-gray-400 rounded text-xs">
                              +{template.tags.length - 3}
                            </span>
                          )}
                        </div>
                        
                        <div className="flex items-center justify-between text-sm text-gray-400 mb-4">
                          <div className="flex items-center space-x-1">
                            <Download className="w-3 h-3" />
                            <span>{template.downloads}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Clock className="w-3 h-3" />
                            <span>{template.estimatedTime}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Users className="w-3 h-3" />
                            <span>{template.difficulty}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="mt-auto border-t border-gray-600 p-4 bg-gray-800 flex items-center justify-between">
                        <div>
                          {template.isPremium ? (
                            <div className="text-lg font-bold text-white">${template.price?.toFixed(2)}</div>
                          ) : (
                            <div className="text-lg font-bold text-green-400">Free</div>
                          )}
                        </div>
                        
                        <div className="flex space-x-2">
                          <button
                            onClick={() => setExpandedTemplate(expandedTemplate === template.id ? null : template.id)}
                            className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 transition-colors"
                          >
                            {expandedTemplate === template.id ? (
                              <ChevronUp className="w-4 h-4" />
                            ) : (
                              <ChevronDown className="w-4 h-4" />
                            )}
                          </button>
                          
                          {template.isPremium ? (
                            <PaymentButton
                              product={{
                                priceId: `price_template_${template.id}`,
                                name: template.name,
                                description: template.description,
                                mode: 'payment',
                                price: template.price || 0,
                                currency: 'usd'
                              }}
                              variant="primary"
                              size="sm"
                            >
                              Purchase
                            </PaymentButton>
                          ) : (
                            <button
                              onClick={() => handleTemplateSelect(template)}
                              className="flex items-center space-x-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm transition-colors"
                            >
                              <Download className="w-3 h-3" />
                              <span>Use Template</span>
                            </button>
                          )}
                        </div>
                      </div>
                      
                      {/* Expanded Details */}
                      {expandedTemplate === template.id && (
                        <div className="p-4 border-t border-gray-600 bg-gray-800">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                            <div>
                              <h4 className="font-medium text-white mb-2">Features</h4>
                              <ul className="space-y-1">
                                {template.features.map((feature, index) => (
                                  <li key={index} className="flex items-start space-x-2 text-sm text-gray-300">
                                    <div className="w-4 h-4 rounded-full bg-green-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                                      <CheckCircle className="w-3 h-3 text-white" />
                                    </div>
                                    <span>{feature}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                            
                            <div>
                              <h4 className="font-medium text-white mb-2">Tech Stack</h4>
                              <div className="flex flex-wrap gap-2">
                                {template.techStack.map((tech) => (
                                  <span key={tech} className="px-2 py-1 bg-blue-900/30 text-blue-300 rounded-full text-xs">
                                    {tech}
                                  </span>
                                ))}
                              </div>
                              
                              <h4 className="font-medium text-white mt-4 mb-2">Author</h4>
                              <div className="flex items-center space-x-2">
                                <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
                                  {template.author.avatar}
                                </div>
                                <div>
                                  <div className="text-sm font-medium text-white">{template.author.name}</div>
                                  <div className="flex items-center">
                                    <Star className="w-3 h-3 text-yellow-400 fill-current" />
                                    <span className="text-xs text-gray-300 ml-1">{template.author.rating.toFixed(1)}</span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex justify-between items-center">
                            <button className="flex items-center space-x-1 text-blue-400 hover:text-blue-300 text-sm">
                              <Heart className="w-3 h-3" />
                              <span>Add to Favorites</span>
                            </button>
                            
                            <button
                              onClick={() => handleTemplateSelect(template)}
                              className="flex items-center space-x-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg text-sm transition-colors"
                            >
                              <ArrowRight className="w-4 h-4" />
                              <span>{template.isPremium ? 'Purchase & Use' : 'Use Template'}</span>
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                
                {sortedTemplates.length === 0 && (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Search className="w-8 h-8 text-gray-500" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">No templates found</h3>
                    <p className="text-gray-400 max-w-md mx-auto">
                      Try adjusting your search or filters to find what you're looking for.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default TemplateMarketplace;