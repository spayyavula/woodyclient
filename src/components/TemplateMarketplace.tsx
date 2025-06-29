import React, { useState, useEffect } from 'react';
import { 
  Code, 
  Search, 
  Filter, 
  Tag, 
  Star, 
  Clock, 
  Download, 
  ArrowRight, 
  CheckCircle, 
  X,
  DollarSign,
  Smartphone,
  Cpu,
  BarChart,
  Zap,
  Shield,
  Database
} from 'lucide-react';

interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  subcategory?: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
  tags: string[];
  price: number;
  isPremium: boolean;
  rating: number;
  downloads: number;
  author: {
    name: string;
    avatar: string;
  };
  lastUpdated: string;
  previewImage?: string;
  features: string[];
}

interface TemplateMarketplaceProps {
  onClose: () => void;
  onSelectTemplate: (template: Template) => void;
}

const TemplateMarketplace: React.FC<TemplateMarketplaceProps> = ({ onClose, onSelectTemplate }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedSubcategory, setSelectedSubcategory] = useState('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string[]>([]);
  const [priceRange, setPriceRange] = useState<[number, number]>([0, 500]);
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);
  const [showPremiumOnly, setShowPremiumOnly] = useState(false);
  const [sortBy, setSortBy] = useState('popular');

  const categories = [
    { id: 'all', name: 'All Categories' },
    { id: 'mobile', name: 'Mobile Development', icon: <Smartphone className="w-4 h-4" /> },
    { id: 'hft', name: 'High-Frequency Trading', icon: <BarChart className="w-4 h-4" /> },
    { id: 'ai', name: 'AI & Machine Learning', icon: <Cpu className="w-4 h-4" /> },
    { id: 'blockchain', name: 'Blockchain', icon: <Database className="w-4 h-4" /> },
    { id: 'security', name: 'Security', icon: <Shield className="w-4 h-4" /> },
    { id: 'performance', name: 'Performance', icon: <Zap className="w-4 h-4" /> }
  ];

  const subcategories: Record<string, { id: string, name: string }[]> = {
    mobile: [
      { id: 'all', name: 'All Mobile' },
      { id: 'android', name: 'Android' },
      { id: 'ios', name: 'iOS' },
      { id: 'flutter', name: 'Flutter' },
      { id: 'react-native', name: 'React Native' }
    ],
    hft: [
      { id: 'all', name: 'All HFT' },
      { id: 'market-data', name: 'Market Data' },
      { id: 'order-execution', name: 'Order Execution' },
      { id: 'strategy', name: 'Trading Strategy' },
      { id: 'risk-management', name: 'Risk Management' }
    ],
    ai: [
      { id: 'all', name: 'All AI/ML' },
      { id: 'nlp', name: 'Natural Language Processing' },
      { id: 'computer-vision', name: 'Computer Vision' },
      { id: 'reinforcement-learning', name: 'Reinforcement Learning' },
      { id: 'generative-ai', name: 'Generative AI' }
    ],
    blockchain: [
      { id: 'all', name: 'All Blockchain' },
      { id: 'smart-contracts', name: 'Smart Contracts' },
      { id: 'defi', name: 'DeFi' },
      { id: 'nft', name: 'NFT' },
      { id: 'wallet', name: 'Wallet Integration' }
    ],
    security: [
      { id: 'all', name: 'All Security' },
      { id: 'encryption', name: 'Encryption' },
      { id: 'authentication', name: 'Authentication' },
      { id: 'penetration-testing', name: 'Penetration Testing' },
      { id: 'secure-coding', name: 'Secure Coding' }
    ],
    performance: [
      { id: 'all', name: 'All Performance' },
      { id: 'optimization', name: 'Code Optimization' },
      { id: 'profiling', name: 'Profiling' },
      { id: 'memory-management', name: 'Memory Management' },
      { id: 'concurrency', name: 'Concurrency' }
    ]
  };

  // Sample templates data
  const templates: Template[] = [
    {
      id: 'android-ml-kit',
      name: 'Android ML Kit Integration',
      description: 'Complete template for integrating Google ML Kit into Android apps with Rust native processing',
      category: 'mobile',
      subcategory: 'android',
      difficulty: 'Intermediate',
      tags: ['Android', 'ML Kit', 'Computer Vision', 'Rust'],
      price: 49.99,
      isPremium: true,
      rating: 4.8,
      downloads: 1245,
      author: {
        name: 'Sarah Chen',
        avatar: '👩‍💻'
      },
      lastUpdated: '2025-05-15',
      features: [
        'Face detection and recognition',
        'Text recognition (OCR)',
        'Barcode scanning',
        'Image labeling',
        'Object detection and tracking',
        'Rust backend for performance'
      ]
    },
    {
      id: 'ios-arkit',
      name: 'iOS ARKit with Rust Processing',
      description: 'Augmented reality template with Rust-powered processing for iOS applications',
      category: 'mobile',
      subcategory: 'ios',
      difficulty: 'Advanced',
      tags: ['iOS', 'ARKit', 'Augmented Reality', 'Rust'],
      price: 79.99,
      isPremium: true,
      rating: 4.9,
      downloads: 876,
      author: {
        name: 'Michael Wong',
        avatar: '👨‍💻'
      },
      lastUpdated: '2025-06-02',
      features: [
        'World tracking',
        'Surface detection',
        'Object placement',
        'Lighting estimation',
        'Rust-powered image processing',
        'Performance optimizations'
      ]
    },
    {
      id: 'flutter-rust-bridge',
      name: 'Flutter Rust Bridge Template',
      description: 'Cross-platform Flutter template with Rust backend for maximum performance',
      category: 'mobile',
      subcategory: 'flutter',
      difficulty: 'Intermediate',
      tags: ['Flutter', 'Rust', 'Cross-platform'],
      price: 39.99,
      isPremium: true,
      rating: 4.7,
      downloads: 2134,
      author: {
        name: 'Elena Rodriguez',
        avatar: '👩‍💻'
      },
      lastUpdated: '2025-05-28',
      features: [
        'Rust FFI integration',
        'Cross-platform support',
        'State management',
        'Optimized rendering',
        'Native performance',
        'Shared business logic'
      ]
    },
    {
      id: 'hft-market-data',
      name: 'HFT Market Data Processing',
      description: 'High-performance market data processing system for high-frequency trading',
      category: 'hft',
      subcategory: 'market-data',
      difficulty: 'Expert',
      tags: ['HFT', 'Market Data', 'Low Latency', 'Rust'],
      price: 299.99,
      isPremium: true,
      rating: 4.9,
      downloads: 342,
      author: {
        name: 'James Wilson',
        avatar: '👨‍💼'
      },
      lastUpdated: '2025-06-10',
      features: [
        'Sub-microsecond processing',
        'Market data normalization',
        'Order book reconstruction',
        'FPGA integration',
        'Multi-exchange support',
        'Latency monitoring'
      ]
    },
    {
      id: 'hft-order-execution',
      name: 'HFT Order Execution Engine',
      description: 'Ultra-low latency order execution engine for high-frequency trading',
      category: 'hft',
      subcategory: 'order-execution',
      difficulty: 'Expert',
      tags: ['HFT', 'Order Execution', 'Low Latency', 'Rust'],
      price: 349.99,
      isPremium: true,
      rating: 4.8,
      downloads: 287,
      author: {
        name: 'Alex Thompson',
        avatar: '👨‍💼'
      },
      lastUpdated: '2025-06-05',
      features: [
        'Smart order routing',
        'Iceberg orders',
        'TWAP/VWAP algorithms',
        'Anti-gaming protection',
        'Risk controls',
        'Performance metrics'
      ]
    },
    {
      id: 'android-hello-world',
      name: 'Android Hello World',
      description: 'A simple Android Hello World app with complete deployment pipeline to Google Play Store',
      category: 'mobile',
      subcategory: 'android',
      difficulty: 'Beginner',
      tags: ['Android', 'Kotlin', 'Deployment', 'Google Play'],
      price: 0,
      isPremium: false,
      rating: 4.5,
      downloads: 5678,
      author: {
        name: 'Community',
        avatar: '👥'
      },
      lastUpdated: '2025-04-20',
      features: [
        'Complete Android app with Kotlin',
        'Automated keystore generation',
        'Secure signing configuration',
        'Google Play Store deployment pipeline',
        'CI/CD integration with GitHub Actions',
        'Step-by-step deployment guide'
      ]
    },
    {
      id: 'react-dashboard',
      name: 'React Dashboard',
      description: 'Modern React dashboard with TypeScript, Tailwind CSS, and data visualization',
      category: 'mobile',
      subcategory: 'react-native',
      difficulty: 'Intermediate',
      tags: ['React', 'TypeScript', 'Dashboard', 'Charts'],
      price: 0,
      isPremium: false,
      rating: 4.6,
      downloads: 4321,
      author: {
        name: 'Community',
        avatar: '👥'
      },
      lastUpdated: '2025-05-10',
      features: [
        'Modern React with TypeScript',
        'Responsive design with Tailwind CSS',
        'Interactive data visualization',
        'Performance metrics',
        'Real-time updates',
        'Mobile-friendly interface'
      ]
    },
    {
      id: 'nlp-sentiment-analysis',
      name: 'NLP Sentiment Analysis',
      description: 'Natural language processing template for sentiment analysis with Rust backend',
      category: 'ai',
      subcategory: 'nlp',
      difficulty: 'Advanced',
      tags: ['NLP', 'Sentiment Analysis', 'Rust', 'Machine Learning'],
      price: 129.99,
      isPremium: true,
      rating: 4.7,
      downloads: 567,
      author: {
        name: 'Dr. Priya Sharma',
        avatar: '👩‍🔬'
      },
      lastUpdated: '2025-05-20',
      features: [
        'Pre-trained sentiment models',
        'Custom model training pipeline',
        'Multi-language support',
        'Rust-powered processing',
        'Real-time analysis',
        'Batch processing capabilities'
      ]
    },
    {
      id: 'computer-vision-object-detection',
      name: 'Computer Vision Object Detection',
      description: 'Advanced computer vision template for real-time object detection',
      category: 'ai',
      subcategory: 'computer-vision',
      difficulty: 'Advanced',
      tags: ['Computer Vision', 'Object Detection', 'Rust', 'YOLO'],
      price: 149.99,
      isPremium: true,
      rating: 4.8,
      downloads: 432,
      author: {
        name: 'Marcus Johnson',
        avatar: '👨‍🔬'
      },
      lastUpdated: '2025-06-01',
      features: [
        'Real-time object detection',
        'Multiple model support (YOLO, SSD, etc.)',
        'Camera integration',
        'Rust-powered processing',
        'Edge deployment optimizations',
        'Custom model training'
      ]
    },
    {
      id: 'smart-contract-defi',
      name: 'DeFi Smart Contract Template',
      description: 'Secure and audited smart contract templates for DeFi applications',
      category: 'blockchain',
      subcategory: 'defi',
      difficulty: 'Expert',
      tags: ['Blockchain', 'DeFi', 'Smart Contracts', 'Solidity'],
      price: 199.99,
      isPremium: true,
      rating: 4.9,
      downloads: 321,
      author: {
        name: 'Satoshi Nakamura',
        avatar: '👨‍💻'
      },
      lastUpdated: '2025-05-25',
      features: [
        'Audited contract templates',
        'Yield farming protocols',
        'Liquidity pool implementations',
        'Flash loan protection',
        'Gas optimization',
        'Security best practices'
      ]
    }
  ];

  const filteredTemplates = templates.filter(template => {
    // Filter by search query
    const matchesSearch = 
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
    const matchesPrice = template.price >= priceRange[0] && template.price <= priceRange[1];
    
    // Filter by premium status
    const matchesPremium = !showPremiumOnly || template.isPremium;
    
    return matchesSearch && matchesCategory && matchesSubcategory && matchesDifficulty && matchesPrice && matchesPremium;
  });

  // Sort templates
  const sortedTemplates = [...filteredTemplates].sort((a, b) => {
    switch (sortBy) {
      case 'popular':
        return b.downloads - a.downloads;
      case 'rating':
        return b.rating - a.rating;
      case 'newest':
        return new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime();
      case 'price-low':
        return a.price - b.price;
      case 'price-high':
        return b.price - a.price;
      default:
        return 0;
    }
  });

  const handleCategoryChange = (category: string) => {
    setSelectedCategory(category);
    setSelectedSubcategory('all');
  };

  const handleDifficultyToggle = (difficulty: string) => {
    setSelectedDifficulty(prev => 
      prev.includes(difficulty) 
        ? prev.filter(d => d !== difficulty) 
        : [...prev, difficulty]
    );
  };

  const handlePriceRangeChange = (value: number, index: number) => {
    setPriceRange(prev => {
      const newRange = [...prev] as [number, number];
      newRange[index] = value;
      return newRange;
    });
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-xl border border-gray-700 w-full max-w-7xl max-h-[95vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-blue-500 rounded-xl shadow-lg">
              <Code className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Template Marketplace</h2>
              <p className="text-gray-400">Browse 50+ premium templates for mobile development and HFT</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-xl"
          >
            ✕
          </button>
        </div>

        <div className="flex h-[80vh]">
          {/* Sidebar */}
          <div className="w-80 bg-gray-900 border-r border-gray-700 p-6 overflow-y-auto">
            <div className="mb-6">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search templates..."
                  className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-3">Categories</h3>
              <div className="space-y-2">
                {categories.map((category) => (
                  <button
                    key={category.id}
                    onClick={() => handleCategoryChange(category.id)}
                    className={`flex items-center space-x-2 w-full px-3 py-2 rounded-lg transition-colors ${
                      selectedCategory === category.id
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    {category.icon}
                    <span>{category.name}</span>
                  </button>
                ))}
              </div>
            </div>

            {selectedCategory !== 'all' && subcategories[selectedCategory] && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Subcategories</h3>
                <div className="space-y-2">
                  {subcategories[selectedCategory].map((subcategory) => (
                    <button
                      key={subcategory.id}
                      onClick={() => setSelectedSubcategory(subcategory.id)}
                      className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                        selectedSubcategory === subcategory.id
                          ? 'bg-blue-600 text-white'
                          : 'text-gray-300 hover:bg-gray-700'
                      }`}
                    >
                      {subcategory.name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-3">Difficulty</h3>
              <div className="space-y-2">
                {['Beginner', 'Intermediate', 'Advanced', 'Expert'].map((difficulty) => (
                  <label key={difficulty} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedDifficulty.includes(difficulty)}
                      onChange={() => handleDifficultyToggle(difficulty)}
                      className="rounded border-gray-600 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-gray-300">{difficulty}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-3">Price Range</h3>
              <div className="space-y-4">
                <div className="flex justify-between text-sm text-gray-400">
                  <span>${priceRange[0]}</span>
                  <span>${priceRange[1]}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="500"
                  step="10"
                  value={priceRange[1]}
                  onChange={(e) => handlePriceRangeChange(parseInt(e.target.value), 1)}
                  className="w-full"
                />
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showPremiumOnly}
                    onChange={() => setShowPremiumOnly(!showPremiumOnly)}
                    className="rounded border-gray-600 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-gray-300">Premium templates only</span>
                </label>
              </div>
            </div>

            <div className="pt-4 border-t border-gray-700">
              <button
                onClick={() => {
                  setSearchQuery('');
                  setSelectedCategory('all');
                  setSelectedSubcategory('all');
                  setSelectedDifficulty([]);
                  setPriceRange([0, 500]);
                  setShowPremiumOnly(false);
                }}
                className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
              >
                Reset Filters
              </button>
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-xl font-semibold text-white">
                  {filteredTemplates.length} Templates Available
                </h3>
                <p className="text-sm text-gray-400">
                  Browse our collection of high-quality templates
                </p>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <Filter className="w-4 h-4 text-gray-400" />
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500"
                  >
                    <option value="popular">Most Popular</option>
                    <option value="rating">Highest Rated</option>
                    <option value="newest">Newest</option>
                    <option value="price-low">Price: Low to High</option>
                    <option value="price-high">Price: High to Low</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {sortedTemplates.map((template) => (
                <div
                  key={template.id}
                  className="bg-gray-700 rounded-lg border border-gray-600 hover:border-blue-500 transition-colors overflow-hidden"
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h4 className="text-lg font-semibold text-white mb-1">{template.name}</h4>
                        <div className="flex items-center space-x-2 mb-2">
                          <div className="flex items-center">
                            <Star className="w-4 h-4 text-yellow-400 fill-current" />
                            <span className="text-sm text-gray-300 ml-1">{template.rating}</span>
                          </div>
                          <span className="text-gray-500">•</span>
                          <div className="flex items-center">
                            <Download className="w-4 h-4 text-gray-400" />
                            <span className="text-sm text-gray-300 ml-1">{template.downloads}</span>
                          </div>
                        </div>
                      </div>
                      <div>
                        {template.isPremium ? (
                          <div className="px-2 py-1 bg-gradient-to-r from-yellow-600 to-yellow-500 text-white text-xs font-bold rounded">
                            PREMIUM
                          </div>
                        ) : (
                          <div className="px-2 py-1 bg-green-600 text-white text-xs font-bold rounded">
                            FREE
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <p className="text-gray-300 text-sm mb-4 line-clamp-2">{template.description}</p>
                    
                    <div className="flex flex-wrap gap-2 mb-4">
                      {template.tags.slice(0, 3).map((tag, index) => (
                        <span key={index} className="px-2 py-1 bg-gray-600 text-gray-300 text-xs rounded">
                          {tag}
                        </span>
                      ))}
                      {template.tags.length > 3 && (
                        <span className="px-2 py-1 bg-gray-600 text-gray-400 text-xs rounded">
                          +{template.tags.length - 3}
                        </span>
                      )}
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 bg-gray-600 rounded-full flex items-center justify-center text-xs">
                          {template.author.avatar}
                        </div>
                        <span className="text-sm text-gray-400">{template.author.name}</span>
                      </div>
                      <div className="text-sm text-gray-400 flex items-center">
                        <Clock className="w-3 h-3 mr-1" />
                        <span>{template.lastUpdated}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center border-t border-gray-600">
                    <div className="flex-1 px-6 py-3 font-bold text-white">
                      {template.price === 0 ? 'Free' : `$${template.price.toFixed(2)}`}
                    </div>
                    <button
                      onClick={() => setSelectedTemplate(template)}
                      className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors"
                    >
                      Details
                    </button>
                  </div>
                </div>
              ))}
            </div>
            
            {filteredTemplates.length === 0 && (
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Search className="w-8 h-8 text-gray-500" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">No templates found</h3>
                <p className="text-gray-400 mb-6">Try adjusting your filters or search query</p>
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedCategory('all');
                    setSelectedSubcategory('all');
                    setSelectedDifficulty([]);
                    setPriceRange([0, 500]);
                    setShowPremiumOnly(false);
                  }}
                  className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  Reset Filters
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Template Details Modal */}
        {selectedTemplate && (
          <div className="fixed inset-0 bg-black/80 z-60 flex items-center justify-center p-4">
            <div className="bg-gray-800 rounded-lg max-w-4xl w-full p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div className="p-4 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg">
                    <Code className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-white">{selectedTemplate.name}</h3>
                    <div className="flex items-center space-x-3 mt-1">
                      <div className="flex items-center">
                        <Star className="w-4 h-4 text-yellow-400 fill-current" />
                        <span className="text-sm text-gray-300 ml-1">{selectedTemplate.rating}</span>
                      </div>
                      <span className="text-gray-500">•</span>
                      <div className="flex items-center">
                        <Download className="w-4 h-4 text-gray-400" />
                        <span className="text-sm text-gray-300 ml-1">{selectedTemplate.downloads} downloads</span>
                      </div>
                      <span className="text-gray-500">•</span>
                      <div className="flex items-center">
                        <Clock className="w-4 h-4 text-gray-400" />
                        <span className="text-sm text-gray-300 ml-1">Updated {selectedTemplate.lastUpdated}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedTemplate(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="md:col-span-2">
                  <div className="bg-gray-700 rounded-lg p-6 h-full">
                    <h4 className="text-lg font-semibold text-white mb-4">Description</h4>
                    <p className="text-gray-300 mb-6">{selectedTemplate.description}</p>
                    
                    <h4 className="text-lg font-semibold text-white mb-4">Features</h4>
                    <ul className="space-y-2">
                      {selectedTemplate.features.map((feature, index) => (
                        <li key={index} className="flex items-start space-x-2 text-gray-300">
                          <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
                
                <div>
                  <div className="bg-gray-700 rounded-lg p-6">
                    <div className="mb-6">
                      <div className="text-3xl font-bold text-white mb-2">
                        {selectedTemplate.price === 0 ? 'Free' : `$${selectedTemplate.price.toFixed(2)}`}
                      </div>
                      {selectedTemplate.isPremium && (
                        <div className="flex items-center space-x-1 text-yellow-400 text-sm">
                          <Star className="w-4 h-4 fill-current" />
                          <span>Premium Template</span>
                        </div>
                      )}
                    </div>
                    
                    <div className="space-y-4 mb-6">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Category:</span>
                        <span className="text-white capitalize">{selectedTemplate.category}</span>
                      </div>
                      {selectedTemplate.subcategory && (
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Subcategory:</span>
                          <span className="text-white capitalize">{selectedTemplate.subcategory}</span>
                        </div>
                      )}
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Difficulty:</span>
                        <span className="text-white">{selectedTemplate.difficulty}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Author:</span>
                        <span className="text-white">{selectedTemplate.author.name}</span>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <button
                        onClick={() => {
                          onSelectTemplate(selectedTemplate);
                          setSelectedTemplate(null);
                        }}
                        className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
                      >
                        <Download className="w-5 h-5" />
                        <span>{selectedTemplate.price === 0 ? 'Download Free' : 'Purchase & Download'}</span>
                      </button>
                      
                      {selectedTemplate.price > 0 && (
                        <div className="flex items-center justify-center space-x-1 text-xs text-gray-400">
                          <DollarSign className="w-3 h-3" />
                          <span>Secure payment via Stripe</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-gray-700 rounded-lg p-6">
                <h4 className="text-lg font-semibold text-white mb-4">Tags</h4>
                <div className="flex flex-wrap gap-2">
                  {selectedTemplate.tags.map((tag, index) => (
                    <span key={index} className="px-3 py-1 bg-gray-600 text-gray-300 rounded">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TemplateMarketplace;