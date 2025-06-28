import React, { useState } from 'react';
import { 
  Code, 
  Smartphone, 
  Zap, 
  Shield, 
  Users, 
  Star, 
  ArrowRight, 
  Play, 
  CheckCircle, 
  Github, 
  Twitter, 
  Mail,
  Eye,
  EyeOff,
  Loader2,
  Rocket,
  Globe,
  Database,
  Cloud,
  Terminal,
  Cpu,
  Lock,
  Heart,
  Award,
  TrendingUp,
  Coffee
} from 'lucide-react';

interface LandingPageProps {
  onLogin: (email: string, password: string) => Promise<void>;
  onSignup: (email: string, password: string) => Promise<void>;
  loading: boolean;
  error: string | null;
}

const LandingPage: React.FC<LandingPageProps> = ({ onLogin, onSignup, loading, error }) => {
  const [isLoginMode, setIsLoginMode] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [activeFeature, setActiveFeature] = useState(0);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isLoginMode) {
      await onLogin(email, password);
    } else {
      await onSignup(email, password);
    }
  };

  const features = [
    {
      icon: <Code className="w-8 h-8" />,
      title: "Multi-Language Support",
      description: "Write in Rust, Dart, Kotlin, Swift, and more with intelligent syntax highlighting and auto-completion.",
      color: "from-orange-500 to-red-500"
    },
    {
      icon: <Smartphone className="w-8 h-8" />,
      title: "Cross-Platform Development",
      description: "Build for Android, iOS, Flutter, React Native, and desktop from a single codebase.",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: "Real-Time Collaboration",
      description: "Work together with your team in real-time with live cursors, chat, and voice calls.",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: <Database className="w-8 h-8" />,
      title: "AI-Powered Development",
      description: "Leverage machine learning and AI tools for code generation, debugging, and optimization.",
      color: "from-green-500 to-emerald-500"
    }
  ];

  const testimonials = [
    {
      name: "Sarah Chen",
      role: "Senior Mobile Developer",
      company: "TechCorp",
      avatar: "👩‍💻",
      content: "Rust Cloud IDE has revolutionized our mobile development workflow. The cross-platform capabilities are incredible!"
    },
    {
      name: "Marcus Johnson",
      role: "CTO",
      company: "StartupXYZ",
      avatar: "👨‍💼",
      content: "The real-time collaboration features have made our remote team 3x more productive. Absolutely game-changing."
    },
    {
      name: "Elena Rodriguez",
      role: "Full-Stack Engineer",
      company: "InnovateLabs",
      avatar: "👩‍🚀",
      content: "From prototype to production in record time. The AI assistance is like having a senior developer on your team."
    }
  ];

  const stats = [
    { value: "50K+", label: "Developers" },
    { value: "1M+", label: "Projects Built" },
    { value: "99.9%", label: "Uptime" },
    { value: "24/7", label: "Support" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Navigation */}
      <nav className="relative z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-red-500 rounded-xl flex items-center justify-center">
              <Code className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-white">Rust Cloud IDE</span>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className="text-gray-300 hover:text-white transition-colors">Features</a>
            <a href="#pricing" className="text-gray-300 hover:text-white transition-colors">Pricing</a>
            <a href="#about" className="text-gray-300 hover:text-white transition-colors">About</a>
            <button className="px-6 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg font-medium transition-colors">
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Column - Content */}
            <div className="space-y-8">
              <div className="space-y-6">
                <div className="inline-flex items-center space-x-2 bg-orange-500/10 border border-orange-500/20 rounded-full px-4 py-2">
                  <Rocket className="w-4 h-4 text-orange-400" />
                  <span className="text-orange-300 text-sm font-medium">Now with AI-powered development</span>
                </div>
                
                <h1 className="text-5xl lg:text-7xl font-bold text-white leading-tight">
                  Build the
                  <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent"> Future </span>
                  of Mobile Apps
                </h1>
                
                <p className="text-xl text-gray-300 leading-relaxed max-w-2xl">
                  The most powerful cloud-based IDE for cross-platform mobile development. 
                  Write once in Rust, deploy everywhere with native performance.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <button className="flex items-center justify-center space-x-2 px-8 py-4 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 text-white rounded-xl font-semibold text-lg transition-all transform hover:scale-105 shadow-lg">
                  <Play className="w-5 h-5" />
                  <span>Start Building</span>
                </button>
                <button className="flex items-center justify-center space-x-2 px-8 py-4 border border-gray-600 hover:border-gray-500 text-gray-300 hover:text-white rounded-xl font-semibold text-lg transition-colors">
                  <Eye className="w-5 h-5" />
                  <span>Watch Demo</span>
                </button>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 pt-8">
                {stats.map((stat, index) => (
                  <div key={index} className="text-center">
                    <div className="text-3xl font-bold text-white">{stat.value}</div>
                    <div className="text-gray-400 text-sm">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Column - Login Form */}
            <div className="relative">
              <div className="bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-2xl p-8 shadow-2xl">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold text-white mb-2">
                    {isLoginMode ? 'Welcome Back' : 'Join the Revolution'}
                  </h2>
                  <p className="text-gray-400">
                    {isLoginMode ? 'Sign in to your account' : 'Create your free account'}
                  </p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                  {error && (
                    <div className="bg-red-900/20 border border-red-500/50 text-red-300 px-4 py-3 rounded-lg text-sm">
                      {error}
                    </div>
                  )}

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Email Address
                      </label>
                      <div className="relative">
                        <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <input
                          type="email"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          required
                          className="w-full pl-10 pr-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-500/20 transition-colors"
                          placeholder="Enter your email"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Password
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <input
                          type={showPassword ? 'text' : 'password'}
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          required
                          className="w-full pl-10 pr-12 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-500/20 transition-colors"
                          placeholder="Enter your password"
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300 transition-colors"
                        >
                          {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                        </button>
                      </div>
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-all transform hover:scale-105"
                  >
                    {loading ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <>
                        <span>{isLoginMode ? 'Sign In' : 'Create Account'}</span>
                        <ArrowRight className="w-5 h-5" />
                      </>
                    )}
                  </button>
                </form>

                <div className="mt-6 text-center">
                  <p className="text-gray-400">
                    {isLoginMode ? "Don't have an account?" : "Already have an account?"}
                    <button
                      onClick={() => setIsLoginMode(!isLoginMode)}
                      className="ml-2 text-orange-400 hover:text-orange-300 font-medium transition-colors"
                    >
                      {isLoginMode ? 'Sign up' : 'Sign in'}
                    </button>
                  </p>
                </div>

                {/* Social Proof */}
                <div className="mt-8 pt-6 border-t border-gray-700">
                  <div className="flex items-center justify-center space-x-4 text-sm text-gray-400">
                    <div className="flex items-center space-x-1">
                      <Shield className="w-4 h-4 text-green-400" />
                      <span>Enterprise Security</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Globe className="w-4 h-4 text-blue-400" />
                      <span>Global CDN</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Floating Elements */}
              <div className="absolute -top-4 -right-4 w-20 h-20 bg-gradient-to-br from-orange-500/20 to-red-500/20 rounded-full blur-xl"></div>
              <div className="absolute -bottom-4 -left-4 w-16 h-16 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full blur-xl"></div>
            </div>
          </div>
        </div>

        {/* Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-gradient-to-br from-orange-500/10 to-red-500/10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-full blur-3xl"></div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="px-6 py-20 bg-gray-800/30">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
              Everything You Need to Build
              <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent"> Amazing Apps</span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              From idea to production, our platform provides all the tools and services you need to create world-class mobile applications.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Feature Cards */}
            <div className="space-y-6">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className={`p-6 rounded-xl border transition-all cursor-pointer ${
                    activeFeature === index
                      ? 'bg-gray-700/50 border-orange-500/50'
                      : 'bg-gray-800/30 border-gray-700/50 hover:border-gray-600/50'
                  }`}
                  onClick={() => setActiveFeature(index)}
                >
                  <div className="flex items-start space-x-4">
                    <div className={`p-3 rounded-lg bg-gradient-to-br ${feature.color}`}>
                      {feature.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                      <p className="text-gray-300">{feature.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Feature Preview */}
            <div className="relative">
              <div className="bg-gray-900 rounded-2xl p-6 border border-gray-700 shadow-2xl">
                <div className="flex items-center space-x-2 mb-4">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="text-gray-400 text-sm ml-4">rust-mobile-app</span>
                </div>
                
                <div className="bg-gray-800 rounded-lg p-4 font-mono text-sm">
                  <div className="text-orange-400">// Cross-platform mobile development</div>
                  <div className="text-blue-400">use <span className="text-white">mobile_framework</span>::<span className="text-green-400">*</span>;</div>
                  <div className="mt-2">
                    <span className="text-purple-400">fn</span> <span className="text-yellow-400">main</span>() {'{'}
                  </div>
                  <div className="ml-4">
                    <span className="text-blue-400">let</span> <span className="text-white">app</span> = <span className="text-green-400">MobileApp</span>::<span className="text-yellow-400">new</span>();
                  </div>
                  <div className="ml-4">
                    <span className="text-white">app</span>.<span className="text-yellow-400">build_for_all_platforms</span>();
                  </div>
                  <div>{'}'}</div>
                </div>
                
                <div className="mt-4 flex items-center space-x-4">
                  <div className="flex items-center space-x-2 text-green-400">
                    <CheckCircle className="w-4 h-4" />
                    <span className="text-sm">Build successful</span>
                  </div>
                  <div className="text-gray-400 text-sm">2.3s</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
              Loved by Developers
              <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent"> Worldwide</span>
            </h2>
            <div className="flex items-center justify-center space-x-1 mb-4">
              {[...Array(5)].map((_, i) => (
                <Star key={i} className="w-6 h-6 text-yellow-400 fill-current" />
              ))}
              <span className="text-gray-300 ml-2">4.9/5 from 10,000+ reviews</span>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl p-6">
                <div className="flex items-center space-x-4 mb-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-500 rounded-full flex items-center justify-center text-xl">
                    {testimonial.avatar}
                  </div>
                  <div>
                    <div className="font-semibold text-white">{testimonial.name}</div>
                    <div className="text-gray-400 text-sm">{testimonial.role} at {testimonial.company}</div>
                  </div>
                </div>
                <p className="text-gray-300 italic">"{testimonial.content}"</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-6 py-20 bg-gradient-to-r from-orange-600/10 to-red-600/10">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
            Ready to Build the Future?
          </h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Join thousands of developers who are already building amazing mobile apps with Rust Cloud IDE.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="flex items-center justify-center space-x-2 px-8 py-4 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 text-white rounded-xl font-semibold text-lg transition-all transform hover:scale-105 shadow-lg">
              <Rocket className="w-5 h-5" />
              <span>Start Free Trial</span>
            </button>
            <button className="flex items-center justify-center space-x-2 px-8 py-4 border border-gray-600 hover:border-gray-500 text-gray-300 hover:text-white rounded-xl font-semibold text-lg transition-colors">
              <Github className="w-5 h-5" />
              <span>View on GitHub</span>
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-12 bg-gray-900 border-t border-gray-800">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-orange-500 to-red-500 rounded-lg flex items-center justify-center">
                  <Code className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold text-white">Rust Cloud IDE</span>
              </div>
              <p className="text-gray-400">
                The future of mobile development is here. Build once, deploy everywhere.
              </p>
              <div className="flex items-center space-x-4">
                <Github className="w-5 h-5 text-gray-400 hover:text-white cursor-pointer transition-colors" />
                <Twitter className="w-5 h-5 text-gray-400 hover:text-white cursor-pointer transition-colors" />
                <Mail className="w-5 h-5 text-gray-400 hover:text-white cursor-pointer transition-colors" />
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold text-white mb-4">Product</h3>
              <div className="space-y-2">
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Features</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Pricing</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Documentation</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">API Reference</a>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold text-white mb-4">Company</h3>
              <div className="space-y-2">
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">About</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Blog</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Careers</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Contact</a>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold text-white mb-4">Support</h3>
              <div className="space-y-2">
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Help Center</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Community</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Status</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Security</a>
              </div>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-sm">
              © 2024 Rust Cloud IDE. All rights reserved.
            </p>
            <div className="flex items-center space-x-6 mt-4 md:mt-0">
              <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors">Privacy Policy</a>
              <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors">Terms of Service</a>
              <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors">Cookie Policy</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;