import React, { useState } from 'react';
import { isSupabaseConfigured } from '../lib/supabase';
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
  Coffee,
  Monitor,
  AlertTriangle,
  GitBranch,
  Workflow
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
    
    if (!email || !password) {
      return;
    }
    
    if (isLoginMode) {
      await onLogin(email, password);
    } else {
      await onSignup(email, password);
    }
  };

  const features = [
    {
      icon: <Code className="w-8 h-8" />,
      title: "Ultra-Fast Analysis Engine",
      description: "Blazing-fast code analysis powered by Rust's zero-cost abstractions. Process millions of lines in seconds with memory-safe performance.",
      color: "from-orange-500 to-red-500"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Enterprise-Grade Security",
      description: "Zero-trust architecture with end-to-end encryption, secure sandboxing, and SOC 2 Type II compliance for all platforms.",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: <Cpu className="w-8 h-8" />,
      title: "Multi-Platform Performance",
      description: "Native performance on Windows, macOS, Linux, and web. WebAssembly compilation for browser-native speed.",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: "Real-Time Security Scanning",
      description: "Continuous vulnerability detection with instant alerts. OWASP Top 10 coverage and custom security rules.",
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: <GitBranch className="w-8 h-8" />,
      title: "GitHub Integration",
      description: "Seamless integration with GitHub repositories. Automated scans on commits, PR checks, and webhook support.",
      color: "from-gray-500 to-slate-500"
    },
    {
      icon: <Workflow className="w-8 h-8" />,
      title: "Zapier Automation",
      description: "Connect with 5000+ apps through Zapier. Automate notifications, tickets, and workflows based on security events.",
      color: "from-orange-500 to-amber-500"
    }
  ];

  const testimonials = [
    {
      name: "Sarah Chen",
      role: "Security Engineer",
      company: "TechCorp",
      avatar: "👩‍💻",
      content: "rustyclint caught 47 critical vulnerabilities in our codebase that other tools missed. The performance is unmatched!"
    },
    {
      name: "Marcus Johnson",
      role: "CTO",
      company: "StartupXYZ",
      avatar: "👨‍💼",
      content: "We reduced our CI/CD pipeline time by 80% with rustyclint's parallel analysis. Security compliance became effortless."
    },
    {
      name: "Elena Rodriguez",
      role: "DevSecOps Lead",
      company: "InnovateLabs",
      avatar: "👩‍🚀",
      content: "The zero-false-positive rate and sub-second analysis times make rustyclint essential for our security-first development."
    }
  ];

  const stats = [
    { value: "10M+", label: "Lines Analyzed/sec" },
    { value: "99.9%", label: "Vulnerability Detection" },
    { value: "99.9%", label: "Uptime" },
    { value: "<100ms", label: "Analysis Time" }
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
            <span className="text-2xl font-bold text-white">rustyclint</span>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className="text-gray-300 hover:text-white transition-colors">Features</a>
            <a href="#pricing" className="text-gray-300 hover:text-white transition-colors">Pricing</a>
            <a href="#about" className="text-gray-300 hover:text-white transition-colors">About</a>
            <button 
              onClick={() => document.getElementById('auth-form')?.scrollIntoView({ behavior: 'smooth' })}
              className="px-6 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg font-medium transition-colors"
            >
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
                  <span className="text-orange-300 text-sm font-medium">✨ Demo Mode - Try it now!</span>
                </div>
                
                <h1 className="text-5xl lg:text-7xl font-bold text-white leading-tight">
                  Build the
                  <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent"> Future </span>
                  of Secure Code
                </h1>
                
                <p className="text-xl text-gray-300 leading-relaxed max-w-2xl">
                  Ultra-fast, enterprise-grade code analysis with military-level security. 
                  Detect vulnerabilities, enforce standards, and ship secure code at lightning speed.
                  <br />
                  <span className="text-orange-300 font-medium">Try the demo with any email and password!</span>
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <button 
                  onClick={() => document.getElementById('auth-form')?.scrollIntoView({ behavior: 'smooth' })}
                  className="flex items-center justify-center space-x-2 px-8 py-4 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 text-white rounded-xl font-semibold text-lg transition-all transform hover:scale-105 shadow-lg"
                >
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
              <div id="auth-form" className="bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-2xl p-8 shadow-2xl">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold text-white mb-2">
                    {isLoginMode ? 'Welcome Back' : 'Join the Revolution'}
                  </h2>
                  <p className="text-gray-400">
                    {isLoginMode ? 'Enter any email and password to try the demo' : 'Enter any email and password to try the demo'}
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
                      <p className="text-xs text-gray-500 mt-1">Demo mode: Use any email (e.g., demo@example.com)</p>
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
                      <p className="text-xs text-gray-500 mt-1">Minimum 6 characters required</p>
                      {!isSupabaseConfigured && (
                        <p className="text-xs text-orange-400 mt-1">Demo mode: Use any email and password</p>
                      )}
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
              <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent"> Secure, Fast Code</span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              From development to deployment, our platform provides military-grade security analysis with unmatched performance across all platforms.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Feature Cards */}
            <div className="space-y-4">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-xl border transition-all cursor-pointer ${
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
                      <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
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
                  <span className="text-gray-400 text-sm ml-4">security-analysis</span>
                </div>
                
                <div className="bg-gray-800 rounded-lg p-4 font-mono text-sm">
                  <div className="text-orange-400">// High-performance security analysis</div>
                  <div className="text-blue-400">use <span className="text-white">rustyclint</span>::<span className="text-green-400">*</span>;</div>
                  <div className="mt-2">
                    <span className="text-purple-400">fn</span> <span className="text-yellow-400">main</span>() {'{'}
                  </div>
                  <div className="ml-4">
                    <span className="text-blue-400">let</span> <span className="text-white">analyzer</span> = <span className="text-green-400">SecurityAnalyzer</span>::<span className="text-yellow-400">new</span>();
                  </div>
                  <div className="ml-4">
                    <span className="text-white">analyzer</span>.<span className="text-yellow-400">scan_vulnerabilities</span>();
                  </div>
                  <div>{'}'}</div>
                </div>
                
                <div className="mt-4 flex items-center space-x-4">
                  <div className="flex items-center space-x-2 text-green-400">
                    <CheckCircle className="w-4 h-4" />
                    <span className="text-sm">47 vulnerabilities detected & fixed</span>
                  </div>
                  <div className="text-gray-400 text-sm">0.08s</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Performance & Security Stats */}
      <section className="px-6 py-16 bg-gray-800/50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-6">
              Unmatched
              <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent"> Performance & Security</span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Built with Rust for maximum performance and memory safety. Trusted by Fortune 500 companies worldwide.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
            <div className="bg-gray-700/50 backdrop-blur-xl border border-gray-600/50 rounded-xl p-6 text-center">
              <div className="text-4xl font-bold text-orange-400 mb-2">10M+</div>
              <div className="text-gray-300 mb-2">Lines/Second</div>
              <div className="text-xs text-gray-500">Parallel analysis engine</div>
            </div>
            <div className="bg-gray-700/50 backdrop-blur-xl border border-gray-600/50 rounded-xl p-6 text-center">
              <div className="text-4xl font-bold text-green-400 mb-2">99.97%</div>
              <div className="text-gray-300 mb-2">Accuracy Rate</div>
              <div className="text-xs text-gray-500">Zero false positives</div>
            </div>
            <div className="bg-gray-700/50 backdrop-blur-xl border border-gray-600/50 rounded-xl p-6 text-center">
              <div className="text-4xl font-bold text-blue-400 mb-2">&lt;50ms</div>
              <div className="text-gray-300 mb-2">Response Time</div>
              <div className="text-xs text-gray-500">Real-time feedback</div>
            </div>
            <div className="bg-gray-700/50 backdrop-blur-xl border border-gray-600/50 rounded-xl p-6 text-center">
              <div className="text-4xl font-bold text-purple-400 mb-2">256-bit</div>
              <div className="text-gray-300 mb-2">Encryption</div>
              <div className="text-xs text-gray-500">End-to-end security</div>
            </div>
          </div>

          {/* Platform Support */}
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-gray-700/30 rounded-xl p-6 border border-gray-600/30">
              <div className="flex items-center space-x-3 mb-4">
                <Monitor className="w-8 h-8 text-blue-400" />
                <h3 className="text-xl font-semibold text-white">Desktop Native</h3>
              </div>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Windows, macOS, Linux</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Native performance</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Offline capability</span>
                </li>
              </ul>
            </div>

            <div className="bg-gray-700/30 rounded-xl p-6 border border-gray-600/30">
              <div className="flex items-center space-x-3 mb-4">
                <Globe className="w-8 h-8 text-green-400" />
                <h3 className="text-xl font-semibold text-white">Web Platform</h3>
              </div>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>WebAssembly powered</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Browser-native speed</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Zero installation</span>
                </li>
              </ul>
            </div>

            <div className="bg-gray-700/30 rounded-xl p-6 border border-gray-600/30">
              <div className="flex items-center space-x-3 mb-4">
                <Cloud className="w-8 h-8 text-purple-400" />
                <h3 className="text-xl font-semibold text-white">Cloud Scale</h3>
              </div>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Auto-scaling infrastructure</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Global edge deployment</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>99.99% uptime SLA</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Integrations Section */}
      <section className="px-6 py-16">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-6">
              Seamless
              <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent"> Integrations</span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Connect rustyclint with your existing development workflow. GitHub, databases, and automation tools work together seamlessly.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mb-16">
            {/* GitHub Integration */}
            <div className="bg-gray-700/30 rounded-xl p-6 border border-gray-600/30">
              <div className="flex items-center space-x-3 mb-4">
                <Github className="w-8 h-8 text-white" />
                <h3 className="text-xl font-semibold text-white">GitHub</h3>
              </div>
              <p className="text-gray-300 mb-4">
                Automated security scans on every commit. PR checks, webhook integration, and repository monitoring.
              </p>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Automatic PR security checks</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Webhook-triggered scans</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Repository vulnerability tracking</span>
                </li>
              </ul>
            </div>

            {/* Supabase Integration */}
            <div className="bg-gray-700/30 rounded-xl p-6 border border-gray-600/30">
              <div className="flex items-center space-x-3 mb-4">
                <Database className="w-8 h-8 text-green-400" />
                <h3 className="text-xl font-semibold text-white">Supabase</h3>
              </div>
              <p className="text-gray-300 mb-4">
                Real-time database for scan results, analytics, and user data. Secure, scalable, and lightning-fast.
              </p>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Real-time vulnerability tracking</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Advanced analytics & reporting</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Row-level security & compliance</span>
                </li>
              </ul>
            </div>

            {/* Zapier Integration */}
            <div className="bg-gray-700/30 rounded-xl p-6 border border-gray-600/30">
              <div className="flex items-center space-x-3 mb-4">
                <Zap className="w-8 h-8 text-orange-400" />
                <h3 className="text-xl font-semibold text-white">Zapier</h3>
              </div>
              <p className="text-gray-300 mb-4">
                Connect with 5000+ apps. Automate notifications, create tickets, and trigger workflows based on security events.
              </p>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Slack/Teams notifications</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Jira/Trello ticket creation</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Custom workflow automation</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Integration Flow Diagram */}
          <div className="bg-gray-800/50 rounded-2xl p-8 border border-gray-700">
            <h3 className="text-xl font-semibold text-white mb-6 text-center">Automated Security Workflow</h3>
            <div className="flex items-center justify-between">
              <div className="flex flex-col items-center space-y-2">
                <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center">
                  <Github className="w-8 h-8 text-white" />
                </div>
                <span className="text-sm text-gray-300">Code Push</span>
              </div>
              <div className="flex-1 h-px bg-gradient-to-r from-gray-600 to-orange-500 mx-4" />
              <div className="flex flex-col items-center space-y-2">
                <div className="w-16 h-16 bg-orange-600 rounded-full flex items-center justify-center">
                  <Shield className="w-8 h-8 text-white" />
                </div>
                <span className="text-sm text-gray-300">Security Scan</span>
              </div>
              <div className="flex-1 h-px bg-gradient-to-r from-orange-500 to-green-500 mx-4" />
              <div className="flex flex-col items-center space-y-2">
                <div className="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center">
                  <Database className="w-8 h-8 text-white" />
                </div>
                <span className="text-sm text-gray-300">Store Results</span>
              </div>
              <div className="flex-1 h-px bg-gradient-to-r from-green-500 to-blue-500 mx-4" />
              <div className="flex flex-col items-center space-y-2">
                <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center">
                  <Zap className="w-8 h-8 text-white" />
                </div>
                <span className="text-sm text-gray-300">Notify Team</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Security Features */}
      <section className="px-6 py-16">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-6">
              Enterprise
              <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent"> Security</span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Military-grade security with zero-trust architecture. Compliant with SOC 2, GDPR, and industry standards.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="space-y-6">
              <div className="flex items-start space-x-4">
                <div className="p-3 bg-red-600/20 rounded-lg">
                  <Shield className="w-6 h-6 text-red-400" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white mb-2">Zero-Trust Architecture</h3>
                  <p className="text-gray-300">Every request verified, encrypted, and logged. No implicit trust, maximum security.</p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="p-3 bg-blue-600/20 rounded-lg">
                  <Lock className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white mb-2">End-to-End Encryption</h3>
                  <p className="text-gray-300">AES-256 encryption for data at rest and in transit. Your code never leaves your control.</p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="p-3 bg-green-600/20 rounded-lg">
                  <Database className="w-6 h-6 text-green-400" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white mb-2">Secure Sandboxing</h3>
                  <p className="text-gray-300">Isolated execution environments with memory-safe Rust runtime. Zero code injection risk.</p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="p-3 bg-purple-600/20 rounded-lg">
                  <Award className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white mb-2">Compliance Ready</h3>
                  <p className="text-gray-300">SOC 2 Type II, GDPR, HIPAA, and PCI DSS compliant. Audit trails included.</p>
                </div>
              </div>
            </div>

            <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold text-white mb-4">Security Scan Results</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-red-900/20 rounded-lg border border-red-500/30">
                  <div className="flex items-center space-x-3">
                    <AlertTriangle className="w-5 h-5 text-red-400" />
                    <span className="text-white">Critical Vulnerabilities</span>
                  </div>
                  <span className="text-red-400 font-bold">0</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                  <div className="flex items-center space-x-3">
                    <AlertTriangle className="w-5 h-5 text-yellow-400" />
                    <span className="text-white">High Risk Issues</span>
                  </div>
                  <span className="text-yellow-400 font-bold">2</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-green-900/20 rounded-lg border border-green-500/30">
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="text-white">Security Score</span>
                  </div>
                  <span className="text-green-400 font-bold">98/100</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-blue-900/20 rounded-lg border border-blue-500/30">
                  <div className="flex items-center space-x-3">
                    <Zap className="w-5 h-5 text-blue-400" />
                    <span className="text-white">Scan Time</span>
                  </div>
                  <span className="text-blue-400 font-bold">0.08s</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="px-6 py-16">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-6">
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
      <section className="px-6 py-16 bg-gradient-to-r from-orange-600/10 to-red-600/10">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl lg:text-4xl font-bold text-white mb-6">
            Ready to Secure Your Code?
          </h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Join thousands of developers who trust rustyclint for enterprise-grade security and performance.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button 
              onClick={() => document.getElementById('auth-form')?.scrollIntoView({ behavior: 'smooth' })}
              className="flex items-center justify-center space-x-2 px-8 py-4 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 text-white rounded-xl font-semibold text-lg transition-all transform hover:scale-105 shadow-lg"
            >
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
                <span className="text-xl font-bold text-white">rustyclint</span>
              </div>
              <p className="text-gray-400">
                The future of secure code analysis. Fast, secure, and reliable.
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
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Security Features</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Performance</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Pricing</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Compliance</a>
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
              © 2024 rustyclint. All rights reserved.
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