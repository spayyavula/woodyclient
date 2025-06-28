import React, { useState } from 'react';
import { 
  X, 
  Search, 
  Filter, 
  Star, 
  Clock, 
  Code, 
  Smartphone, 
  Tablet, 
  Monitor,
  Zap,
  Shield,
  Database,
  Camera,
  MapPin,
  ShoppingCart,
  Heart,
  MessageCircle,
  Music,
  Gamepad2,
  TrendingUp,
  Users,
  Calendar,
  FileText,
  Settings,
  Globe,
  Cpu,
  Layers,
  Palette,
  Target,
  Briefcase,
  BookOpen,
  Coffee,
  Headphones,
  Video,
  Lock,
  Wallet,
  Activity,
  Cloud,
  Wifi,
  Battery,
  Bell,
  Image,
  Play,
  Download
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
  platforms: ('iOS' | 'Android' | 'Flutter' | 'React Native' | 'Web')[];
  rating: number;
  downloads: number;
}

interface ProjectTemplatesProps {
  isVisible: boolean;
  onClose: () => void;
  onSelectTemplate: (template: Template) => void;
}

const ProjectTemplates: React.FC<ProjectTemplatesProps> = ({ isVisible, onClose, onSelectTemplate }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState('all');
  const [selectedPlatform, setSelectedPlatform] = useState('all');

  const templates: Template[] = [
    // Native iOS Templates
    {
      id: 'ios-swift-social',
      name: 'iOS Social Media App',
      description: 'Native iOS social media app with real-time messaging, photo sharing, and user profiles using Swift and UIKit',
      category: 'Social',
      difficulty: 'Advanced',
      tags: ['Swift', 'UIKit', 'Core Data', 'CloudKit', 'Camera', 'Real-time'],
      icon: <MessageCircle className="w-8 h-8 text-blue-400" />,
      estimatedTime: '3-4 weeks',
      platforms: ['iOS'],
      rating: 4.8,
      downloads: 2847,
      files: {
        'AppDelegate.swift': {
          content: `import UIKit
import CloudKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Configure CloudKit
        setupCloudKit()
        
        // Setup navigation appearance
        setupNavigationAppearance()
        
        return true
    }
    
    private func setupCloudKit() {
        // CloudKit configuration for real-time sync
        let container = CKContainer.default()
        container.accountStatus { status, error in
            DispatchQueue.main.async {
                switch status {
                case .available:
                    print("CloudKit available")
                case .noAccount:
                    print("No iCloud account")
                default:
                    print("CloudKit status: \\(status)")
                }
            }
        }
    }
    
    private func setupNavigationAppearance() {
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor.systemBlue
        appearance.titleTextAttributes = [.foregroundColor: UIColor.white]
        
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }
}`,
          language: 'swift'
        },
        'FeedViewController.swift': {
          content: `import UIKit
import CloudKit

class FeedViewController: UIViewController {
    @IBOutlet weak var tableView: UITableView!
    
    private var posts: [Post] = []
    private let refreshControl = UIRefreshControl()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadPosts()
    }
    
    private func setupUI() {
        title = "Feed"
        
        // Setup table view
        tableView.delegate = self
        tableView.dataSource = self
        tableView.register(PostTableViewCell.self, forCellReuseIdentifier: "PostCell")
        
        // Setup refresh control
        refreshControl.addTarget(self, action: #selector(refreshPosts), for: .valueChanged)
        tableView.refreshControl = refreshControl
        
        // Add compose button
        navigationItem.rightBarButtonItem = UIBarButtonItem(
            barButtonSystemItem: .compose,
            target: self,
            action: #selector(composePost)
        )
    }
    
    @objc private func refreshPosts() {
        loadPosts()
    }
    
    @objc private func composePost() {
        let composeVC = ComposeViewController()
        let navController = UINavigationController(rootViewController: composeVC)
        present(navController, animated: true)
    }
    
    private func loadPosts() {
        // CloudKit query for posts
        let predicate = NSPredicate(value: true)
        let query = CKQuery(recordType: "Post", predicate: predicate)
        query.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        
        CKContainer.default().publicCloudDatabase.perform(query, inZoneWith: nil) { records, error in
            DispatchQueue.main.async {
                self.refreshControl.endRefreshing()
                
                if let records = records {
                    self.posts = records.compactMap { Post(record: $0) }
                    self.tableView.reloadData()
                }
            }
        }
    }
}

extension FeedViewController: UITableViewDataSource, UITableViewDelegate {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return posts.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "PostCell", for: indexPath) as! PostTableViewCell
        cell.configure(with: posts[indexPath.row])
        return cell
    }
}`,
          language: 'swift'
        },
        'Post.swift': {
          content: `import Foundation
import CloudKit

struct Post {
    let id: String
    let authorName: String
    let content: String
    let imageURL: String?
    let createdAt: Date
    let likesCount: Int
    
    init?(record: CKRecord) {
        guard let authorName = record["authorName"] as? String,
              let content = record["content"] as? String else {
            return nil
        }
        
        self.id = record.recordID.recordName
        self.authorName = authorName
        self.content = content
        self.imageURL = record["imageURL"] as? String
        self.createdAt = record.creationDate ?? Date()
        self.likesCount = record["likesCount"] as? Int ?? 0
    }
}`,
          language: 'swift'
        }
      },
      features: [
        'Real-time messaging with CloudKit',
        'Photo sharing with camera integration',
        'User profiles and authentication',
        'Push notifications',
        'Offline data synchronization',
        'Social features (likes, comments)',
        'Custom UI animations',
        'Dark mode support'
      ],
      useCase: 'Build a complete social media platform for iOS with native performance and Apple ecosystem integration',
      techStack: ['Swift', 'UIKit', 'CloudKit', 'Core Data', 'AVFoundation', 'UserNotifications']
    },

    {
      id: 'ios-swiftui-finance',
      name: 'iOS Finance Tracker',
      description: 'Modern SwiftUI finance app with expense tracking, budgeting, and investment portfolio management',
      category: 'Finance',
      difficulty: 'Intermediate',
      tags: ['SwiftUI', 'Core Data', 'Charts', 'Biometrics', 'Widgets'],
      icon: <TrendingUp className="w-8 h-8 text-green-400" />,
      estimatedTime: '2-3 weeks',
      platforms: ['iOS'],
      rating: 4.9,
      downloads: 3421,
      files: {
        'ContentView.swift': {
          content: `import SwiftUI
import Charts

struct ContentView: View {
    @StateObject private var financeManager = FinanceManager()
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView()
                .tabItem {
                    Image(systemName: "chart.pie.fill")
                    Text("Dashboard")
                }
                .tag(0)
            
            ExpensesView()
                .tabItem {
                    Image(systemName: "creditcard.fill")
                    Text("Expenses")
                }
                .tag(1)
            
            BudgetView()
                .tabItem {
                    Image(systemName: "target")
                    Text("Budget")
                }
                .tag(2)
            
            InvestmentsView()
                .tabItem {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                    Text("Investments")
                }
                .tag(3)
        }
        .environmentObject(financeManager)
        .onAppear {
            financeManager.loadData()
        }
    }
}

struct DashboardView: View {
    @EnvironmentObject var financeManager: FinanceManager
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Balance Card
                    BalanceCardView(balance: financeManager.totalBalance)
                    
                    // Spending Chart
                    SpendingChartView(expenses: financeManager.recentExpenses)
                    
                    // Quick Actions
                    QuickActionsView()
                    
                    // Recent Transactions
                    RecentTransactionsView(transactions: financeManager.recentTransactions)
                }
                .padding()
            }
            .navigationTitle("Finance Tracker")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        // Add transaction
                    }) {
                        Image(systemName: "plus.circle.fill")
                            .foregroundColor(.blue)
                    }
                }
            }
        }
    }
}

struct BalanceCardView: View {
    let balance: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Total Balance")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("$\\(balance, specifier: "%.2f")")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.primary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
        )
    }
}`,
          language: 'swift'
        },
        'FinanceManager.swift': {
          content: `import Foundation
import CoreData
import Combine

class FinanceManager: ObservableObject {
    @Published var totalBalance: Double = 0
    @Published var recentExpenses: [Expense] = []
    @Published var recentTransactions: [Transaction] = []
    @Published var budgets: [Budget] = []
    @Published var investments: [Investment] = []
    
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        setupDataObservers()
    }
    
    func loadData() {
        loadBalance()
        loadExpenses()
        loadTransactions()
        loadBudgets()
        loadInvestments()
    }
    
    private func setupDataObservers() {
        // Observe changes and update UI
        $recentExpenses
            .sink { [weak self] expenses in
                self?.calculateTotalBalance()
            }
            .store(in: &cancellables)
    }
    
    private func loadBalance() {
        // Calculate total balance from all accounts
        totalBalance = 15420.50 // Mock data
    }
    
    private func loadExpenses() {
        // Load recent expenses from Core Data
        recentExpenses = [
            Expense(id: UUID(), amount: 45.99, category: "Food", date: Date(), description: "Grocery shopping"),
            Expense(id: UUID(), amount: 12.50, category: "Transport", date: Date().addingTimeInterval(-86400), description: "Uber ride"),
            Expense(id: UUID(), amount: 89.99, category: "Shopping", date: Date().addingTimeInterval(-172800), description: "Clothing")
        ]
    }
    
    private func loadTransactions() {
        // Load recent transactions
        recentTransactions = [
            Transaction(id: UUID(), amount: -45.99, description: "Grocery Store", date: Date(), category: "Food"),
            Transaction(id: UUID(), amount: 2500.00, description: "Salary Deposit", date: Date().addingTimeInterval(-86400), category: "Income"),
            Transaction(id: UUID(), amount: -12.50, description: "Uber", date: Date().addingTimeInterval(-172800), category: "Transport")
        ]
    }
    
    private func loadBudgets() {
        // Load budget data
        budgets = [
            Budget(id: UUID(), category: "Food", limit: 500, spent: 245.50, period: .monthly),
            Budget(id: UUID(), category: "Transport", limit: 200, spent: 89.25, period: .monthly),
            Budget(id: UUID(), category: "Entertainment", limit: 150, spent: 67.80, period: .monthly)
        ]
    }
    
    private func loadInvestments() {
        // Load investment portfolio
        investments = [
            Investment(id: UUID(), symbol: "AAPL", name: "Apple Inc.", shares: 10, currentPrice: 175.50, totalValue: 1755.00),
            Investment(id: UUID(), symbol: "GOOGL", name: "Alphabet Inc.", shares: 5, currentPrice: 2750.25, totalValue: 13751.25),
            Investment(id: UUID(), symbol: "TSLA", name: "Tesla Inc.", shares: 8, currentPrice: 245.80, totalValue: 1966.40)
        ]
    }
    
    private func calculateTotalBalance() {
        // Recalculate total balance based on transactions
        let totalExpenses = recentExpenses.reduce(0) { $0 + $1.amount }
        // Update balance calculation logic
    }
}`,
          language: 'swift'
        }
      },
      features: [
        'SwiftUI modern interface',
        'Expense tracking and categorization',
        'Budget management with alerts',
        'Investment portfolio tracking',
        'Biometric authentication',
        'Home screen widgets',
        'Charts and analytics',
        'Core Data persistence'
      ],
      useCase: 'Personal finance management with comprehensive tracking, budgeting, and investment features',
      techStack: ['SwiftUI', 'Core Data', 'Charts', 'LocalAuthentication', 'WidgetKit', 'Combine']
    },

    // Native Android Templates
    {
      id: 'android-kotlin-ecommerce',
      name: 'Android E-commerce App',
      description: 'Modern Android e-commerce app with Jetpack Compose, product catalog, shopping cart, and payment integration',
      category: 'E-commerce',
      difficulty: 'Advanced',
      tags: ['Kotlin', 'Jetpack Compose', 'Room', 'Retrofit', 'Payment'],
      icon: <ShoppingCart className="w-8 h-8 text-green-400" />,
      estimatedTime: '4-5 weeks',
      platforms: ['Android'],
      rating: 4.7,
      downloads: 4156,
      files: {
        'MainActivity.kt': {
          content: `package com.rustyclint.ecommerce

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.rustyclint.ecommerce.ui.theme.EcommerceTheme
import com.rustyclint.ecommerce.ui.screens.*
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            EcommerceTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val navController = rememberNavController()
                    
                    NavHost(
                        navController = navController,
                        startDestination = "home"
                    ) {
                        composable("home") {
                            HomeScreen(
                                navController = navController,
                                viewModel = hiltViewModel()
                            )
                        }
                        composable("product/{productId}") { backStackEntry ->
                            val productId = backStackEntry.arguments?.getString("productId")
                            ProductDetailScreen(
                                productId = productId,
                                navController = navController,
                                viewModel = hiltViewModel()
                            )
                        }
                        composable("cart") {
                            CartScreen(
                                navController = navController,
                                viewModel = hiltViewModel()
                            )
                        }
                        composable("checkout") {
                            CheckoutScreen(
                                navController = navController,
                                viewModel = hiltViewModel()
                            )
                        }
                        composable("profile") {
                            ProfileScreen(
                                navController = navController,
                                viewModel = hiltViewModel()
                            )
                        }
                    }
                }
            }
        }
    }
}`,
          language: 'kotlin'
        },
        'HomeScreen.kt': {
          content: `package com.rustyclint.ecommerce.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import coil.compose.AsyncImage
import com.rustyclint.ecommerce.data.model.Product
import com.rustyclint.ecommerce.ui.components.ProductCard
import com.rustyclint.ecommerce.ui.components.CategoryChip
import com.rustyclint.ecommerce.viewmodel.HomeViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    navController: NavController,
    viewModel: HomeViewModel
) {
    val uiState by viewModel.uiState.collectAsState()
    var searchQuery by remember { mutableStateOf("") }
    
    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        // Top App Bar
        TopAppBar(
            title = { Text("RustyClint Store") },
            actions = {
                IconButton(onClick = { navController.navigate("cart") }) {
                    BadgedBox(
                        badge = {
                            if (uiState.cartItemCount > 0) {
                                Badge { Text(uiState.cartItemCount.toString()) }
                            }
                        }
                    ) {
                        Icon(Icons.Default.ShoppingCart, contentDescription = "Cart")
                    }
                }
            }
        )
        
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Search Bar
            item {
                OutlinedTextField(
                    value = searchQuery,
                    onValueChange = { 
                        searchQuery = it
                        viewModel.searchProducts(it)
                    },
                    modifier = Modifier.fillMaxWidth(),
                    placeholder = { Text("Search products...") },
                    leadingIcon = {
                        Icon(Icons.Default.Search, contentDescription = "Search")
                    },
                    singleLine = true
                )
            }
            
            // Categories
            item {
                Text(
                    text = "Categories",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(uiState.categories) { category ->
                        CategoryChip(
                            category = category,
                            isSelected = category == uiState.selectedCategory,
                            onClick = { viewModel.selectCategory(category) }
                        )
                    }
                }
            }
            
            // Featured Products
            item {
                Text(
                    text = "Featured Products",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
            }
            
            // Products Grid
            items(uiState.products.chunked(2)) { productPair ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    productPair.forEach { product ->
                        ProductCard(
                            product = product,
                            modifier = Modifier.weight(1f),
                            onClick = { 
                                navController.navigate("product/\${product.id}")
                            },
                            onAddToCart = { 
                                viewModel.addToCart(product)
                            }
                        )
                    }
                    // Fill remaining space if odd number of products
                    if (productPair.size == 1) {
                        Spacer(modifier = Modifier.weight(1f))
                    }
                }
            }
        }
    }
    
    // Loading state
    if (uiState.isLoading) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            CircularProgressIndicator()
        }
    }
}`,
          language: 'kotlin'
        },
        'Product.kt': {
          content: `package com.rustyclint.ecommerce.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "products")
data class Product(
    @PrimaryKey
    val id: String,
    val name: String,
    val description: String,
    val price: Double,
    val imageUrl: String,
    val category: String,
    val rating: Float,
    val reviewCount: Int,
    val inStock: Boolean,
    val tags: List<String> = emptyList()
)

@Entity(tableName = "cart_items")
data class CartItem(
    @PrimaryKey
    val id: String,
    val productId: String,
    val quantity: Int,
    val addedAt: Long = System.currentTimeMillis()
)

data class CartItemWithProduct(
    val cartItem: CartItem,
    val product: Product
) {
    val totalPrice: Double
        get() = product.price * cartItem.quantity
}`,
          language: 'kotlin'
        }
      },
      features: [
        'Jetpack Compose modern UI',
        'Product catalog with search',
        'Shopping cart functionality',
        'Payment gateway integration',
        'User authentication',
        'Order tracking',
        'Push notifications',
        'Offline support with Room'
      ],
      useCase: 'Complete e-commerce solution with modern Android architecture and payment processing',
      techStack: ['Kotlin', 'Jetpack Compose', 'Room', 'Retrofit', 'Hilt', 'Navigation', 'Coil']
    },

    {
      id: 'android-kotlin-fitness',
      name: 'Android Fitness Tracker',
      description: 'Comprehensive fitness tracking app with workout plans, progress monitoring, and health integration',
      category: 'Health & Fitness',
      difficulty: 'Intermediate',
      tags: ['Kotlin', 'Health Connect', 'Sensors', 'Charts', 'Notifications'],
      icon: <Activity className="w-8 h-8 text-red-400" />,
      estimatedTime: '3-4 weeks',
      platforms: ['Android'],
      rating: 4.6,
      downloads: 2934,
      files: {
        'FitnessActivity.kt': {
          content: `package com.rustyclint.fitness

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.StepsRecord
import com.rustyclint.fitness.ui.theme.FitnessTheme
import com.rustyclint.fitness.ui.screens.MainScreen
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class FitnessActivity : ComponentActivity(), SensorEventListener {
    
    private lateinit var sensorManager: SensorManager
    private var stepCounterSensor: Sensor? = null
    private var healthConnectClient: HealthConnectClient? = null
    
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions[Manifest.permission.ACTIVITY_RECOGNITION] == true) {
            initializeSensors()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Health Connect
        if (HealthConnectClient.isAvailable(this)) {
            healthConnectClient = HealthConnectClient.getOrCreate(this)
        }
        
        requestPermissions()
        
        setContent {
            FitnessTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
    }
    
    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.ACTIVITY_RECOGNITION,
            Manifest.permission.BODY_SENSORS
        )
        
        val permissionsToRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (permissionsToRequest.isNotEmpty()) {
            permissionLauncher.launch(permissionsToRequest.toTypedArray())
        } else {
            initializeSensors()
        }
    }
    
    private fun initializeSensors() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        stepCounterSensor = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)
        
        stepCounterSensor?.let { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_UI)
        }
    }
    
    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            when (it.sensor.type) {
                Sensor.TYPE_STEP_COUNTER -> {
                    val stepCount = it.values[0].toInt()
                    // Update step count in ViewModel
                    updateStepCount(stepCount)
                }
            }
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Handle accuracy changes if needed
    }
    
    private fun updateStepCount(steps: Int) {
        // Update step count through ViewModel or repository
    }
    
    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
    }
}`,
          language: 'kotlin'
        },
        'WorkoutScreen.kt': {
          content: `package com.rustyclint.fitness.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.rustyclint.fitness.data.model.Workout
import com.rustyclint.fitness.data.model.Exercise
import com.rustyclint.fitness.ui.components.ExerciseCard
import com.rustyclint.fitness.ui.components.WorkoutTimer
import com.rustyclint.fitness.viewmodel.WorkoutViewModel

@Composable
fun WorkoutScreen(
    viewModel: WorkoutViewModel
) {
    val uiState by viewModel.uiState.collectAsState()
    var showTimer by remember { mutableStateOf(false) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Workout Header
        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = uiState.currentWorkout?.name ?: "Select Workout",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                
                if (uiState.currentWorkout != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "Duration: \${uiState.currentWorkout.estimatedDuration} min",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = "Difficulty: \${uiState.currentWorkout.difficulty}",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = { 
                                viewModel.startWorkout()
                                showTimer = true
                            },
                            modifier = Modifier.weight(1f),
                            enabled = !uiState.isWorkoutActive
                        ) {
                            Icon(Icons.Default.PlayArrow, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Start Workout")
                        }
                        
                        OutlinedButton(
                            onClick = { showTimer = !showTimer },
                            modifier = Modifier.weight(1f)
                        ) {
                            Icon(Icons.Default.Timer, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Timer")
                        }
                    }
                }
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Exercise List
        if (uiState.currentWorkout != null) {
            Text(
                text = "Exercises",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(uiState.currentWorkout.exercises) { exercise ->
                    ExerciseCard(
                        exercise = exercise,
                        isCompleted = uiState.completedExercises.contains(exercise.id),
                        onComplete = { viewModel.completeExercise(exercise.id) },
                        onSkip = { viewModel.skipExercise(exercise.id) }
                    )
                }
            }
        } else {
            // Workout Selection
            Text(
                text = "Available Workouts",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(uiState.availableWorkouts) { workout ->
                    WorkoutCard(
                        workout = workout,
                        onClick = { viewModel.selectWorkout(workout) }
                    )
                }
            }
        }
    }
    
    // Timer Overlay
    if (showTimer) {
        WorkoutTimer(
            isVisible = showTimer,
            onDismiss = { showTimer = false },
            onTimerComplete = { viewModel.completeCurrentExercise() }
        )
    }
}

@Composable
fun WorkoutCard(
    workout: Workout,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth(),
        onClick = onClick,
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = workout.name,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(4.dp))
            
            Text(
                text = workout.description,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "\${workout.exercises.size} exercises",
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = "\${workout.estimatedDuration} min",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    }
}`,
          language: 'kotlin'
        }
      },
      features: [
        'Step counting with sensors',
        'Workout plans and tracking',
        'Health Connect integration',
        'Progress analytics',
        'Custom timer functionality',
        'Achievement system',
        'Nutrition tracking',
        'Social challenges'
      ],
      useCase: 'Complete fitness tracking solution with sensor integration and health data synchronization',
      techStack: ['Kotlin', 'Jetpack Compose', 'Health Connect', 'Room', 'Sensors API', 'Charts']
    },

    // Cross-Platform Flutter Templates
    {
      id: 'flutter-travel-app',
      name: 'Flutter Travel Booking',
      description: 'Beautiful travel booking app with destination discovery, hotel booking, and trip planning features',
      category: 'Travel',
      difficulty: 'Advanced',
      tags: ['Flutter', 'Dart', 'Maps', 'Booking', 'Animations'],
      icon: <MapPin className="w-8 h-8 text-purple-400" />,
      estimatedTime: '4-6 weeks',
      platforms: ['Flutter', 'iOS', 'Android'],
      rating: 4.8,
      downloads: 5234,
      files: {
        'main.dart': {
          content: `import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'screens/home_screen.dart';
import 'screens/search_screen.dart';
import 'screens/booking_screen.dart';
import 'screens/profile_screen.dart';
import 'providers/travel_provider.dart';
import 'providers/booking_provider.dart';
import 'utils/app_theme.dart';

void main() {
  runApp(const TravelApp());
}

class TravelApp extends StatelessWidget {
  const TravelApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => TravelProvider()),
        ChangeNotifierProvider(create: (_) => BookingProvider()),
      ],
      child: MaterialApp(
        title: 'Travel Booking',
        theme: AppTheme.lightTheme,
        darkTheme: AppTheme.darkTheme,
        themeMode: ThemeMode.system,
        home: const MainScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({Key? key}) : super(key: key);

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;
  
  final List<Widget> _screens = [
    const HomeScreen(),
    const SearchScreen(),
    const BookingScreen(),
    const ProfileScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 10,
              offset: const Offset(0, -5),
            ),
          ],
        ),
        child: BottomNavigationBar(
          currentIndex: _currentIndex,
          onTap: (index) => setState(() => _currentIndex = index),
          type: BottomNavigationBarType.fixed,
          selectedItemColor: Theme.of(context).primaryColor,
          unselectedItemColor: Colors.grey,
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.home_outlined),
              activeIcon: Icon(Icons.home),
              label: 'Home',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.search_outlined),
              activeIcon: Icon(Icons.search),
              label: 'Search',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.bookmark_outline),
              activeIcon: Icon(Icons.bookmark),
              label: 'Bookings',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.person_outline),
              activeIcon: Icon(Icons.person),
              label: 'Profile',
            ),
          ],
        ),
      ),
    );
  }
}`,
          language: 'dart'
        },
        'home_screen.dart': {
          content: `import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/travel_provider.dart';
import '../models/destination.dart';
import '../widgets/destination_card.dart';
import '../widgets/search_bar_widget.dart';
import '../widgets/category_chips.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    ));
    
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOutCubic,
    ));
    
    _animationController.forward();
    
    // Load destinations
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<TravelProvider>().loadDestinations();
    });
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: FadeTransition(
          opacity: _fadeAnimation,
          child: SlideTransition(
            position: _slideAnimation,
            child: CustomScrollView(
              slivers: [
                // App Bar
                SliverAppBar(
                  expandedHeight: 120,
                  floating: true,
                  pinned: true,
                  backgroundColor: Colors.transparent,
                  elevation: 0,
                  flexibleSpace: FlexibleSpaceBar(
                    background: Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [
                            Theme.of(context).primaryColor.withOpacity(0.8),
                            Theme.of(context).primaryColor.withOpacity(0.6),
                          ],
                        ),
                      ),
                      child: const Padding(
                        padding: EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            Text(
                              'Discover',
                              style: TextStyle(
                                fontSize: 32,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            Text(
                              'Amazing destinations around the world',
                              style: TextStyle(
                                fontSize: 16,
                                color: Colors.white70,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
                
                // Search Bar
                SliverToBoxAdapter(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: SearchBarWidget(
                      onSearch: (query) {
                        context.read<TravelProvider>().searchDestinations(query);
                      },
                    ),
                  ),
                ),
                
                // Categories
                SliverToBoxAdapter(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16.0),
                    child: CategoryChips(
                      categories: const [
                        'All', 'Beach', 'Mountain', 'City', 'Adventure', 'Culture'
                      ],
                      onCategorySelected: (category) {
                        context.read<TravelProvider>().filterByCategory(category);
                      },
                    ),
                  ),
                ),
                
                // Popular Destinations
                SliverToBoxAdapter(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          'Popular Destinations',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        TextButton(
                          onPressed: () {
                            // Navigate to all destinations
                          },
                          child: const Text('See All'),
                        ),
                      ],
                    ),
                  ),
                ),
                
                // Destinations Grid
                Consumer<TravelProvider>(
                  builder: (context, travelProvider, child) {
                    if (travelProvider.isLoading) {
                      return const SliverToBoxAdapter(
                        child: Center(
                          child: Padding(
                            padding: EdgeInsets.all(32.0),
                            child: CircularProgressIndicator(),
                          ),
                        ),
                      );
                    }
                    
                    return SliverPadding(
                      padding: const EdgeInsets.symmetric(horizontal: 16.0),
                      sliver: SliverGrid(
                        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          childAspectRatio: 0.8,
                          crossAxisSpacing: 16,
                          mainAxisSpacing: 16,
                        ),
                        delegate: SliverChildBuilderDelegate(
                          (context, index) {
                            final destination = travelProvider.destinations[index];
                            return DestinationCard(
                              destination: destination,
                              onTap: () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) => DestinationDetailScreen(
                                      destination: destination,
                                    ),
                                  ),
                                );
                              },
                            );
                          },
                          childCount: travelProvider.destinations.length,
                        ),
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}`,
          language: 'dart'
        },
        'destination.dart': {
          content: `class Destination {
  final String id;
  final String name;
  final String country;
  final String description;
  final List<String> imageUrls;
  final double rating;
  final int reviewCount;
  final double price;
  final String currency;
  final List<String> categories;
  final double latitude;
  final double longitude;
  final List<String> highlights;
  final String bestTimeToVisit;
  final int durationDays;

  Destination({
    required this.id,
    required this.name,
    required this.country,
    required this.description,
    required this.imageUrls,
    required this.rating,
    required this.reviewCount,
    required this.price,
    required this.currency,
    required this.categories,
    required this.latitude,
    required this.longitude,
    required this.highlights,
    required this.bestTimeToVisit,
    required this.durationDays,
  });

  factory Destination.fromJson(Map<String, dynamic> json) {
    return Destination(
      id: json['id'],
      name: json['name'],
      country: json['country'],
      description: json['description'],
      imageUrls: List<String>.from(json['imageUrls']),
      rating: json['rating'].toDouble(),
      reviewCount: json['reviewCount'],
      price: json['price'].toDouble(),
      currency: json['currency'],
      categories: List<String>.from(json['categories']),
      latitude: json['latitude'].toDouble(),
      longitude: json['longitude'].toDouble(),
      highlights: List<String>.from(json['highlights']),
      bestTimeToVisit: json['bestTimeToVisit'],
      durationDays: json['durationDays'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'country': country,
      'description': description,
      'imageUrls': imageUrls,
      'rating': rating,
      'reviewCount': reviewCount,
      'price': price,
      'currency': currency,
      'categories': categories,
      'latitude': latitude,
      'longitude': longitude,
      'highlights': highlights,
      'bestTimeToVisit': bestTimeToVisit,
      'durationDays': durationDays,
    };
  }
}

class Hotel {
  final String id;
  final String name;
  final String address;
  final double rating;
  final int reviewCount;
  final double pricePerNight;
  final String currency;
  final List<String> imageUrls;
  final List<String> amenities;
  final double latitude;
  final double longitude;

  Hotel({
    required this.id,
    required this.name,
    required this.address,
    required this.rating,
    required this.reviewCount,
    required this.pricePerNight,
    required this.currency,
    required this.imageUrls,
    required this.amenities,
    required this.latitude,
    required this.longitude,
  });
}

class Booking {
  final String id;
  final String destinationId;
  final String hotelId;
  final DateTime checkInDate;
  final DateTime checkOutDate;
  final int guests;
  final double totalPrice;
  final String currency;
  final BookingStatus status;
  final DateTime createdAt;

  Booking({
    required this.id,
    required this.destinationId,
    required this.hotelId,
    required this.checkInDate,
    required this.checkOutDate,
    required this.guests,
    required this.totalPrice,
    required this.currency,
    required this.status,
    required this.createdAt,
  });
}

enum BookingStatus {
  pending,
  confirmed,
  cancelled,
  completed,
}`,
          language: 'dart'
        }
      },
      features: [
        'Beautiful destination discovery',
        'Interactive maps integration',
        'Hotel booking system',
        'Trip planning tools',
        'Real-time price updates',
        'Offline map support',
        'Photo galleries',
        'Review and rating system'
      ],
      useCase: 'Complete travel booking platform with destination discovery and trip planning',
      techStack: ['Flutter', 'Dart', 'Google Maps', 'Provider', 'HTTP', 'Shared Preferences']
    },

    {
      id: 'flutter-music-player',
      name: 'Flutter Music Player',
      description: 'Feature-rich music player with streaming, playlists, lyrics, and social sharing capabilities',
      category: 'Entertainment',
      difficulty: 'Advanced',
      tags: ['Flutter', 'Audio', 'Streaming', 'Animations', 'Social'],
      icon: <Music className="w-8 h-8 text-pink-400" />,
      estimatedTime: '3-4 weeks',
      platforms: ['Flutter', 'iOS', 'Android'],
      rating: 4.9,
      downloads: 6789,
      files: {
        'main.dart': {
          content: `import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:just_audio/just_audio.dart';
import 'package:audio_session/audio_session.dart';
import 'screens/home_screen.dart';
import 'screens/player_screen.dart';
import 'screens/library_screen.dart';
import 'screens/search_screen.dart';
import 'providers/music_provider.dart';
import 'providers/player_provider.dart';
import 'utils/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Configure audio session
  final session = await AudioSession.instance;
  await session.configure(const AudioSessionConfiguration.music());
  
  runApp(const MusicApp());
}

class MusicApp extends StatelessWidget {
  const MusicApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => MusicProvider()),
        ChangeNotifierProvider(create: (_) => PlayerProvider()),
      ],
      child: MaterialApp(
        title: 'Music Player',
        theme: AppTheme.darkTheme,
        home: const MainScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({Key? key}) : super(key: key);

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> with TickerProviderStateMixin {
  int _currentIndex = 0;
  late AnimationController _playerAnimationController;
  late Animation<double> _playerSlideAnimation;
  
  final List<Widget> _screens = [
    const HomeScreen(),
    const SearchScreen(),
    const LibraryScreen(),
  ];

  @override
  void initState() {
    super.initState();
    _playerAnimationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    
    _playerSlideAnimation = Tween<double>(
      begin: 1.0,
      end: 0.0,
    ).animate(CurvedAnimation(
      parent: _playerAnimationController,
      curve: Curves.easeInOut,
    ));
  }

  @override
  void dispose() {
    _playerAnimationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Main content
          IndexedStack(
            index: _currentIndex,
            children: _screens,
          ),
          
          // Mini player
          Consumer<PlayerProvider>(
            builder: (context, playerProvider, child) {
              if (playerProvider.currentSong == null) {
                return const SizedBox.shrink();
              }
              
              return Positioned(
                left: 0,
                right: 0,
                bottom: 80,
                child: GestureDetector(
                  onTap: () {
                    Navigator.push(
                      context,
                      PageRouteBuilder(
                        pageBuilder: (context, animation, secondaryAnimation) =>
                            const PlayerScreen(),
                        transitionsBuilder: (context, animation, secondaryAnimation, child) {
                          return SlideTransition(
                            position: Tween<Offset>(
                              begin: const Offset(0, 1),
                              end: Offset.zero,
                            ).animate(animation),
                            child: child,
                          );
                        },
                      ),
                    );
                  },
                  child: Container(
                    height: 70,
                    margin: const EdgeInsets.symmetric(horizontal: 16),
                    decoration: BoxDecoration(
                      color: Theme.of(context).cardColor,
                      borderRadius: BorderRadius.circular(12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.3),
                          blurRadius: 10,
                          offset: const Offset(0, 5),
                        ),
                      ],
                    ),
                    child: Row(
                      children: [
                        // Album art
                        Container(
                          width: 70,
                          height: 70,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(12),
                            image: DecorationImage(
                              image: NetworkImage(playerProvider.currentSong!.albumArt),
                              fit: BoxFit.cover,
                            ),
                          ),
                        ),
                        
                        // Song info
                        Expanded(
                          child: Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 12),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  playerProvider.currentSong!.title,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.bold,
                                    fontSize: 16,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                                Text(
                                  playerProvider.currentSong!.artist,
                                  style: TextStyle(
                                    color: Colors.grey[400],
                                    fontSize: 14,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ],
                            ),
                          ),
                        ),
                        
                        // Play/pause button
                        IconButton(
                          onPressed: playerProvider.togglePlayPause,
                          icon: Icon(
                            playerProvider.isPlaying ? Icons.pause : Icons.play_arrow,
                            size: 30,
                          ),
                        ),
                        
                        const SizedBox(width: 8),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
        ],
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 10,
              offset: const Offset(0, -5),
            ),
          ],
        ),
        child: BottomNavigationBar(
          currentIndex: _currentIndex,
          onTap: (index) => setState(() => _currentIndex = index),
          type: BottomNavigationBarType.fixed,
          backgroundColor: Theme.of(context).scaffoldBackgroundColor,
          selectedItemColor: Theme.of(context).primaryColor,
          unselectedItemColor: Colors.grey,
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.home_outlined),
              activeIcon: Icon(Icons.home),
              label: 'Home',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.search_outlined),
              activeIcon: Icon(Icons.search),
              label: 'Search',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.library_music_outlined),
              activeIcon: Icon(Icons.library_music),
              label: 'Library',
            ),
          ],
        ),
      ),
    );
  }
}`,
          language: 'dart'
        },
        'player_screen.dart': {
          content: `import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/player_provider.dart';
import '../widgets/audio_visualizer.dart';
import '../widgets/lyrics_view.dart';

class PlayerScreen extends StatefulWidget {
  const PlayerScreen({Key? key}) : super(key: key);

  @override
  State<PlayerScreen> createState() => _PlayerScreenState();
}

class _PlayerScreenState extends State<PlayerScreen> with TickerProviderStateMixin {
  late AnimationController _albumRotationController;
  late AnimationController _lyricsController;
  late Animation<double> _albumRotation;
  bool _showLyrics = false;

  @override
  void initState() {
    super.initState();
    
    _albumRotationController = AnimationController(
      duration: const Duration(seconds: 20),
      vsync: this,
    );
    
    _lyricsController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    
    _albumRotation = Tween<double>(
      begin: 0,
      end: 1,
    ).animate(_albumRotationController);
    
    // Start rotation if playing
    final playerProvider = context.read<PlayerProvider>();
    if (playerProvider.isPlaying) {
      _albumRotationController.repeat();
    }
  }

  @override
  void dispose() {
    _albumRotationController.dispose();
    _lyricsController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Theme.of(context).primaryColor.withOpacity(0.8),
              Theme.of(context).scaffoldBackgroundColor,
            ],
          ),
        ),
        child: SafeArea(
          child: Consumer<PlayerProvider>(
            builder: (context, playerProvider, child) {
              if (playerProvider.currentSong == null) {
                return const Center(
                  child: Text('No song selected'),
                );
              }
              
              // Control album rotation based on play state
              if (playerProvider.isPlaying && !_albumRotationController.isAnimating) {
                _albumRotationController.repeat();
              } else if (!playerProvider.isPlaying && _albumRotationController.isAnimating) {
                _albumRotationController.stop();
              }
              
              return Column(
                children: [
                  // Header
                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        IconButton(
                          onPressed: () => Navigator.pop(context),
                          icon: const Icon(Icons.keyboard_arrow_down, size: 30),
                        ),
                        Column(
                          children: [
                            const Text(
                              'PLAYING FROM PLAYLIST',
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                                letterSpacing: 1,
                              ),
                            ),
                            Text(
                              playerProvider.currentPlaylist?.name ?? 'Unknown',
                              style: const TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                        IconButton(
                          onPressed: () {
                            // Show more options
                          },
                          icon: const Icon(Icons.more_vert),
                        ),
                      ],
                    ),
                  ),
                  
                  // Album art or lyrics
                  Expanded(
                    flex: 3,
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 32.0),
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 300),
                        child: _showLyrics
                            ? LyricsView(
                                key: const ValueKey('lyrics'),
                                song: playerProvider.currentSong!,
                                currentPosition: playerProvider.position,
                              )
                            : GestureDetector(
                                key: const ValueKey('album'),
                                onTap: () {
                                  setState(() {
                                    _showLyrics = !_showLyrics;
                                  });
                                },
                                child: AnimatedBuilder(
                                  animation: _albumRotation,
                                  builder: (context, child) {
                                    return Transform.rotate(
                                      angle: _albumRotation.value * 2 * 3.14159,
                                      child: Container(
                                        width: double.infinity,
                                        decoration: BoxDecoration(
                                          shape: BoxShape.circle,
                                          boxShadow: [
                                            BoxShadow(
                                              color: Colors.black.withOpacity(0.3),
                                              blurRadius: 20,
                                              offset: const Offset(0, 10),
                                            ),
                                          ],
                                        ),
                                        child: ClipOval(
                                          child: AspectRatio(
                                            aspectRatio: 1,
                                            child: Image.network(
                                              playerProvider.currentSong!.albumArt,
                                              fit: BoxFit.cover,
                                            ),
                                          ),
                                        ),
                                      ),
                                    );
                                  },
                                ),
                              ),
                      ),
                    ),
                  ),
                  
                  // Audio visualizer
                  if (playerProvider.isPlaying && !_showLyrics)
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 16.0),
                      child: AudioVisualizer(),
                    ),
                  
                  // Song info
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32.0),
                    child: Column(
                      children: [
                        Text(
                          playerProvider.currentSong!.title,
                          style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                          ),
                          textAlign: TextAlign.center,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          playerProvider.currentSong!.artist,
                          style: TextStyle(
                            fontSize: 18,
                            color: Colors.grey[400],
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                  
                  // Progress bar
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 16.0),
                    child: Column(
                      children: [
                        SliderTheme(
                          data: SliderTheme.of(context).copyWith(
                            thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
                            trackHeight: 4,
                            overlayShape: const RoundSliderOverlayShape(overlayRadius: 12),
                          ),
                          child: Slider(
                            value: playerProvider.position.inSeconds.toDouble(),
                            max: playerProvider.duration.inSeconds.toDouble(),
                            onChanged: (value) {
                              playerProvider.seek(Duration(seconds: value.toInt()));
                            },
                          ),
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              _formatDuration(playerProvider.position),
                              style: TextStyle(color: Colors.grey[400]),
                            ),
                            Text(
                              _formatDuration(playerProvider.duration),
                              style: TextStyle(color: Colors.grey[400]),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  
                  // Controls
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 16.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        IconButton(
                          onPressed: playerProvider.toggleShuffle,
                          icon: Icon(
                            Icons.shuffle,
                            color: playerProvider.isShuffleEnabled
                                ? Theme.of(context).primaryColor
                                : Colors.grey,
                          ),
                        ),
                        IconButton(
                          onPressed: playerProvider.previousSong,
                          icon: const Icon(Icons.skip_previous, size: 40),
                        ),
                        Container(
                          width: 70,
                          height: 70,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Theme.of(context).primaryColor,
                          ),
                          child: IconButton(
                            onPressed: playerProvider.togglePlayPause,
                            icon: Icon(
                              playerProvider.isPlaying ? Icons.pause : Icons.play_arrow,
                              size: 35,
                              color: Colors.white,
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: playerProvider.nextSong,
                          icon: const Icon(Icons.skip_next, size: 40),
                        ),
                        IconButton(
                          onPressed: playerProvider.toggleRepeat,
                          icon: Icon(
                            playerProvider.repeatMode == RepeatMode.one
                                ? Icons.repeat_one
                                : Icons.repeat,
                            color: playerProvider.repeatMode != RepeatMode.off
                                ? Theme.of(context).primaryColor
                                : Colors.grey,
                          ),
                        ),
                      ],
                    ),
                  ),
                  
                  const SizedBox(height: 16),
                ],
              );
            },
          ),
        ),
      ),
    );
  }
  
  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    String twoDigitMinutes = twoDigits(duration.inMinutes.remainder(60));
    String twoDigitSeconds = twoDigits(duration.inSeconds.remainder(60));
    return '\${twoDigits(duration.inHours)}:\$twoDigitMinutes:\$twoDigitSeconds';
  }
}`,
          language: 'dart'
        }
      },
      features: [
        'High-quality audio streaming',
        'Custom playlist creation',
        'Lyrics synchronization',
        'Audio visualizer',
        'Social sharing features',
        'Offline music support',
        'Background playback',
        'Equalizer controls'
      ],
      useCase: 'Professional music streaming app with advanced audio features and social integration',
      techStack: ['Flutter', 'Dart', 'just_audio', 'Provider', 'HTTP', 'Shared Preferences']
    },

    // React Native Templates
    {
      id: 'react-native-food-delivery',
      name: 'React Native Food Delivery',
      description: 'Complete food delivery app with restaurant discovery, ordering, real-time tracking, and payment integration',
      category: 'Food & Delivery',
      difficulty: 'Advanced',
      tags: ['React Native', 'TypeScript', 'Maps', 'Payment', 'Real-time'],
      icon: <Coffee className="w-8 h-8 text-orange-400" />,
      estimatedTime: '5-6 weeks',
      platforms: ['React Native', 'iOS', 'Android'],
      rating: 4.7,
      downloads: 3892,
      files: {
        'App.tsx': {
          content: `import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import Icon from 'react-native-vector-icons/Ionicons';
import { store, persistor } from './src/store';
import HomeScreen from './src/screens/HomeScreen';
import SearchScreen from './src/screens/SearchScreen';
import OrdersScreen from './src/screens/OrdersScreen';
import ProfileScreen from './src/screens/ProfileScreen';
import RestaurantScreen from './src/screens/RestaurantScreen';
import CartScreen from './src/screens/CartScreen';
import CheckoutScreen from './src/screens/CheckoutScreen';
import OrderTrackingScreen from './src/screens/OrderTrackingScreen';
import { colors } from './src/theme/colors';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

const TabNavigator = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string;

          switch (route.name) {
            case 'Home':
              iconName = focused ? 'home' : 'home-outline';
              break;
            case 'Search':
              iconName = focused ? 'search' : 'search-outline';
              break;
            case 'Orders':
              iconName = focused ? 'receipt' : 'receipt-outline';
              break;
            case 'Profile':
              iconName = focused ? 'person' : 'person-outline';
              break;
            default:
              iconName = 'home-outline';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.gray,
        headerShown: false,
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Search" component={SearchScreen} />
      <Tab.Screen name="Orders" component={OrdersScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
};

const App = () => {
  return (
    <Provider store={store}>
      <PersistGate loading={null} persistor={persistor}>
        <NavigationContainer>
          <Stack.Navigator screenOptions={{ headerShown: false }}>
            <Stack.Screen name="Main" component={TabNavigator} />
            <Stack.Screen name="Restaurant" component={RestaurantScreen} />
            <Stack.Screen name="Cart" component={CartScreen} />
            <Stack.Screen name="Checkout" component={CheckoutScreen} />
            <Stack.Screen name="OrderTracking" component={OrderTrackingScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </PersistGate>
    </Provider>
  );
};

export default App;`,
          language: 'typescript'
        },
        'HomeScreen.tsx': {
          content: `import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  FlatList,
  Dimensions,
} from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import Icon from 'react-native-vector-icons/Ionicons';
import { RootState } from '../store';
import { fetchRestaurants, fetchCategories } from '../store/slices/restaurantSlice';
import SearchBar from '../components/SearchBar';
import CategoryCard from '../components/CategoryCard';
import RestaurantCard from '../components/RestaurantCard';
import PromoCard from '../components/PromoCard';
import { colors } from '../theme/colors';
import { Restaurant, Category } from '../types';

const { width } = Dimensions.get('window');

const HomeScreen = ({ navigation }: any) => {
  const dispatch = useDispatch();
  const { restaurants, categories, loading } = useSelector(
    (state: RootState) => state.restaurant
  );
  const { user } = useSelector((state: RootState) => state.auth);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  useEffect(() => {
    dispatch(fetchRestaurants());
    dispatch(fetchCategories());
  }, [dispatch]);

  const filteredRestaurants = selectedCategory === 'all' 
    ? restaurants 
    : restaurants.filter(restaurant => 
        restaurant.categories.includes(selectedCategory)
      );

  const renderHeader = () => (
    <View style={styles.header}>
      <View style={styles.headerTop}>
        <View>
          <Text style={styles.greeting}>Good morning,</Text>
          <Text style={styles.userName}>{user?.name || 'Guest'}</Text>
        </View>
        <TouchableOpacity 
          style={styles.cartButton}
          onPress={() => navigation.navigate('Cart')}
        >
          <Icon name="bag-outline" size={24} color={colors.primary} />
          <View style={styles.cartBadge}>
            <Text style={styles.cartBadgeText}>2</Text>
          </View>
        </TouchableOpacity>
      </View>
      
      <View style={styles.locationContainer}>
        <Icon name="location-outline" size={16} color={colors.gray} />
        <Text style={styles.locationText}>Deliver to: Home</Text>
        <Icon name="chevron-down-outline" size={16} color={colors.gray} />
      </View>
    </View>
  );

  const renderSearchAndPromo = () => (
    <View style={styles.searchPromoContainer}>
      <SearchBar
        placeholder="Search for restaurants or dishes..."
        onPress={() => navigation.navigate('Search')}
      />
      
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false}
        style={styles.promoScroll}
      >
        <PromoCard
          title="Free Delivery"
          subtitle="On orders over $30"
          backgroundColor={colors.primary}
          image="https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=300&h=150&fit=crop"
        />
        <PromoCard
          title="20% Off"
          subtitle="First order discount"
          backgroundColor={colors.secondary}
          image="https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=300&h=150&fit=crop"
        />
      </ScrollView>
    </View>
  );

  const renderCategories = () => (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Categories</Text>
      <FlatList
        data={[{ id: 'all', name: 'All', icon: 'grid-outline' }, ...categories]}
        horizontal
        showsHorizontalScrollIndicator={false}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <CategoryCard
            category={item}
            isSelected={selectedCategory === item.id}
            onPress={() => setSelectedCategory(item.id)}
          />
        )}
        contentContainerStyle={styles.categoriesList}
      />
    </View>
  );

  const renderPopularRestaurants = () => (
    <View style={styles.section}>
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Popular Restaurants</Text>
        <TouchableOpacity>
          <Text style={styles.seeAllText}>See all</Text>
        </TouchableOpacity>
      </View>
      
      <FlatList
        data={filteredRestaurants.slice(0, 5)}
        horizontal
        showsHorizontalScrollIndicator={false}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <RestaurantCard
            restaurant={item}
            onPress={() => navigation.navigate('Restaurant', { restaurant: item })}
            style={styles.popularRestaurantCard}
          />
        )}
        contentContainerStyle={styles.restaurantsList}
      />
    </View>
  );

  const renderNearbyRestaurants = () => (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Nearby Restaurants</Text>
      {filteredRestaurants.map((restaurant) => (
        <RestaurantCard
          key={restaurant.id}
          restaurant={restaurant}
          onPress={() => navigation.navigate('Restaurant', { restaurant })}
          style={styles.nearbyRestaurantCard}
          horizontal
        />
      ))}
    </View>
  );

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {renderHeader()}
      {renderSearchAndPromo()}
      {renderCategories()}
      {renderPopularRestaurants()}
      {renderNearbyRestaurants()}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  header: {
    padding: 20,
    paddingTop: 50,
    backgroundColor: colors.white,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  greeting: {
    fontSize: 16,
    color: colors.gray,
  },
  userName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.dark,
  },
  cartButton: {
    position: 'relative',
    padding: 10,
    backgroundColor: colors.lightGray,
    borderRadius: 12,
  },
  cartBadge: {
    position: 'absolute',
    top: 5,
    right: 5,
    backgroundColor: colors.primary,
    borderRadius: 10,
    width: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cartBadgeText: {
    color: colors.white,
    fontSize: 12,
    fontWeight: 'bold',
  },
  locationContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  locationText: {
    marginLeft: 5,
    marginRight: 5,
    color: colors.gray,
    fontSize: 14,
  },
  searchPromoContainer: {
    padding: 20,
    backgroundColor: colors.white,
  },
  promoScroll: {
    marginTop: 15,
  },
  section: {
    marginTop: 20,
    paddingHorizontal: 20,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.dark,
  },
  seeAllText: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '600',
  },
  categoriesList: {
    paddingVertical: 10,
  },
  restaurantsList: {
    paddingVertical: 10,
  },
  popularRestaurantCard: {
    width: width * 0.7,
    marginRight: 15,
  },
  nearbyRestaurantCard: {
    marginBottom: 15,
  },
});

export default HomeScreen;`,
          language: 'typescript'
        },
        'types.ts': {
          content: `export interface Restaurant {
  id: string;
  name: string;
  description: string;
  image: string;
  rating: number;
  reviewCount: number;
  deliveryTime: string;
  deliveryFee: number;
  minimumOrder: number;
  categories: string[];
  cuisine: string;
  address: string;
  latitude: number;
  longitude: number;
  isOpen: boolean;
  menu: MenuItem[];
  promotions?: Promotion[];
}

export interface MenuItem {
  id: string;
  name: string;
  description: string;
  price: number;
  image: string;
  category: string;
  isVegetarian: boolean;
  isSpicy: boolean;
  allergens: string[];
  customizations?: Customization[];
}

export interface Customization {
  id: string;
  name: string;
  type: 'single' | 'multiple';
  required: boolean;
  options: CustomizationOption[];
}

export interface CustomizationOption {
  id: string;
  name: string;
  price: number;
}

export interface Category {
  id: string;
  name: string;
  icon: string;
  image?: string;
}

export interface CartItem {
  id: string;
  menuItem: MenuItem;
  quantity: number;
  customizations: SelectedCustomization[];
  specialInstructions?: string;
}

export interface SelectedCustomization {
  customizationId: string;
  optionIds: string[];
}

export interface Order {
  id: string;
  restaurantId: string;
  restaurant: Restaurant;
  items: CartItem[];
  subtotal: number;
  deliveryFee: number;
  tax: number;
  total: number;
  status: OrderStatus;
  deliveryAddress: Address;
  paymentMethod: PaymentMethod;
  estimatedDeliveryTime: Date;
  actualDeliveryTime?: Date;
  driver?: Driver;
  createdAt: Date;
  updatedAt: Date;
}

export enum OrderStatus {
  PLACED = 'placed',
  CONFIRMED = 'confirmed',
  PREPARING = 'preparing',
  READY = 'ready',
  PICKED_UP = 'picked_up',
  ON_THE_WAY = 'on_the_way',
  DELIVERED = 'delivered',
  CANCELLED = 'cancelled',
}

export interface Address {
  id: string;
  label: string;
  street: string;
  city: string;
  state: string;
  zipCode: string;
  latitude: number;
  longitude: number;
  instructions?: string;
}

export interface PaymentMethod {
  id: string;
  type: 'credit_card' | 'debit_card' | 'paypal' | 'apple_pay' | 'google_pay';
  last4?: string;
  brand?: string;
  isDefault: boolean;
}

export interface Driver {
  id: string;
  name: string;
  photo: string;
  rating: number;
  phone: string;
  vehicle: {
    type: string;
    model: string;
    licensePlate: string;
  };
  location: {
    latitude: number;
    longitude: number;
  };
}

export interface Promotion {
  id: string;
  title: string;
  description: string;
  type: 'percentage' | 'fixed_amount' | 'free_delivery';
  value: number;
  minimumOrder?: number;
  validUntil: Date;
  code?: string;
}

export interface User {
  id: string;
  name: string;
  email: string;
  phone: string;
  photo?: string;
  addresses: Address[];
  paymentMethods: PaymentMethod[];
  favoriteRestaurants: string[];
  orderHistory: Order[];
}`,
          language: 'typescript'
        }
      },
      features: [
        'Restaurant discovery and search',
        'Real-time order tracking',
        'Multiple payment options',
        'GPS-based delivery',
        'Push notifications',
        'Favorites and order history',
        'Rating and review system',
        'Promo codes and discounts'
      ],
      useCase: 'Complete food delivery platform with restaurant management and real-time tracking',
      techStack: ['React Native', 'TypeScript', 'Redux', 'React Navigation', 'Maps', 'Push Notifications']
    },

    {
      id: 'react-native-crypto-wallet',
      name: 'React Native Crypto Wallet',
      description: 'Secure cryptocurrency wallet with portfolio tracking, trading, and DeFi integration',
      category: 'Finance',
      difficulty: 'Expert',
      tags: ['React Native', 'Blockchain', 'Security', 'Charts', 'Biometrics'],
      icon: <Wallet className="w-8 h-8 text-yellow-400" />,
      estimatedTime: '6-8 weeks',
      platforms: ['React Native', 'iOS', 'Android'],
      rating: 4.6,
      downloads: 2156,
      files: {
        'App.tsx': {
          content: `import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import SplashScreen from 'react-native-splash-screen';
import { store, persistor } from './src/store';
import AuthNavigator from './src/navigation/AuthNavigator';
import MainNavigator from './src/navigation/MainNavigator';
import { useSelector } from 'react-redux';
import { RootState } from './src/store';
import BiometricAuth from './src/components/BiometricAuth';
import { WalletProvider } from './src/context/WalletContext';
import { SecurityProvider } from './src/context/SecurityContext';

const Stack = createStackNavigator();

const AppContent = () => {
  const { isAuthenticated, user } = useSelector((state: RootState) => state.auth);
  const { isUnlocked } = useSelector((state: RootState) => state.security);

  useEffect(() => {
    SplashScreen.hide();
  }, []);

  if (isAuthenticated && !isUnlocked) {
    return <BiometricAuth />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {isAuthenticated ? (
          <Stack.Screen name="Main" component={MainNavigator} />
        ) : (
          <Stack.Screen name="Auth" component={AuthNavigator} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const App = () => {
  return (
    <Provider store={store}>
      <PersistGate loading={null} persistor={persistor}>
        <SecurityProvider>
          <WalletProvider>
            <AppContent />
          </WalletProvider>
        </SecurityProvider>
      </PersistGate>
    </Provider>
  );
};

export default App;`,
          language: 'typescript'
        },
        'WalletScreen.tsx': {
          content: `import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Dimensions,
} from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import { LineChart } from 'react-native-chart-kit';
import Icon from 'react-native-vector-icons/Ionicons';
import { RootState } from '../store';
import { fetchWalletData, fetchPrices } from '../store/slices/walletSlice';
import CryptoCard from '../components/CryptoCard';
import PortfolioChart from '../components/PortfolioChart';
import QuickActions from '../components/QuickActions';
import { colors } from '../theme/colors';
import { formatCurrency, formatPercentage } from '../utils/formatters';

const { width } = Dimensions.get('window');

const WalletScreen = ({ navigation }: any) => {
  const dispatch = useDispatch();
  const { 
    portfolio, 
    totalBalance, 
    totalChange24h, 
    totalChangePercentage24h,
    loading 
  } = useSelector((state: RootState) => state.wallet);
  
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');

  useEffect(() => {
    dispatch(fetchWalletData());
    dispatch(fetchPrices());
  }, [dispatch]);

  const onRefresh = async () => {
    setRefreshing(true);
    await Promise.all([
      dispatch(fetchWalletData()),
      dispatch(fetchPrices())
    ]);
    setRefreshing(false);
  };

  const timeframes = [
    { label: '1H', value: '1h' },
    { label: '24H', value: '24h' },
    { label: '7D', value: '7d' },
    { label: '30D', value: '30d' },
    { label: '1Y', value: '1y' },
  ];

  const renderHeader = () => (
    <View style={styles.header}>
      <View style={styles.headerTop}>
        <Text style={styles.greeting}>Portfolio</Text>
        <TouchableOpacity 
          style={styles.settingsButton}
          onPress={() => navigation.navigate('Settings')}
        >
          <Icon name="settings-outline" size={24} color={colors.white} />
        </TouchableOpacity>
      </View>
      
      <View style={styles.balanceContainer}>
        <Text style={styles.totalBalance}>
          {formatCurrency(totalBalance)}
        </Text>
        <View style={styles.changeContainer}>
          <Icon 
            name={totalChange24h >= 0 ? "trending-up" : "trending-down"} 
            size={16} 
            color={totalChange24h >= 0 ? colors.success : colors.error} 
          />
          <Text style={[
            styles.changeText,
            { color: totalChange24h >= 0 ? colors.success : colors.error }
          ]}>
            {formatCurrency(Math.abs(totalChange24h))} ({formatPercentage(totalChangePercentage24h)})
          </Text>
        </View>
      </View>
    </View>
  );

  const renderTimeframeSelector = () => (
    <View style={styles.timeframeContainer}>
      {timeframes.map((timeframe) => (
        <TouchableOpacity
          key={timeframe.value}
          style={[
            styles.timeframeButton,
            selectedTimeframe === timeframe.value && styles.timeframeButtonActive
          ]}
          onPress={() => setSelectedTimeframe(timeframe.value)}
        >
          <Text style={[
            styles.timeframeText,
            selectedTimeframe === timeframe.value && styles.timeframeTextActive
          ]}>
            {timeframe.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderPortfolioChart = () => (
    <View style={styles.chartContainer}>
      <PortfolioChart 
        timeframe={selectedTimeframe}
        data={portfolio}
      />
    </View>
  );

  const renderQuickActions = () => (
    <View style={styles.quickActionsContainer}>
      <QuickActions
        onSend={() => navigation.navigate('Send')}
        onReceive={() => navigation.navigate('Receive')}
        onBuy={() => navigation.navigate('Buy')}
        onSwap={() => navigation.navigate('Swap')}
      />
    </View>
  );

  const renderAssets = () => (
    <View style={styles.assetsContainer}>
      <View style={styles.assetsHeader}>
        <Text style={styles.assetsTitle}>Your Assets</Text>
        <TouchableOpacity onPress={() => navigation.navigate('AllAssets')}>
          <Text style={styles.seeAllText}>See all</Text>
        </TouchableOpacity>
      </View>
      
      {portfolio.map((asset) => (
        <CryptoCard
          key={asset.symbol}
          asset={asset}
          onPress={() => navigation.navigate('AssetDetail', { asset })}
        />
      ))}
    </View>
  );

  const renderDeFiSection = () => (
    <View style={styles.defiContainer}>
      <Text style={styles.sectionTitle}>DeFi Opportunities</Text>
      
      <TouchableOpacity 
        style={styles.defiCard}
        onPress={() => navigation.navigate('Staking')}
      >
        <View style={styles.defiCardContent}>
          <View style={styles.defiIcon}>
            <Icon name="trending-up" size={24} color={colors.primary} />
          </View>
          <View style={styles.defiInfo}>
            <Text style={styles.defiTitle}>Staking Rewards</Text>
            <Text style={styles.defiSubtitle}>Earn up to 12% APY</Text>
          </View>
          <Icon name="chevron-forward" size={20} color={colors.gray} />
        </View>
      </TouchableOpacity>
      
      <TouchableOpacity 
        style={styles.defiCard}
        onPress={() => navigation.navigate('Lending')}
      >
        <View style={styles.defiCardContent}>
          <View style={styles.defiIcon}>
            <Icon name="wallet" size={24} color={colors.secondary} />
          </View>
          <View style={styles.defiInfo}>
            <Text style={styles.defiTitle}>Lending</Text>
            <Text style={styles.defiSubtitle}>Lend and earn interest</Text>
          </View>
          <Icon name="chevron-forward" size={20} color={colors.gray} />
        </View>
      </TouchableOpacity>
    </View>
  );

  return (
    <ScrollView 
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
      showsVerticalScrollIndicator={false}
    >
      {renderHeader()}
      {renderTimeframeSelector()}
      {renderPortfolioChart()}
      {renderQuickActions()}
      {renderAssets()}
      {renderDeFiSection()}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  header: {
    padding: 20,
    paddingTop: 50,
    backgroundColor: colors.primary,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  greeting: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.white,
  },
  settingsButton: {
    padding: 8,
  },
  balanceContainer: {
    alignItems: 'center',
  },
  totalBalance: {
    fontSize: 36,
    fontWeight: 'bold',
    color: colors.white,
    marginBottom: 8,
  },
  changeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  changeText: {
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 4,
  },
  timeframeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: colors.white,
  },
  timeframeButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  timeframeButtonActive: {
    backgroundColor: colors.primary,
  },
  timeframeText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.gray,
  },
  timeframeTextActive: {
    color: colors.white,
  },
  chartContainer: {
    backgroundColor: colors.white,
    paddingVertical: 20,
  },
  quickActionsContainer: {
    backgroundColor: colors.white,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  assetsContainer: {
    backgroundColor: colors.white,
    marginTop: 10,
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  assetsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  assetsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.dark,
  },
  seeAllText: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '600',
  },
  defiContainer: {
    backgroundColor: colors.white,
    marginTop: 10,
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.dark,
    marginBottom: 15,
  },
  defiCard: {
    backgroundColor: colors.lightGray,
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  defiCardContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  defiIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.white,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  defiInfo: {
    flex: 1,
  },
  defiTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: colors.dark,
    marginBottom: 4,
  },
  defiSubtitle: {
    fontSize: 14,
    color: colors.gray,
  },
});

export default WalletScreen;`,
          language: 'typescript'
        }
      },
      features: [
        'Multi-currency wallet support',
        'Real-time price tracking',
        'Portfolio analytics',
        'DeFi integration (staking, lending)',
        'Biometric security',
        'Hardware wallet support',
        'Trading and swapping',
        'Transaction history'
      ],
      useCase: 'Professional cryptocurrency wallet with advanced trading and DeFi capabilities',
      techStack: ['React Native', 'TypeScript', 'Redux', 'Biometrics', 'Charts', 'Blockchain APIs']
    },

    // Additional Mobile Templates
    {
      id: 'flutter-video-streaming',
      name: 'Flutter Video Streaming',
      description: 'Netflix-style video streaming app with offline downloads, personalized recommendations, and social features',
      category: 'Entertainment',
      difficulty: 'Advanced',
      tags: ['Flutter', 'Video', 'Streaming', 'Offline', 'AI'],
      icon: <Video className="w-8 h-8 text-red-400" />,
      estimatedTime: '5-7 weeks',
      platforms: ['Flutter', 'iOS', 'Android'],
      rating: 4.8,
      downloads: 4567,
      files: {
        'main.dart': {
          content: `import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:video_player/video_player.dart';
import 'screens/splash_screen.dart';
import 'screens/home_screen.dart';
import 'screens/video_player_screen.dart';
import 'screens/downloads_screen.dart';
import 'screens/profile_screen.dart';
import 'providers/video_provider.dart';
import 'providers/download_provider.dart';
import 'providers/user_provider.dart';
import 'utils/app_theme.dart';

void main() {
  runApp(const VideoStreamingApp());
}

class VideoStreamingApp extends StatelessWidget {
  const VideoStreamingApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => VideoProvider()),
        ChangeNotifierProvider(create: (_) => DownloadProvider()),
        ChangeNotifierProvider(create: (_) => UserProvider()),
      ],
      child: MaterialApp(
        title: 'StreamFlix',
        theme: AppTheme.darkTheme,
        home: const SplashScreen(),
        debugShowCheckedModeBanner: false,
        routes: {
          '/home': (context) => const HomeScreen(),
          '/player': (context) => const VideoPlayerScreen(),
          '/downloads': (context) => const DownloadsScreen(),
          '/profile': (context) => const ProfileScreen(),
        },
      ),
    );
  }
}`,
          language: 'dart'
        }
      },
      features: [
        'HD/4K video streaming',
        'Offline video downloads',
        'Personalized recommendations',
        'Multiple user profiles',
        'Chromecast support',
        'Subtitle support',
        'Parental controls',
        'Social sharing features'
      ],
      useCase: 'Professional video streaming platform with Netflix-like features and offline capabilities',
      techStack: ['Flutter', 'Dart', 'Video Player', 'Provider', 'HTTP', 'SQLite']
    },

    {
      id: 'ios-swiftui-ar-shopping',
      name: 'iOS AR Shopping App',
      description: 'Augmented reality shopping experience with 3D product visualization and virtual try-on features',
      category: 'AR/Shopping',
      difficulty: 'Expert',
      tags: ['SwiftUI', 'ARKit', 'RealityKit', '3D', 'Shopping'],
      icon: <Smartphone className="w-8 h-8 text-purple-400" />,
      estimatedTime: '6-8 weeks',
      platforms: ['iOS'],
      rating: 4.9,
      downloads: 1234,
      files: {
        'ContentView.swift': {
          content: `import SwiftUI
import ARKit
import RealityKit

struct ContentView: View {
    @StateObject private var arManager = ARManager()
    @StateObject private var shoppingCart = ShoppingCart()
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView()
                .tabItem {
                    Image(systemName: "house.fill")
                    Text("Home")
                }
                .tag(0)
            
            ARView()
                .tabItem {
                    Image(systemName: "camera.viewfinder")
                    Text("AR Try-On")
                }
                .tag(1)
            
            CartView()
                .tabItem {
                    Image(systemName: "cart.fill")
                    Text("Cart")
                }
                .tag(2)
            
            ProfileView()
                .tabItem {
                    Image(systemName: "person.fill")
                    Text("Profile")
                }
                .tag(3)
        }
        .environmentObject(arManager)
        .environmentObject(shoppingCart)
    }
}`,
          language: 'swift'
        }
      },
      features: [
        'AR product visualization',
        'Virtual try-on for clothing',
        '3D product models',
        'Room placement preview',
        'Social AR sharing',
        'Size recommendation AI',
        'Real-time lighting',
        'Multi-user AR sessions'
      ],
      useCase: 'Revolutionary shopping experience using augmented reality for product visualization',
      techStack: ['SwiftUI', 'ARKit', 'RealityKit', 'Core ML', 'Vision', 'Metal']
    },

    {
      id: 'android-kotlin-iot-home',
      name: 'Android IoT Smart Home',
      description: 'Comprehensive smart home control app with device management, automation, and energy monitoring',
      category: 'IoT/Smart Home',
      difficulty: 'Advanced',
      tags: ['Kotlin', 'IoT', 'Bluetooth', 'WiFi', 'Automation'],
      icon: <Settings className="w-8 h-8 text-blue-400" />,
      estimatedTime: '4-6 weeks',
      platforms: ['Android'],
      rating: 4.7,
      downloads: 2890,
      files: {
        'MainActivity.kt': {
          content: `package com.rustyclint.smarthome

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import com.rustyclint.smarthome.ui.theme.SmartHomeTheme
import com.rustyclint.smarthome.ui.screens.MainScreen
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    
    private val bluetoothManager by lazy {
        getSystemService(BluetoothManager::class.java)
    }
    
    private val bluetoothAdapter by lazy {
        bluetoothManager?.adapter
    }
    
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            initializeConnections()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        requestPermissions()
        
        setContent {
            SmartHomeTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
    }
    
    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_WIFI_STATE,
            Manifest.permission.CHANGE_WIFI_STATE
        )
        
        val permissionsToRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (permissionsToRequest.isNotEmpty()) {
            permissionLauncher.launch(permissionsToRequest.toTypedArray())
        } else {
            initializeConnections()
        }
    }
    
    private fun initializeConnections() {
        // Initialize Bluetooth and WiFi connections
        // Start device discovery
    }
}`,
          language: 'kotlin'
        }
      },
      features: [
        'Device discovery and pairing',
        'Voice control integration',
        'Energy usage monitoring',
        'Automation rules engine',
        'Security camera integration',
        'Weather-based automation',
        'Remote access via cloud',
        'Family member management'
      ],
      useCase: 'Complete smart home management system with IoT device control and automation',
      techStack: ['Kotlin', 'Jetpack Compose', 'Bluetooth', 'WiFi', 'MQTT', 'Room', 'Hilt']
    },

    {
      id: 'react-native-language-learning',
      name: 'React Native Language Learning',
      description: 'Interactive language learning app with speech recognition, AI tutoring, and gamification',
      category: 'Education',
      difficulty: 'Advanced',
      tags: ['React Native', 'AI', 'Speech', 'Gamification', 'Education'],
      icon: <BookOpen className="w-8 h-8 text-indigo-400" />,
      estimatedTime: '5-6 weeks',
      platforms: ['React Native', 'iOS', 'Android'],
      rating: 4.8,
      downloads: 3456,
      files: {
        'App.tsx': {
          content: `import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider } from 'react-redux';
import Icon from 'react-native-vector-icons/Ionicons';
import { store } from './src/store';
import LessonsScreen from './src/screens/LessonsScreen';
import PracticeScreen from './src/screens/PracticeScreen';
import ProgressScreen from './src/screens/ProgressScreen';
import ProfileScreen from './src/screens/ProfileScreen';
import { colors } from './src/theme/colors';

const Tab = createBottomTabNavigator();

const App = () => {
  return (
    <Provider store={store}>
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            tabBarIcon: ({ focused, color, size }) => {
              let iconName: string;

              switch (route.name) {
                case 'Lessons':
                  iconName = focused ? 'book' : 'book-outline';
                  break;
                case 'Practice':
                  iconName = focused ? 'mic' : 'mic-outline';
                  break;
                case 'Progress':
                  iconName = focused ? 'stats-chart' : 'stats-chart-outline';
                  break;
                case 'Profile':
                  iconName = focused ? 'person' : 'person-outline';
                  break;
                default:
                  iconName = 'book-outline';
              }

              return <Icon name={iconName} size={size} color={color} />;
            },
            tabBarActiveTintColor: colors.primary,
            tabBarInactiveTintColor: colors.gray,
            headerShown: false,
          })}
        >
          <Tab.Screen name="Lessons" component={LessonsScreen} />
          <Tab.Screen name="Practice" component={PracticeScreen} />
          <Tab.Screen name="Progress" component={ProgressScreen} />
          <Tab.Screen name="Profile" component={ProfileScreen} />
        </Tab.Navigator>
      </NavigationContainer>
    </Provider>
  );
};

export default App;`,
          language: 'typescript'
        }
      },
      features: [
        'Interactive lessons with AI',
        'Speech recognition practice',
        'Pronunciation feedback',
        'Gamified learning paths',
        'Offline lesson downloads',
        'Progress tracking',
        'Social learning features',
        'Adaptive difficulty'
      ],
      useCase: 'Comprehensive language learning platform with AI-powered tutoring and speech recognition',
      techStack: ['React Native', 'TypeScript', 'Speech Recognition', 'AI/ML', 'Redux', 'Animations']
    }
  ];

  const categories = ['all', 'Social', 'Finance', 'E-commerce', 'Health & Fitness', 'Travel', 'Entertainment', 'Food & Delivery', 'AR/Shopping', 'IoT/Smart Home', 'Education'];
  const difficulties = ['all', 'Beginner', 'Intermediate', 'Advanced', 'Expert'];
  const platforms = ['all', 'iOS', 'Android', 'Flutter', 'React Native'];

  const filteredTemplates = templates.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
    const matchesDifficulty = selectedDifficulty === 'all' || template.difficulty === selectedDifficulty;
    const matchesPlatform = selectedPlatform === 'all' || template.platforms.includes(selectedPlatform as any);
    
    return matchesSearch && matchesCategory && matchesDifficulty && matchesPlatform;
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

  const getPlatformIcon = (platform: string) => {
    switch (platform) {
      case 'iOS': return <Smartphone className="w-4 h-4 text-blue-400" />;
      case 'Android': return <Smartphone className="w-4 h-4 text-green-400" />;
      case 'Flutter': return <Tablet className="w-4 h-4 text-cyan-400" />;
      case 'React Native': return <Monitor className="w-4 h-4 text-purple-400" />;
      case 'Web': return <Globe className="w-4 h-4 text-orange-400" />;
      default: return <Code className="w-4 h-4 text-gray-400" />;
    }
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-lg max-w-7xl w-full h-[90vh] flex flex-col border border-gray-700">
        {/* Header */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-orange-600 rounded-lg">
                <Smartphone className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Mobile Development Templates</h2>
                <p className="text-gray-400">Choose from 10+ professional mobile app templates</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors text-xl"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Search and Filters */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search templates..."
                className="w-full pl-10 pr-4 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-orange-500 focus:outline-none"
              />
            </div>
            
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-orange-500 focus:outline-none"
            >
              {categories.map(category => (
                <option key={category} value={category}>
                  {category === 'all' ? 'All Categories' : category}
                </option>
              ))}
            </select>
            
            <select
              value={selectedDifficulty}
              onChange={(e) => setSelectedDifficulty(e.target.value)}
              className="px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-orange-500 focus:outline-none"
            >
              {difficulties.map(difficulty => (
                <option key={difficulty} value={difficulty}>
                  {difficulty === 'all' ? 'All Levels' : difficulty}
                </option>
              ))}
            </select>
            
            <select
              value={selectedPlatform}
              onChange={(e) => setSelectedPlatform(e.target.value)}
              className="px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-orange-500 focus:outline-none"
            >
              {platforms.map(platform => (
                <option key={platform} value={platform}>
                  {platform === 'all' ? 'All Platforms' : platform}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Templates Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredTemplates.map((template) => (
              <div
                key={template.id}
                className="bg-gray-700 rounded-lg border border-gray-600 hover:border-orange-500 transition-all duration-300 transform hover:scale-105 cursor-pointer"
                onClick={() => onSelectTemplate(template)}
              >
                <div className="p-6">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {template.icon}
                      <div>
                        <h3 className="text-lg font-semibold text-white">{template.name}</h3>
                        <span className={`inline-block px-2 py-1 rounded text-xs ${getDifficultyColor(template.difficulty)}`}>
                          {template.difficulty}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Star className="w-4 h-4 text-yellow-400 fill-current" />
                      <span className="text-sm text-gray-300">{template.rating}</span>
                    </div>
                  </div>

                  {/* Description */}
                  <p className="text-gray-300 text-sm mb-4 line-clamp-3">{template.description}</p>

                  {/* Platforms */}
                  <div className="flex items-center space-x-2 mb-4">
                    {template.platforms.map((platform, index) => (
                      <div key={index} className="flex items-center space-x-1">
                        {getPlatformIcon(platform)}
                        <span className="text-xs text-gray-400">{platform}</span>
                      </div>
                    ))}
                  </div>

                  {/* Tags */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {template.tags.slice(0, 3).map((tag, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-600 text-gray-300 rounded text-xs">
                        {tag}
                      </span>
                    ))}
                    {template.tags.length > 3 && (
                      <span className="px-2 py-1 bg-gray-600 text-gray-400 rounded text-xs">
                        +{template.tags.length - 3} more
                      </span>
                    )}
                  </div>

                  {/* Stats */}
                  <div className="flex items-center justify-between text-sm text-gray-400">
                    <div className="flex items-center space-x-1">
                      <Clock className="w-4 h-4" />
                      <span>{template.estimatedTime}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Download className="w-4 h-4" />
                      <span>{template.downloads.toLocaleString()}</span>
                    </div>
                  </div>

                  {/* Features Preview */}
                  <div className="mt-4 pt-4 border-t border-gray-600">
                    <p className="text-xs text-gray-500 mb-2">Key Features:</p>
                    <div className="space-y-1">
                      {template.features.slice(0, 3).map((feature, index) => (
                        <div key={index} className="flex items-center space-x-2">
                          <CheckCircle className="w-3 h-3 text-green-400" />
                          <span className="text-xs text-gray-300">{feature}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Action Button */}
                <div className="px-6 pb-6">
                  <button className="w-full flex items-center justify-center space-x-2 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg font-medium transition-colors">
                    <Play className="w-4 h-4" />
                    
                    <span>Use Template</span>
                  </button>
                </div>
              </div>
            ))}
          </div>

          {filteredTemplates.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12">
              <Search className="w-12 h-12 text-gray-500 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">No templates found</h3>
              <p className="text-gray-400 text-center max-w-md">
                Try adjusting your search or filters to find what you're looking for.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProjectTemplates;