#!/usr/bin/env node

/**
 * Comprehensive Test Runner
 * Orchestrates all E2E tests for Stripe and Supabase integration
 */

import chalk from 'chalk';
import { spawn } from 'child_process';
import path from 'path';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config();

function log(message, type = 'info') {
  const timestamp = new Date().toISOString();
  const colors = {
    info: chalk.blue,
    success: chalk.green,
    error: chalk.red,
    warning: chalk.yellow,
    header: chalk.cyan.bold
  };
  
  console.log(`${chalk.gray(timestamp)} ${colors[type](message)}`);
}

function runScript(scriptPath, args = []) {
  return new Promise((resolve, reject) => {
    const child = spawn('node', [scriptPath, ...args], {
      stdio: 'inherit',
      cwd: process.cwd()
    });
    
    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Script ${scriptPath} exited with code ${code}`));
      }
    });
    
    child.on('error', (error) => {
      reject(error);
    });
  });
}

async function runAllTests() {
  log('🚀 Starting Comprehensive E2E Test Suite', 'header');
  log('==========================================', 'header');
  
  const startTime = Date.now();
  let testsPassed = 0;
  let testsFailed = 0;
  
  const tests = [
    {
      name: 'Database Tests',
      script: path.join(__dirname, 'database-tests.js'),
      description: 'Testing Supabase database schema and RLS policies'
    },
    {
      name: 'Stripe E2E Tests',
      script: path.join(__dirname, 'stripe-e2e-tests.js'),
      description: 'Testing Stripe integration and payment flows'
    },
    {
      name: 'Webhook Tests',
      script: path.join(__dirname, 'webhook-tests.js'),
      description: 'Testing webhook handling and database synchronization'
    }
  ];
  
  for (const test of tests) {
    log('', 'info');
    log(`📋 Running: ${test.name}`, 'header');
    log(`Description: ${test.description}`, 'info');
    log('─'.repeat(50), 'info');
    
    try {
      await runScript(test.script);
      testsPassed++;
      log(`✅ ${test.name} completed successfully`, 'success');
    } catch (error) {
      testsFailed++;
      log(`❌ ${test.name} failed: ${error.message}`, 'error');
    }
  }
  
  const endTime = Date.now();
  const duration = ((endTime - startTime) / 1000).toFixed(2);
  
  log('', 'info');
  log('📊 Final Test Results', 'header');
  log('====================', 'header');
  log(`Total Test Suites: ${tests.length}`, 'info');
  log(`Passed: ${chalk.green(testsPassed)}`, 'info');
  log(`Failed: ${chalk.red(testsFailed)}`, 'info');
  log(`Duration: ${chalk.cyan(duration)}s`, 'info');
  
  if (testsFailed > 0) {
    log('', 'info');
    log('❌ Some tests failed. Please check the output above for details.', 'error');
    process.exit(1);
  } else {
    log('', 'info');
    log('🎉 All test suites passed successfully!', 'success');
    log('Your Stripe and Supabase integration is working correctly.', 'success');
    process.exit(0);
  }
}

// Handle command line arguments
const args = process.argv.slice(2);

if (args.includes('--help') || args.includes('-h')) {
  console.log(`
${chalk.cyan.bold('Stripe E2E Test Runner')}

Usage: node test-runner.js [options]

Options:
  --help, -h     Show this help message
  --watch, -w    Run tests in watch mode (not implemented)
  --verbose, -v  Enable verbose output

Test Suites:
  1. Database Tests - Validates Supabase schema and RLS policies
  2. Stripe E2E Tests - Tests payment flows and integrations
  3. Webhook Tests - Validates webhook handling and data sync

Environment Variables Required:
  - STRIPE_SECRET_KEY
  - STRIPE_WEBHOOK_SECRET
  - VITE_SUPABASE_URL
  - SUPABASE_SERVICE_ROLE_KEY
  - VITE_SUPABASE_ANON_KEY

Examples:
  npm run test:all
  node scripts/test-runner.js
  `);
  process.exit(0);
}

// Validate environment
const requiredEnvVars = [
  'STRIPE_SECRET_KEY',
  'STRIPE_WEBHOOK_SECRET',
  'VITE_SUPABASE_URL',
  'SUPABASE_SERVICE_ROLE_KEY',
  'VITE_SUPABASE_ANON_KEY'
];

const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar]);

if (missingEnvVars.length > 0) {
  log('❌ Missing required environment variables:', 'error');
  missingEnvVars.forEach(envVar => {
    log(`  • ${envVar}`, 'error');
  });
  log('', 'info');
  log('Please check your .env file and ensure all required variables are set.', 'warning');
  process.exit(1);
}

// Run tests
runAllTests().catch((error) => {
  log(`❌ Test runner failed: ${error.message}`, 'error');
  process.exit(1);
});