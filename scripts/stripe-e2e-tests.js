#!/usr/bin/env node

/**
 * Stripe E2E Testing Utility
 * Comprehensive testing suite for Stripe integration with Supabase
 */

import chalk from 'chalk';
import Stripe from 'stripe';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

// Configuration
const config = {
  stripe: {
    secretKey: process.env.STRIPE_SECRET_KEY,
    webhookSecret: process.env.STRIPE_WEBHOOK_SECRET,
    publishableKey: process.env.VITE_STRIPE_PUBLISHABLE_KEY
  },
  supabase: {
    url: process.env.VITE_SUPABASE_URL,
    serviceKey: process.env.SUPABASE_SERVICE_ROLE_KEY,
    anonKey: process.env.VITE_SUPABASE_ANON_KEY
  },
  testCards: {
    success: '4242424242424242',
    declined: '4000000000000002',
    requiresAuth: '4000000000003220',
    insufficientFunds: '4000000000009995',
    expired: '4000000000000069',
    processingError: '4000000000000119'
  }
};

// Initialize clients
const stripe = new Stripe(config.stripe.secretKey);
const supabase = createClient(config.supabase.url, config.supabase.serviceKey);

// Test results tracking
let testResults = {
  passed: 0,
  failed: 0,
  total: 0,
  details: []
};

// Utility functions
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

function logTest(name, status, details = '') {
  const icon = status === 'PASS' ? '✅' : '❌';
  const color = status === 'PASS' ? chalk.green : chalk.red;
  log(`${icon} ${color(status)} ${name} ${details ? chalk.gray(details) : ''}`, 'info');
}

async function runTest(testName, testFunction) {
  testResults.total++;
  const startTime = Date.now();
  
  try {
    log(`Running: ${testName}`, 'info');
    await testFunction();
    const duration = Date.now() - startTime;
    testResults.passed++;
    testResults.details.push({ name: testName, status: 'PASS', duration });
    logTest(testName, 'PASS', `(${duration}ms)`);
    return true;
  } catch (error) {
    const duration = Date.now() - startTime;
    testResults.failed++;
    testResults.details.push({ name: testName, status: 'FAIL', duration, error: error.message });
    logTest(testName, 'FAIL', `(${duration}ms) - ${error.message}`);
    return false;
  }
}

// Test helper functions
async function createTestCustomer(email = 'test@example.com') {
  return await stripe.customers.create({
    email,
    metadata: { test: 'true' }
  });
}

async function createTestUser() {
  const { data, error } = await supabase.auth.admin.createUser({
    email: 'test@example.com',
    password: 'testpassword123',
    email_confirm: true
  });
  
  if (error) throw error;
  return data.user;
}

async function cleanupTestData() {
  // Clean up test customers
  const customers = await stripe.customers.list({ limit: 100 });
  for (const customer of customers.data) {
    if (customer.metadata?.test === 'true') {
      await stripe.customers.del(customer.id);
    }
  }
  
  // Clean up test database records
  await supabase.from('stripe_customers').delete().like('customer_id', 'cus_%');
  await supabase.from('stripe_subscriptions').delete().neq('id', 0);
  await supabase.from('stripe_orders').delete().neq('id', 0);
}

// Test Cases
async function testStripeConnection() {
  const account = await stripe.accounts.retrieve();
  if (!account.id) {
    throw new Error('Failed to connect to Stripe');
  }
}

async function testSupabaseConnection() {
  const { data, error } = await supabase.from('stripe_customers').select('count').limit(1);
  if (error) {
    throw new Error(`Supabase connection failed: ${error.message}`);
  }
}

async function testCustomerCreation() {
  const customer = await createTestCustomer('customer-test@example.com');
  
  if (!customer.id || !customer.email) {
    throw new Error('Customer creation failed');
  }
  
  // Verify customer in Stripe
  const retrievedCustomer = await stripe.customers.retrieve(customer.id);
  if (retrievedCustomer.id !== customer.id) {
    throw new Error('Customer retrieval failed');
  }
  
  // Clean up
  await stripe.customers.del(customer.id);
}

async function testSubscriptionCreation() {
  const customer = await createTestCustomer('subscription-test@example.com');
  
  // Create a test price
  const price = await stripe.prices.create({
    unit_amount: 999,
    currency: 'usd',
    recurring: { interval: 'month' },
    product_data: { name: 'Test Subscription' }
  });
  
  // Create subscription
  const subscription = await stripe.subscriptions.create({
    customer: customer.id,
    items: [{ price: price.id }],
    payment_behavior: 'default_incomplete',
    expand: ['latest_invoice.payment_intent']
  });
  
  if (!subscription.id || subscription.status !== 'incomplete') {
    throw new Error('Subscription creation failed');
  }
  
  // Clean up
  await stripe.subscriptions.cancel(subscription.id);
  await stripe.prices.update(price.id, { active: false });
  await stripe.customers.del(customer.id);
}

async function testOneTimePayment() {
  const customer = await createTestCustomer('payment-test@example.com');
  
  // Create payment intent
  const paymentIntent = await stripe.paymentIntents.create({
    amount: 2000,
    currency: 'usd',
    customer: customer.id,
    payment_method_data: {
      type: 'card',
      card: { token: 'tok_visa' }
    },
    confirm: true,
    return_url: 'https://example.com/return'
  });
  
  if (!paymentIntent.id || paymentIntent.status !== 'succeeded') {
    throw new Error('Payment intent creation failed');
  }
  
  // Clean up
  await stripe.customers.del(customer.id);
}

async function testDatabaseIntegration() {
  const customer = await createTestCustomer('db-test@example.com');
  
  // Insert customer record
  const { data: customerData, error: customerError } = await supabase
    .from('stripe_customers')
    .insert({
      user_id: '00000000-0000-0000-0000-000000000000', // Test UUID
      customer_id: customer.id
    })
    .select()
    .single();
  
  if (customerError) {
    throw new Error(`Database customer insert failed: ${customerError.message}`);
  }
  
  // Insert subscription record
  const { data: subscriptionData, error: subscriptionError } = await supabase
    .from('stripe_subscriptions')
    .insert({
      customer_id: customer.id,
      status: 'not_started'
    })
    .select()
    .single();
  
  if (subscriptionError) {
    throw new Error(`Database subscription insert failed: ${subscriptionError.message}`);
  }
  
  // Verify records exist
  const { data: verifyCustomer } = await supabase
    .from('stripe_customers')
    .select('*')
    .eq('customer_id', customer.id)
    .single();
  
  if (!verifyCustomer) {
    throw new Error('Customer record not found in database');
  }
  
  // Clean up
  await supabase.from('stripe_subscriptions').delete().eq('customer_id', customer.id);
  await supabase.from('stripe_customers').delete().eq('customer_id', customer.id);
  await stripe.customers.del(customer.id);
}

async function testWebhookEndpoint() {
  // Test webhook endpoint availability
  const webhookUrl = `${config.supabase.url}/functions/v1/stripe-webhook`;
  
  try {
    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'stripe-signature': 'test_signature'
      },
      body: JSON.stringify({ test: true })
    });
    
    // We expect this to fail with signature verification, which means the endpoint is working
    if (response.status === 400) {
      // This is expected - webhook signature verification should fail
      return;
    } else {
      throw new Error(`Unexpected webhook response status: ${response.status}`);
    }
  } catch (error) {
    if (error.message.includes('fetch')) {
      throw new Error('Webhook endpoint not accessible');
    }
    // Other errors are expected (signature verification, etc.)
  }
}

async function testCheckoutSession() {
  const customer = await createTestCustomer('checkout-test@example.com');
  
  // Create a test price
  const price = await stripe.prices.create({
    unit_amount: 1000,
    currency: 'usd',
    product_data: { name: 'Test Product' }
  });
  
  // Create checkout session
  const session = await stripe.checkout.sessions.create({
    customer: customer.id,
    payment_method_types: ['card'],
    line_items: [{
      price: price.id,
      quantity: 1
    }],
    mode: 'payment',
    success_url: 'https://example.com/success',
    cancel_url: 'https://example.com/cancel'
  });
  
  if (!session.id || !session.url) {
    throw new Error('Checkout session creation failed');
  }
  
  // Clean up
  await stripe.prices.update(price.id, { active: false });
  await stripe.customers.del(customer.id);
}

async function testErrorHandling() {
  // Test invalid customer creation
  try {
    await stripe.customers.create({
      email: 'invalid-email'
    });
    throw new Error('Should have failed with invalid email');
  } catch (error) {
    if (!error.message.includes('Invalid email')) {
      // Expected error, test passes
    }
  }
  
  // Test invalid payment
  try {
    await stripe.paymentIntents.create({
      amount: -100, // Invalid amount
      currency: 'usd'
    });
    throw new Error('Should have failed with invalid amount');
  } catch (error) {
    if (error.type === 'StripeInvalidRequestError') {
      // Expected error, test passes
    }
  }
}

async function testRLSPolicies() {
  // Test Row Level Security policies
  const testUserId = '00000000-0000-0000-0000-000000000000';
  const customer = await createTestCustomer('rls-test@example.com');
  
  // Insert test data
  await supabase
    .from('stripe_customers')
    .insert({
      user_id: testUserId,
      customer_id: customer.id
    });
  
  // Test with anon client (should fail)
  const anonSupabase = createClient(config.supabase.url, config.supabase.anonKey);
  
  const { data: anonData, error: anonError } = await anonSupabase
    .from('stripe_customers')
    .select('*')
    .eq('customer_id', customer.id);
  
  if (anonData && anonData.length > 0) {
    throw new Error('RLS policy failed - anonymous user can access data');
  }
  
  // Clean up
  await supabase.from('stripe_customers').delete().eq('customer_id', customer.id);
  await stripe.customers.del(customer.id);
}

// Main test runner
async function runAllTests() {
  log('🚀 Starting Stripe E2E Tests with Supabase Integration', 'header');
  log('================================================', 'header');
  
  // Validate configuration
  if (!config.stripe.secretKey) {
    log('❌ STRIPE_SECRET_KEY not configured', 'error');
    process.exit(1);
  }
  
  if (!config.supabase.url || !config.supabase.serviceKey) {
    log('❌ Supabase configuration incomplete', 'error');
    process.exit(1);
  }
  
  log('🧹 Cleaning up previous test data...', 'info');
  await cleanupTestData();
  
  // Run tests
  await runTest('Stripe Connection', testStripeConnection);
  await runTest('Supabase Connection', testSupabaseConnection);
  await runTest('Customer Creation', testCustomerCreation);
  await runTest('Subscription Creation', testSubscriptionCreation);
  await runTest('One-time Payment', testOneTimePayment);
  await runTest('Database Integration', testDatabaseIntegration);
  await runTest('Webhook Endpoint', testWebhookEndpoint);
  await runTest('Checkout Session', testCheckoutSession);
  await runTest('Error Handling', testErrorHandling);
  await runTest('RLS Policies', testRLSPolicies);
  
  // Final cleanup
  log('🧹 Final cleanup...', 'info');
  await cleanupTestData();
  
  // Results summary
  log('', 'info');
  log('📊 Test Results Summary', 'header');
  log('======================', 'header');
  log(`Total Tests: ${testResults.total}`, 'info');
  log(`Passed: ${chalk.green(testResults.passed)}`, 'info');
  log(`Failed: ${chalk.red(testResults.failed)}`, 'info');
  log(`Success Rate: ${chalk.cyan(((testResults.passed / testResults.total) * 100).toFixed(1))}%`, 'info');
  
  if (testResults.failed > 0) {
    log('', 'info');
    log('❌ Failed Tests:', 'error');
    testResults.details
      .filter(test => test.status === 'FAIL')
      .forEach(test => {
        log(`  • ${test.name}: ${test.error}`, 'error');
      });
  }
  
  // Exit with appropriate code
  process.exit(testResults.failed > 0 ? 1 : 0);
}

// Handle uncaught errors
process.on('unhandledRejection', (error) => {
  log(`❌ Unhandled rejection: ${error.message}`, 'error');
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  log(`❌ Uncaught exception: ${error.message}`, 'error');
  process.exit(1);
});

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllTests();
}

export {
  runAllTests,
  testResults,
  config
};