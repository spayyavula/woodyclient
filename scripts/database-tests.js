#!/usr/bin/env node

/**
 * Database Integration Tests
 * Tests Supabase database schema and RLS policies
 */

import chalk from 'chalk';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

// Configuration
const config = {
  supabase: {
    url: process.env.VITE_SUPABASE_URL,
    serviceKey: process.env.SUPABASE_SERVICE_ROLE_KEY,
    anonKey: process.env.VITE_SUPABASE_ANON_KEY
  }
};

const supabase = createClient(config.supabase.url, config.supabase.serviceKey);
const anonSupabase = createClient(config.supabase.url, config.supabase.anonKey);

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

async function testDatabaseSchema() {
  log('Testing database schema...', 'info');
  
  // Test stripe_customers table
  const { data: customers, error: customersError } = await supabase
    .from('stripe_customers')
    .select('*')
    .limit(1);
  
  if (customersError) {
    throw new Error(`stripe_customers table error: ${customersError.message}`);
  }
  
  // Test stripe_subscriptions table
  const { data: subscriptions, error: subscriptionsError } = await supabase
    .from('stripe_subscriptions')
    .select('*')
    .limit(1);
  
  if (subscriptionsError) {
    throw new Error(`stripe_subscriptions table error: ${subscriptionsError.message}`);
  }
  
  // Test stripe_orders table
  const { data: orders, error: ordersError } = await supabase
    .from('stripe_orders')
    .select('*')
    .limit(1);
  
  if (ordersError) {
    throw new Error(`stripe_orders table error: ${ordersError.message}`);
  }
  
  log('✅ Database schema test passed', 'success');
}

async function testCustomerCRUD() {
  log('Testing customer CRUD operations...', 'info');
  
  const testUserId = '00000000-0000-0000-0000-000000000000';
  const testCustomerId = 'cus_test_' + Date.now();
  
  // Create
  const { data: created, error: createError } = await supabase
    .from('stripe_customers')
    .insert({
      user_id: testUserId,
      customer_id: testCustomerId
    })
    .select()
    .single();
  
  if (createError) {
    throw new Error(`Customer create failed: ${createError.message}`);
  }
  
  // Read
  const { data: read, error: readError } = await supabase
    .from('stripe_customers')
    .select('*')
    .eq('customer_id', testCustomerId)
    .single();
  
  if (readError || !read) {
    throw new Error(`Customer read failed: ${readError?.message || 'Not found'}`);
  }
  
  // Update
  const { data: updated, error: updateError } = await supabase
    .from('stripe_customers')
    .update({ updated_at: new Date().toISOString() })
    .eq('customer_id', testCustomerId)
    .select()
    .single();
  
  if (updateError) {
    throw new Error(`Customer update failed: ${updateError.message}`);
  }
  
  // Delete
  const { error: deleteError } = await supabase
    .from('stripe_customers')
    .delete()
    .eq('customer_id', testCustomerId);
  
  if (deleteError) {
    throw new Error(`Customer delete failed: ${deleteError.message}`);
  }
  
  log('✅ Customer CRUD test passed', 'success');
}

async function testSubscriptionCRUD() {
  log('Testing subscription CRUD operations...', 'info');
  
  const testCustomerId = 'cus_test_' + Date.now();
  const testSubscriptionId = 'sub_test_' + Date.now();
  
  // Create
  const { data: created, error: createError } = await supabase
    .from('stripe_subscriptions')
    .insert({
      customer_id: testCustomerId,
      subscription_id: testSubscriptionId,
      status: 'active',
      price_id: 'price_test_123'
    })
    .select()
    .single();
  
  if (createError) {
    throw new Error(`Subscription create failed: ${createError.message}`);
  }
  
  // Read
  const { data: read, error: readError } = await supabase
    .from('stripe_subscriptions')
    .select('*')
    .eq('customer_id', testCustomerId)
    .single();
  
  if (readError || !read) {
    throw new Error(`Subscription read failed: ${readError?.message || 'Not found'}`);
  }
  
  // Update
  const { data: updated, error: updateError } = await supabase
    .from('stripe_subscriptions')
    .update({ status: 'past_due' })
    .eq('customer_id', testCustomerId)
    .select()
    .single();
  
  if (updateError || updated.status !== 'past_due') {
    throw new Error(`Subscription update failed: ${updateError?.message || 'Status not updated'}`);
  }
  
  // Delete
  const { error: deleteError } = await supabase
    .from('stripe_subscriptions')
    .delete()
    .eq('customer_id', testCustomerId);
  
  if (deleteError) {
    throw new Error(`Subscription delete failed: ${deleteError.message}`);
  }
  
  log('✅ Subscription CRUD test passed', 'success');
}

async function testOrderCRUD() {
  log('Testing order CRUD operations...', 'info');
  
  const testCustomerId = 'cus_test_' + Date.now();
  const testSessionId = 'cs_test_' + Date.now();
  const testPaymentIntentId = 'pi_test_' + Date.now();
  
  // Create
  const { data: created, error: createError } = await supabase
    .from('stripe_orders')
    .insert({
      checkout_session_id: testSessionId,
      payment_intent_id: testPaymentIntentId,
      customer_id: testCustomerId,
      amount_subtotal: 1000,
      amount_total: 1000,
      currency: 'usd',
      payment_status: 'paid',
      status: 'completed'
    })
    .select()
    .single();
  
  if (createError) {
    throw new Error(`Order create failed: ${createError.message}`);
  }
  
  // Read
  const { data: read, error: readError } = await supabase
    .from('stripe_orders')
    .select('*')
    .eq('customer_id', testCustomerId)
    .single();
  
  if (readError || !read) {
    throw new Error(`Order read failed: ${readError?.message || 'Not found'}`);
  }
  
  // Update
  const { data: updated, error: updateError } = await supabase
    .from('stripe_orders')
    .update({ status: 'canceled' })
    .eq('customer_id', testCustomerId)
    .select()
    .single();
  
  if (updateError || updated.status !== 'canceled') {
    throw new Error(`Order update failed: ${updateError?.message || 'Status not updated'}`);
  }
  
  // Delete
  const { error: deleteError } = await supabase
    .from('stripe_orders')
    .delete()
    .eq('customer_id', testCustomerId);
  
  if (deleteError) {
    throw new Error(`Order delete failed: ${deleteError.message}`);
  }
  
  log('✅ Order CRUD test passed', 'success');
}

async function testRLSPolicies() {
  log('Testing Row Level Security policies...', 'info');
  
  const testUserId = '00000000-0000-0000-0000-000000000000';
  const testCustomerId = 'cus_test_rls_' + Date.now();
  
  // Insert test data with service role
  await supabase.from('stripe_customers').insert({
    user_id: testUserId,
    customer_id: testCustomerId
  });
  
  await supabase.from('stripe_subscriptions').insert({
    customer_id: testCustomerId,
    status: 'active'
  });
  
  await supabase.from('stripe_orders').insert({
    checkout_session_id: 'cs_test_rls',
    payment_intent_id: 'pi_test_rls',
    customer_id: testCustomerId,
    amount_subtotal: 1000,
    amount_total: 1000,
    currency: 'usd',
    payment_status: 'paid'
  });
  
  // Test anonymous access (should be denied)
  const { data: anonCustomers } = await anonSupabase
    .from('stripe_customers')
    .select('*')
    .eq('customer_id', testCustomerId);
  
  if (anonCustomers && anonCustomers.length > 0) {
    throw new Error('RLS failed: Anonymous user can access customer data');
  }
  
  const { data: anonSubscriptions } = await anonSupabase
    .from('stripe_subscriptions')
    .select('*')
    .eq('customer_id', testCustomerId);
  
  if (anonSubscriptions && anonSubscriptions.length > 0) {
    throw new Error('RLS failed: Anonymous user can access subscription data');
  }
  
  const { data: anonOrders } = await anonSupabase
    .from('stripe_orders')
    .select('*')
    .eq('customer_id', testCustomerId);
  
  if (anonOrders && anonOrders.length > 0) {
    throw new Error('RLS failed: Anonymous user can access order data');
  }
  
  // Clean up
  await supabase.from('stripe_orders').delete().eq('customer_id', testCustomerId);
  await supabase.from('stripe_subscriptions').delete().eq('customer_id', testCustomerId);
  await supabase.from('stripe_customers').delete().eq('customer_id', testCustomerId);
  
  log('✅ RLS policies test passed', 'success');
}

async function testViews() {
  log('Testing database views...', 'info');
  
  // Test stripe_user_subscriptions view
  const { data: userSubs, error: userSubsError } = await supabase
    .from('stripe_user_subscriptions')
    .select('*')
    .limit(1);
  
  if (userSubsError) {
    throw new Error(`stripe_user_subscriptions view error: ${userSubsError.message}`);
  }
  
  // Test stripe_user_orders view
  const { data: userOrders, error: userOrdersError } = await supabase
    .from('stripe_user_orders')
    .select('*')
    .limit(1);
  
  if (userOrdersError) {
    throw new Error(`stripe_user_orders view error: ${userOrdersError.message}`);
  }
  
  log('✅ Database views test passed', 'success');
}

async function testEnumTypes() {
  log('Testing enum types...', 'info');
  
  const testCustomerId = 'cus_test_enum_' + Date.now();
  
  // Test subscription status enum
  const validStatuses = ['not_started', 'incomplete', 'incomplete_expired', 'trialing', 'active', 'past_due', 'canceled', 'unpaid', 'paused'];
  
  for (const status of validStatuses) {
    const { error } = await supabase
      .from('stripe_subscriptions')
      .insert({
        customer_id: testCustomerId + '_' + status,
        status: status
      });
    
    if (error) {
      throw new Error(`Invalid subscription status enum: ${status} - ${error.message}`);
    }
  }
  
  // Test order status enum
  const validOrderStatuses = ['pending', 'completed', 'canceled'];
  
  for (const status of validOrderStatuses) {
    const { error } = await supabase
      .from('stripe_orders')
      .insert({
        checkout_session_id: 'cs_test_enum_' + status,
        payment_intent_id: 'pi_test_enum_' + status,
        customer_id: testCustomerId + '_order_' + status,
        amount_subtotal: 1000,
        amount_total: 1000,
        currency: 'usd',
        payment_status: 'paid',
        status: status
      });
    
    if (error) {
      throw new Error(`Invalid order status enum: ${status} - ${error.message}`);
    }
  }
  
  // Clean up
  await supabase.from('stripe_subscriptions').delete().like('customer_id', testCustomerId + '%');
  await supabase.from('stripe_orders').delete().like('customer_id', testCustomerId + '%');
  
  log('✅ Enum types test passed', 'success');
}

async function runDatabaseTests() {
  log('🗄️  Starting Database Tests', 'header');
  log('==========================', 'header');
  
  // Validate configuration
  if (!config.supabase.url || !config.supabase.serviceKey) {
    log('❌ Supabase configuration incomplete', 'error');
    process.exit(1);
  }
  
  try {
    await testDatabaseSchema();
    await testCustomerCRUD();
    await testSubscriptionCRUD();
    await testOrderCRUD();
    await testRLSPolicies();
    await testViews();
    await testEnumTypes();
    
    log('', 'info');
    log('🎉 All database tests passed!', 'success');
  } catch (error) {
    log(`❌ Database test failed: ${error.message}`, 'error');
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runDatabaseTests();
}

export { runDatabaseTests };