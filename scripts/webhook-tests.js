#!/usr/bin/env node

/**
 * Stripe Webhook Testing Utility
 * Tests webhook handling and database synchronization
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
    webhookSecret: process.env.STRIPE_WEBHOOK_SECRET
  },
  supabase: {
    url: process.env.VITE_SUPABASE_URL,
    serviceKey: process.env.SUPABASE_SERVICE_ROLE_KEY
  }
};

const stripe = new Stripe(config.stripe.secretKey);
const supabase = createClient(config.supabase.url, config.supabase.serviceKey);

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

async function createWebhookEvent(type, data) {
  return {
    id: `evt_test_${Date.now()}`,
    object: 'event',
    api_version: '2023-10-16',
    created: Math.floor(Date.now() / 1000),
    data: { object: data },
    livemode: false,
    pending_webhooks: 1,
    request: { id: null, idempotency_key: null },
    type
  };
}

async function sendWebhookEvent(event) {
  const webhookUrl = `${config.supabase.url}/functions/v1/stripe-webhook`;
  const payload = JSON.stringify(event);
  
  // Create webhook signature
  const signature = stripe.webhooks.generateTestHeaderString({
    payload,
    secret: config.stripe.webhookSecret
  });
  
  const response = await fetch(webhookUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'stripe-signature': signature
    },
    body: payload
  });
  
  return response;
}

async function testSubscriptionCreatedWebhook() {
  log('Testing subscription.created webhook...', 'info');
  
  // Create test customer
  const customer = await stripe.customers.create({
    email: 'webhook-test@example.com',
    metadata: { test: 'true' }
  });
  
  // Create subscription data
  const subscriptionData = {
    id: 'sub_test_123',
    object: 'subscription',
    customer: customer.id,
    status: 'active',
    current_period_start: Math.floor(Date.now() / 1000),
    current_period_end: Math.floor(Date.now() / 1000) + 2592000, // 30 days
    items: {
      data: [{
        price: {
          id: 'price_test_123',
          unit_amount: 1999,
          currency: 'usd'
        }
      }]
    },
    default_payment_method: {
      card: {
        brand: 'visa',
        last4: '4242'
      }
    }
  };
  
  // Create webhook event
  const event = await createWebhookEvent('customer.subscription.created', subscriptionData);
  
  // Send webhook
  const response = await sendWebhookEvent(event);
  
  if (!response.ok) {
    throw new Error(`Webhook failed with status: ${response.status}`);
  }
  
  // Wait for processing
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Verify database was updated
  const { data: subscription } = await supabase
    .from('stripe_subscriptions')
    .select('*')
    .eq('customer_id', customer.id)
    .single();
  
  if (!subscription || subscription.status !== 'active') {
    throw new Error('Subscription not found or incorrect status in database');
  }
  
  // Clean up
  await supabase.from('stripe_subscriptions').delete().eq('customer_id', customer.id);
  await stripe.customers.del(customer.id);
  
  log('✅ Subscription created webhook test passed', 'success');
}

async function testPaymentSucceededWebhook() {
  log('Testing payment_intent.succeeded webhook...', 'info');
  
  // Create test customer
  const customer = await stripe.customers.create({
    email: 'payment-webhook-test@example.com',
    metadata: { test: 'true' }
  });
  
  // Create payment intent data
  const paymentIntentData = {
    id: 'pi_test_123',
    object: 'payment_intent',
    amount: 2000,
    currency: 'usd',
    customer: customer.id,
    status: 'succeeded',
    invoice: null // One-time payment
  };
  
  // Create webhook event
  const event = await createWebhookEvent('payment_intent.succeeded', paymentIntentData);
  
  // Send webhook
  const response = await sendWebhookEvent(event);
  
  if (!response.ok) {
    throw new Error(`Webhook failed with status: ${response.status}`);
  }
  
  log('✅ Payment succeeded webhook test passed', 'success');
  
  // Clean up
  await stripe.customers.del(customer.id);
}

async function testCheckoutSessionCompletedWebhook() {
  log('Testing checkout.session.completed webhook...', 'info');
  
  // Create test customer
  const customer = await stripe.customers.create({
    email: 'checkout-webhook-test@example.com',
    metadata: { test: 'true' }
  });
  
  // Create checkout session data
  const sessionData = {
    id: 'cs_test_123',
    object: 'checkout.session',
    customer: customer.id,
    mode: 'payment',
    payment_status: 'paid',
    payment_intent: 'pi_test_456',
    amount_subtotal: 1500,
    amount_total: 1500,
    currency: 'usd'
  };
  
  // Create webhook event
  const event = await createWebhookEvent('checkout.session.completed', sessionData);
  
  // Send webhook
  const response = await sendWebhookEvent(event);
  
  if (!response.ok) {
    throw new Error(`Webhook failed with status: ${response.status}`);
  }
  
  // Wait for processing
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Verify order was created
  const { data: order } = await supabase
    .from('stripe_orders')
    .select('*')
    .eq('customer_id', customer.id)
    .single();
  
  if (!order || order.status !== 'completed') {
    throw new Error('Order not found or incorrect status in database');
  }
  
  // Clean up
  await supabase.from('stripe_orders').delete().eq('customer_id', customer.id);
  await stripe.customers.del(customer.id);
  
  log('✅ Checkout session completed webhook test passed', 'success');
}

async function testSubscriptionUpdatedWebhook() {
  log('Testing customer.subscription.updated webhook...', 'info');
  
  // Create test customer and subscription record
  const customer = await stripe.customers.create({
    email: 'sub-update-test@example.com',
    metadata: { test: 'true' }
  });
  
  // Insert initial subscription
  await supabase.from('stripe_subscriptions').insert({
    customer_id: customer.id,
    subscription_id: 'sub_test_update',
    status: 'active'
  });
  
  // Create updated subscription data
  const subscriptionData = {
    id: 'sub_test_update',
    object: 'subscription',
    customer: customer.id,
    status: 'past_due',
    current_period_start: Math.floor(Date.now() / 1000),
    current_period_end: Math.floor(Date.now() / 1000) + 2592000,
    items: {
      data: [{
        price: {
          id: 'price_test_123',
          unit_amount: 1999,
          currency: 'usd'
        }
      }]
    }
  };
  
  // Create webhook event
  const event = await createWebhookEvent('customer.subscription.updated', subscriptionData);
  
  // Send webhook
  const response = await sendWebhookEvent(event);
  
  if (!response.ok) {
    throw new Error(`Webhook failed with status: ${response.status}`);
  }
  
  // Wait for processing
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Verify subscription was updated
  const { data: subscription } = await supabase
    .from('stripe_subscriptions')
    .select('*')
    .eq('customer_id', customer.id)
    .single();
  
  if (!subscription || subscription.status !== 'past_due') {
    throw new Error('Subscription status not updated correctly');
  }
  
  // Clean up
  await supabase.from('stripe_subscriptions').delete().eq('customer_id', customer.id);
  await stripe.customers.del(customer.id);
  
  log('✅ Subscription updated webhook test passed', 'success');
}

async function testInvalidWebhookSignature() {
  log('Testing invalid webhook signature handling...', 'info');
  
  const webhookUrl = `${config.supabase.url}/functions/v1/stripe-webhook`;
  const payload = JSON.stringify({ test: 'invalid' });
  
  const response = await fetch(webhookUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'stripe-signature': 'invalid_signature'
    },
    body: payload
  });
  
  if (response.status !== 400) {
    throw new Error(`Expected 400 status for invalid signature, got ${response.status}`);
  }
  
  log('✅ Invalid webhook signature test passed', 'success');
}

async function runWebhookTests() {
  log('🔗 Starting Webhook Tests', 'header');
  log('========================', 'header');
  
  // Validate configuration
  if (!config.stripe.secretKey || !config.stripe.webhookSecret) {
    log('❌ Stripe configuration incomplete', 'error');
    process.exit(1);
  }
  
  if (!config.supabase.url || !config.supabase.serviceKey) {
    log('❌ Supabase configuration incomplete', 'error');
    process.exit(1);
  }
  
  try {
    await testSubscriptionCreatedWebhook();
    await testPaymentSucceededWebhook();
    await testCheckoutSessionCompletedWebhook();
    await testSubscriptionUpdatedWebhook();
    await testInvalidWebhookSignature();
    
    log('', 'info');
    log('🎉 All webhook tests passed!', 'success');
  } catch (error) {
    log(`❌ Webhook test failed: ${error.message}`, 'error');
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runWebhookTests();
}

export { runWebhookTests };