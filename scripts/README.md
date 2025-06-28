# Stripe E2E Testing Suite

This directory contains comprehensive end-to-end testing utilities for the Stripe and Supabase integration.

## Overview

The testing suite consists of several Node.js scripts that validate different aspects of the payment integration:

- **Database Tests** - Validates Supabase schema, RLS policies, and CRUD operations
- **Stripe E2E Tests** - Tests payment flows, customer creation, and error handling
- **Webhook Tests** - Validates webhook handling and database synchronization
- **Test Runner** - Orchestrates all tests and provides comprehensive reporting

## Prerequisites

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Stripe Configuration
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_...

# Supabase Configuration
VITE_SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
VITE_SUPABASE_ANON_KEY=eyJ...
```

### Dependencies

Install the required dependencies:

```bash
npm install
```

## Running Tests

### Run All Tests

```bash
npm run test:all
```

This runs all test suites in sequence and provides a comprehensive report.

### Run Individual Test Suites

```bash
# Database tests only
npm run test:db

# Stripe integration tests only
npm run test:stripe

# Webhook tests only
npm run test:webhooks
```

### Watch Mode

```bash
npm run test:stripe:watch
```

Runs Stripe tests in watch mode using nodemon.

## Test Suites

### 1. Database Tests (`database-tests.js`)

Tests the Supabase database integration:

- **Schema Validation** - Ensures all required tables exist
- **CRUD Operations** - Tests create, read, update, delete for all entities
- **RLS Policies** - Validates Row Level Security is working correctly
- **Views** - Tests database views for user data
- **Enum Types** - Validates custom enum types

### 2. Stripe E2E Tests (`stripe-e2e-tests.js`)

Tests the complete Stripe integration:

- **Connection Tests** - Validates Stripe and Supabase connectivity
- **Customer Management** - Tests customer creation and management
- **Subscription Flows** - Tests subscription creation and management
- **One-time Payments** - Tests payment intent creation
- **Checkout Sessions** - Tests checkout session creation
- **Error Handling** - Tests various error scenarios
- **Database Integration** - Tests data synchronization

### 3. Webhook Tests (`webhook-tests.js`)

Tests webhook handling and processing:

- **Subscription Events** - Tests subscription created/updated webhooks
- **Payment Events** - Tests payment succeeded webhooks
- **Checkout Events** - Tests checkout session completed webhooks
- **Error Handling** - Tests invalid signature handling
- **Database Sync** - Validates data is correctly synchronized

## Test Cards

The tests use Stripe's test card numbers:

- `4242424242424242` - Successful payment
- `4000000000000002` - Card declined
- `4000000000003220` - Requires 3D Secure authentication
- `4000000000009995` - Insufficient funds
- `4000000000000069` - Expired card
- `4000000000000119` - Processing error

## Output and Reporting

### Console Output

Tests provide real-time colored console output with:
- Timestamps for each operation
- Success/failure indicators
- Detailed error messages
- Performance metrics (duration)

### Test Results

Each test suite provides:
- Total tests run
- Pass/fail counts
- Success rate percentage
- Detailed failure information

### Export Results

The Stripe E2E tests can export results to JSON:

```bash
# Results are automatically saved to stripe-test-results-{timestamp}.json
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '18'
      - run: npm install
      - run: npm run test:all
        env:
          STRIPE_SECRET_KEY: ${{ secrets.STRIPE_SECRET_KEY }}
          STRIPE_WEBHOOK_SECRET: ${{ secrets.STRIPE_WEBHOOK_SECRET }}
          VITE_SUPABASE_URL: ${{ secrets.VITE_SUPABASE_URL }}
          SUPABASE_SERVICE_ROLE_KEY: ${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}
          VITE_SUPABASE_ANON_KEY: ${{ secrets.VITE_SUPABASE_ANON_KEY }}
```

## Troubleshooting

### Common Issues

1. **Environment Variables Not Set**
   - Ensure all required environment variables are configured
   - Check `.env` file exists and is properly formatted

2. **Stripe API Errors**
   - Verify Stripe keys are correct and have proper permissions
   - Check if test mode is enabled

3. **Supabase Connection Issues**
   - Verify Supabase URL and keys are correct
   - Check if RLS policies are properly configured

4. **Webhook Failures**
   - Ensure webhook endpoint is accessible
   - Verify webhook secret is correctly configured

### Debug Mode

Add debug logging by setting:

```bash
DEBUG=true npm run test:all
```

### Manual Testing

You can also run individual test functions by importing them:

```javascript
const { testCustomerCreation } = require('./stripe-e2e-tests');

testCustomerCreation()
  .then(() => console.log('Test passed'))
  .catch(console.error);
```

## Contributing

When adding new tests:

1. Follow the existing naming convention
2. Add proper error handling and cleanup
3. Include descriptive test names and comments
4. Update this README with new test descriptions

## Security Notes

- Never commit real API keys to version control
- Use test mode for all Stripe operations
- Ensure test data is properly cleaned up
- Validate RLS policies are working correctly

## Support

For issues with the testing suite:

1. Check the console output for detailed error messages
2. Verify all environment variables are set correctly
3. Ensure your Stripe and Supabase accounts are properly configured
4. Review the individual test files for specific requirements