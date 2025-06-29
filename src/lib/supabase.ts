import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

// Memoize the check to avoid recalculating
const isValidUrl = (url: string) => url && url !== 'https://your-production-project.supabase.co' && url.includes('supabase.co');
const isValidKey = (key: string) => key && key !== 'your_production_supabase_anon_key';

// Check if we have valid Supabase configuration
export const isSupabaseConfigured = isValidUrl(supabaseUrl) && isValidKey(supabaseAnonKey);

// Use real Supabase if configured, otherwise use safe demo values
const finalUrl = isSupabaseConfigured ? supabaseUrl : 'https://example.com';
const finalKey = isSupabaseConfigured ? supabaseAnonKey : 'dummy_key';

// Create a single instance to avoid multiple client warnings
let supabaseInstance: any = null;

export const supabase = (() => {
  if (!supabaseInstance) {
    if (!isSupabaseConfigured) {
      // Only log in development
      if (import.meta.env.DEV) {
        console.info('Supabase not configured. Running in demo mode.');
      }
    }
    supabaseInstance = createClient(finalUrl, finalKey);
  }
  return supabaseInstance;
})();