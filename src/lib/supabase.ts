import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;


// Check if we have valid Supabase configuration
export const isSupabaseConfigured = supabaseUrl && 
  supabaseAnonKey && 
  supabaseUrl !== 'https://your-production-project.supabase.co' &&
  supabaseAnonKey !== 'your_production_supabase_anon_key' &&
  supabaseUrl.includes('supabase.co');

// Use real Supabase if configured, otherwise use safe demo values
const finalUrl = isSupabaseConfigured ? supabaseUrl : 'https://demo.supabase.co';
const finalKey = isSupabaseConfigured ? supabaseAnonKey : 'demo-key';

if (!isSupabaseConfigured) {
  console.warn('Supabase not configured. Running in demo mode.');
}

export const supabase = createClient(finalUrl, finalKey);