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
const finalUrl = isSupabaseConfigured ? supabaseUrl : 'https://localhost:54321';
const finalKey = isSupabaseConfigured ? supabaseAnonKey : 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0';

// Create a single instance to avoid multiple client warnings
let supabaseInstance: any = null;

export const supabase = (() => {
  if (!supabaseInstance) {
    if (!isSupabaseConfigured) {
      console.warn('Supabase not configured. Running in demo mode.');
    }
    supabaseInstance = createClient(finalUrl, finalKey);
  }
  return supabaseInstance;
})();