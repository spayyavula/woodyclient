import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// For demo purposes, provide safe fallback values that won't cause network errors
const defaultUrl = 'https://demo.supabase.co';
const defaultKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0';

// Always use demo values to prevent network errors
const finalUrl = supabaseUrl && supabaseUrl !== 'your_supabase_project_url_here' ? supabaseUrl : defaultUrl;
const finalKey = supabaseAnonKey && supabaseAnonKey !== 'your_production_supabase_anon_key' ? supabaseAnonKey : defaultKey;

if (!supabaseUrl || !supabaseAnonKey || supabaseUrl.includes('your_supabase_project_url_here')) {
  console.warn('Supabase environment variables not configured properly. Using demo mode.');
}

export const supabase = createClient(finalUrl, finalKey);