import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

// For demo purposes, provide fallback values
const defaultUrl = supabaseUrl || 'https://demo.supabase.co';
const defaultKey = supabaseAnonKey || 'demo-key';

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn('Supabase environment variables not configured. Using demo mode.');
}

export const supabase = createClient(defaultUrl, defaultKey);