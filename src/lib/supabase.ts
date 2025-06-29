import { createClient } from '@supabase/supabase-js';
import { getCachedData, CACHE_EXPIRATION, createCacheKey } from '../utils/cacheUtils';

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

/**
 * Enhanced Supabase query with caching
 * @param queryFn Function that returns a Supabase query
 * @param cacheKey Key to use for caching
 * @param expiration Cache expiration time
 */
export async function cachedQuery<T>(
  queryFn: () => Promise<{ data: T | null; error: any }>,
  cacheKey: string,
  expiration = CACHE_EXPIRATION.MEDIUM
): Promise<{ data: T | null; error: any }> {
  try {
    // Use getCachedData to handle caching logic
    const data = await getCachedData<T>(
      cacheKey,
      async () => {
        const { data, error } = await queryFn();
        if (error) throw error;
        return data as T;
      },
      expiration
    );
    
    return { data, error: null };
  } catch (error) {
    console.error('Cached query error:', error);
    return { data: null, error };
  }
}

/**
 * Create a cache key for Supabase queries
 */
export function createSupabaseQueryKey(table: string, query: string, params?: any): string {
  return createCacheKey('supabase', table, query, params ? JSON.stringify(params) : '');
}