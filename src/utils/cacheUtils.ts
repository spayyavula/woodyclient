import localforage from 'localforage';

// Configure localforage
localforage.config({
  name: 'rustyclint',
  storeName: 'cache',
  description: 'Cache for rustyclint application'
});

// Cache expiration times (in milliseconds)
export const CACHE_EXPIRATION = {
  SHORT: 5 * 60 * 1000, // 5 minutes
  MEDIUM: 30 * 60 * 1000, // 30 minutes
  LONG: 24 * 60 * 60 * 1000, // 1 day
  VERY_LONG: 7 * 24 * 60 * 60 * 1000 // 1 week
};

// Cache item with expiration
interface CacheItem<T> {
  data: T;
  expiry: number;
}

/**
 * Set an item in the cache with expiration
 */
export async function setCache<T>(key: string, data: T, expiration = CACHE_EXPIRATION.MEDIUM): Promise<void> {
  const item: CacheItem<T> = {
    data,
    expiry: Date.now() + expiration
  };
  
  try {
    await localforage.setItem(key, item);
  } catch (error) {
    console.error('Cache set error:', error);
  }
}

/**
 * Get an item from the cache, returns null if expired or not found
 */
export async function getCache<T>(key: string): Promise<T | null> {
  try {
    const item = await localforage.getItem<CacheItem<T>>(key);
    
    // Return null if item doesn't exist or is expired
    if (!item || Date.now() > item.expiry) {
      return null;
    }
    
    return item.data;
  } catch (error) {
    console.error('Cache get error:', error);
    return null;
  }
}

/**
 * Remove an item from the cache
 */
export async function removeCache(key: string): Promise<void> {
  try {
    await localforage.removeItem(key);
  } catch (error) {
    console.error('Cache remove error:', error);
  }
}

/**
 * Clear all items from the cache
 */
export async function clearCache(): Promise<void> {
  try {
    await localforage.clear();
  } catch (error) {
    console.error('Cache clear error:', error);
  }
}

/**
 * Get data with caching
 * If data is in cache and not expired, returns cached data
 * Otherwise, fetches data using fetchFn and caches it
 */
export async function getCachedData<T>(
  key: string, 
  fetchFn: () => Promise<T>, 
  expiration = CACHE_EXPIRATION.MEDIUM
): Promise<T> {
  // Try to get from cache first
  const cachedData = await getCache<T>(key);
  
  if (cachedData !== null) {
    return cachedData;
  }
  
  // If not in cache or expired, fetch fresh data
  const freshData = await fetchFn();
  
  // Cache the fresh data
  await setCache(key, freshData, expiration);
  
  return freshData;
}

/**
 * Prefetch and cache data
 */
export async function prefetchData<T>(
  key: string, 
  fetchFn: () => Promise<T>, 
  expiration = CACHE_EXPIRATION.MEDIUM
): Promise<void> {
  try {
    const data = await fetchFn();
    await setCache(key, data, expiration);
  } catch (error) {
    console.error('Prefetch error:', error);
  }
}

/**
 * Generate a cache key with namespace
 */
export function createCacheKey(namespace: string, ...parts: (string | number)[]): string {
  return `${namespace}:${parts.join(':')}`;
}