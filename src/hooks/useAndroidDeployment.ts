import { useState, useCallback } from 'react';
import { supabase, cachedQuery, createSupabaseQueryKey } from '../lib/supabase';
import { CACHE_EXPIRATION } from '../utils/cacheUtils';

interface DeploymentConfig {
  versionName: string;
  versionCode: number;
  buildType: 'debug' | 'release';
  outputType: 'apk' | 'aab';
  track?: 'internal' | 'alpha' | 'beta' | 'production';
}

interface DeploymentStatus {
  id: number;
  status: 'pending' | 'building' | 'signing' | 'uploading' | 'completed' | 'failed';
  progress_percentage?: number;
  current_step?: string;
  buildLogs?: string;
  errorMessage?: string;
  filePath?: string;
  fileSize?: number;
  startedAt: string;
  completedAt?: string;
}

interface UseAndroidDeploymentReturn {
  createDeployment: (config: DeploymentConfig) => Promise<{ id: number }>;
  updateDeployment: (id: number, status: Partial<DeploymentStatus>) => Promise<void>;
  getDeployment: (id: number) => Promise<DeploymentStatus>;
  getDeployments: () => Promise<DeploymentStatus[]>;
  loading: boolean;
  error: string | null;
  updateProgress: (id: number, progress: number, step: string, message?: string) => Promise<void>;
}

export const useAndroidDeployment = (): UseAndroidDeploymentReturn => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const callDeploymentFunction = async (action: string, data: any) => {
    setLoading(true);
    setError(null);
    
    try {
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        throw new Error('Not authenticated');
      }
      
      const response = await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/android-deployment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          action,
          ...data,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to call deployment function');
      }
      
      return await response.json();
    } catch (err: any) {
      setError(err.message || 'An error occurred');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const createDeployment = useCallback(async (config: DeploymentConfig) => {
    const result = await callDeploymentFunction('create', config);
    return { id: result.deployment.id };
  }, []);

  const updateDeployment = useCallback(async (deploymentId: number, status: Partial<DeploymentStatus>) => {
    await callDeploymentFunction('update', {
      deploymentId,
      ...status,
    });
  }, []);

  const getDeployment = useCallback(async (deploymentId: number) => {
    const result = await callDeploymentFunction('get', { deploymentId });
    return result.deployment;
  }, []);

  const getDeployments = useCallback(async () => {
    // Use cached query for deployments list
    const cacheKey = createSupabaseQueryKey('android_deployments', 'list');
    
    try {
      const { data, error } = await cachedQuery(
        () => callDeploymentFunction('get', {}),
        cacheKey,
        CACHE_EXPIRATION.MEDIUM
      );
      
      if (error) throw error;
      return data.deployments;
    } catch (error) {
      console.error('Failed to get deployments:', error);
      return [];
    }
  }, []);

  const updateProgress = useCallback(async (deploymentId: number, progress: number, step: string, message?: string) => {
    try {
      // Update the deployment progress
      await callDeploymentFunction('update', {
        deploymentId,
        progress_percentage: progress,
        current_step: step
      });
      
      // Also create a progress entry for detailed tracking
      await callDeploymentFunction('progress', {
        deploymentId,
        step,
        progress,
        message
      });
    } catch (err) {
      console.error('Failed to update progress:', err);
      setError(err instanceof Error ? err.message : 'Failed to update progress');
    }
  }, []);

  return {
    createDeployment,
    updateDeployment,
    getDeployment,
    getDeployments,
    loading,
    error,
    updateProgress
  };
};