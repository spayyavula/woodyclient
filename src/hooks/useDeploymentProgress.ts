import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../lib/supabase';

interface DeploymentDetails {
  id: number;
  visual_progress: number;
  progress_message: string;
  status: string;
}

interface ProgressEvent {
  id: number;
  deployment_id: number;
  event_type: 'start' | 'progress' | 'success' | 'error' | 'warning' | 'info';
  message: string;
  percentage?: number;
  timestamp: string;
  metadata?: any;
}

interface DeploymentProgress {
  id: number;
  visual_progress: number;
  progress_message: string;
  status: string;
  events: ProgressEvent[];
  isLoading: boolean;
  error: string | null;
}

export const useDeploymentProgress = (deploymentId?: number) => {
  const [progress, setProgress] = useState<DeploymentProgress>({
    id: deploymentId || 0,
    visual_progress: 0,
    progress_message: '',
    status: 'pending',
    events: [],
    isLoading: false,
    error: null
  });

  const fetchProgress = useCallback(async () => {
    if (!deploymentId) return;

    let retryCount = 0;
    const maxRetries = 3;

    try {
      setProgress(prev => ({ ...prev, isLoading: true, error: null }));

      // Retry logic for fetching deployment details
      let deploymentData = null;
      let deploymentError = null;
      
      while (retryCount < maxRetries && !deploymentData) {
        const result = await supabase
          .from('android_deployments')
          .select('id, visual_progress, progress_message, status')
          .eq('id', deploymentId)
          .single();
        
        deploymentData = result.data;
        deploymentError = result.error;
        
        if (deploymentError && retryCount < maxRetries - 1) {
          // Wait before retrying (exponential backoff)
          await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, retryCount)));
          retryCount++;
        } else {
          break;
        }
      }

      if (deploymentError) {
        console.warn('Error fetching deployment:', deploymentError);
        // Don't throw here, just continue with default values
      }
      
      // Ensure we have valid deployment data
      const deployment: DeploymentDetails = deploymentData || {
        id: deploymentId,
        visual_progress: 0,
        progress_message: '',
        status: 'pending'
      };

      // Fetch progress events
      const { data: events, error: eventsError } = await supabase
        .from('deployment_progress_events')
        .select('id, deployment_id, event_type, message, percentage, timestamp, metadata')
        .eq('deployment_id', deploymentId)
        .order('timestamp', { ascending: true });

      if (eventsError) {
        console.warn('Error fetching events:', eventsError);
        // Don't throw here, just continue with empty events
      }

      setProgress({
        id: deployment.id,
        visual_progress: deployment.visual_progress ?? 0,
        progress_message: deployment.progress_message ?? '',
        status: deployment.status ?? 'pending',
        events: events ?? [],
        isLoading: false,
        error: null
      });
    } catch (error) {
      console.error('Error fetching deployment progress:', error);
      setProgress(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch progress'
      }));
    }
  }, [deploymentId]);

  const addProgressEvent = useCallback(async (
    event_type: ProgressEvent['event_type'],
    message: string,
    percentage?: number,
    metadata?: any
  ) => {
    if (!deploymentId) return;

    try {
      const { data, error } = await supabase
        .from('deployment_progress_events')
        .insert({
          deployment_id: deploymentId,
          event_type,
          message,
          percentage,
          metadata
        })
        .select()
        .single();

      if (error) throw error;

      // Update local state
      setProgress(prev => ({
        ...prev,
        visual_progress: percentage || prev.visual_progress,
        progress_message: message,
        events: [...prev.events, data]
      }));

      return data;
    } catch (error) {
      console.error('Error adding progress event:', error);
      return null;
    }
  }, [deploymentId]);

  // Set up real-time subscription
  useEffect(() => {
    if (!deploymentId) return;
    
    // Initial fetch
    fetchProgress();
    
    // Set up polling for progress updates
    const pollingInterval = setInterval(() => {
      fetchProgress();
    }, 3000); // Poll every 3 seconds

    return () => {
      clearInterval(pollingInterval);
    };
  }, [deploymentId, fetchProgress]);

  return {
    progress: progress.visual_progress,
    message: progress.progress_message,
    status: progress.status,
    events: progress.events,
    isLoading: progress.isLoading,
    error: progress.error,
    addProgressEvent,
    refreshProgress: fetchProgress
  };
};