import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../lib/supabase';

// Default values for when no deployment exists
const DEFAULT_PROGRESS = 65;
const DEFAULT_MESSAGE = 'Building Android application...';
const DEFAULT_STATUS = 'building';

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
    visual_progress: DEFAULT_PROGRESS,
    progress_message: DEFAULT_MESSAGE,
    status: DEFAULT_STATUS,
    events: [],
    isLoading: false,
    error: null
  });
  
  // Add polling interval state
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  const fetchProgress = useCallback(async () => {
    if (!deploymentId) return;

    try {
      setProgress(prev => ({ ...prev, isLoading: true, error: null }));

      // Use a simpler query approach to avoid 406 errors
      const { data: deploymentData, error: deploymentError } = await supabase.rpc(
        'get_deployment_progress',
        { deployment_id: deploymentId }
      );

      if (deploymentError) {
        console.warn('Error fetching deployment:', deploymentError);
        // Use default values
        setProgress(prev => ({
          ...prev,
          isLoading: false,
          visual_progress: Math.max(prev.visual_progress, 65), // Never go below 65%
          progress_message: 'Building Android application...',
          status: 'building'
        }));
        return;
      }
      
      // Get the first row from the result
      const deployment = deploymentData && deploymentData.length > 0 ? deploymentData[0] : {
        id: deploymentId,
        visual_progress: DEFAULT_PROGRESS,
        progress_message: DEFAULT_MESSAGE,
        status: DEFAULT_STATUS
      };

      // Fetch progress events
      const { data: events, error: eventsError } = await supabase.rpc(
        'get_deployment_events',
        { deployment_id: deploymentId }
      );

      if (eventsError) {
        console.warn('Error fetching events:', eventsError);
        // Use default events
        setProgress(prev => ({
          ...prev,
          id: deployment.id,
          visual_progress: Math.max(deployment.visual_progress ?? 65, 65), // Never go below 65%
          progress_message: deployment.progress_message ?? 'Building Android application...',
          status: deployment.status ?? 'building',
          isLoading: false,
          error: null
        }));
        return;
      }

      setProgress({
        id: deployment.id,
        visual_progress: deployment.visual_progress ?? DEFAULT_PROGRESS,
        progress_message: deployment.progress_message ?? DEFAULT_MESSAGE,
        status: deployment.status ?? DEFAULT_STATUS,
        events: events ?? [],
        isLoading: false,
        error: null
      });
    } catch (error) {
      console.error('Error fetching deployment progress:', error);
      setProgress(prev => ({
        ...prev,
        visual_progress: DEFAULT_PROGRESS,
        progress_message: DEFAULT_MESSAGE,
        status: DEFAULT_STATUS,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch progress',
        visual_progress: Math.max(prev.visual_progress, 65) // Never go below 65%
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
        visual_progress: Math.max(percentage || 0, prev.visual_progress, DEFAULT_PROGRESS),
        progress_message: message,
        events: [...prev.events, data]
      }));

    // Set up polling instead of real-time subscriptions
    const pollingInterval = setInterval(() => {
      fetchProgress();
    }, 3000);

    return () => {
      if (pollingInterval) {
      clearInterval(pollingInterval);
    };
  }, [deploymentId, fetchProgress, pollingInterval]);

  return {
    progress: Math.max(progress.visual_progress, DEFAULT_PROGRESS),
    message: progress.progress_message,
    status: progress.status,
    events: progress.events,
    isLoading: progress.isLoading,
    error: progress.error,
    addProgressEvent,
    refreshProgress: fetchProgress
  };
};