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
    visual_progress: 65,
    progress_message: 'Building Android application...',
    status: 'building',
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

      // Use the RPC function to get deployment progress with fallback
      const { data: deploymentData, error: deploymentError } = await supabase
        .rpc('get_deployment_progress', { p_deployment_id: deploymentId })
        .single();
        .single();

      if (deploymentError) {
        console.warn('Error fetching deployment:', deploymentError);
        // Use default values
        deploymentData = {
          id: deploymentId,
          visual_progress: 65,
          progress_message: 'Building Android application...',
          status: 'building'
        };
      }
      
      // Ensure we have valid deployment data
      const deployment: DeploymentDetails = deploymentData || {
        id: deploymentId,
        visual_progress: 65,
        progress_message: 'Building Android application...',
        status: 'building'
      };

      // Use the RPC function to get events with fallback
      const { data: events, error: eventsError } = await supabase
        .rpc('get_deployment_events', { p_deployment_id: deploymentId });
        // Use default event
        events = [{
          id: 1,
          deployment_id: deploymentId,
          event_type: 'progress',
          message: 'Building Android application...',
          percentage: 65,
          event_timestamp: new Date().toISOString(),
          metadata: null
        }];
      }

      // Map event_timestamp back to timestamp for consistency
      const mappedEvents = events ? events.map(e => ({
        ...e,
        timestamp: e.event_timestamp,
      })) : [];

      setProgress({
        id: deployment.id,
        visual_progress: Math.max(deployment.visual_progress ?? 0, 65),
        progress_message: deployment.progress_message ?? 'Building Android application...',
        status: deployment.status ?? 'building',
        events: mappedEvents,
        isLoading: false,
        error: null
      });
    } catch (error) {
      console.error('Error fetching deployment progress:', error);
      setProgress(prev => ({
        ...prev,
        visual_progress: 65,
        progress_message: 'Building Android application...',
        status: 'building',
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch progress',
        visual_progress: 65,
        progress_message: 'Building Android application...',
        status: 'building'
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

        visual_progress: Math.max(percentage || prev.visual_progress, 65),
      setProgress(prev => ({
        ...prev,
        visual_progress: Math.max(percentage || prev.visual_progress, 65),
        progress_message: message,
        events: [...prev.events, data]
      }));
    }

    // Set up polling instead of real-time subscriptions
    const pollingInterval = setInterval(() => {
      fetchProgress();
    }, 3000);

    return () => {
      if (pollingInterval) {
      }
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
          visual_progress: Math.max(payload.new.visual_progress || 0, 65),
          progress_message: payload.new.progress_message || 'Building Android application...',