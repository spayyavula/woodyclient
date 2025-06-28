import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../lib/supabase';

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

    setProgress(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      // Fetch deployment details
      const { data: deployment, error: deploymentError } = await supabase
        .from('android_deployments')
        .select('id, visual_progress, progress_message, status')
        .eq('id', deploymentId)
        .single();

      if (deploymentError) throw deploymentError;

      // Fetch progress events
      const { data: events, error: eventsError } = await supabase
        .from('deployment_progress_events')
        .select('*')
        .eq('deployment_id', deploymentId)
        .order('timestamp', { ascending: true });

      if (eventsError) throw eventsError;

      setProgress({
        id: deploymentId,
        visual_progress: deployment.visual_progress || 0,
        progress_message: deployment.progress_message || '',
        status: deployment.status,
        events: events || [],
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

    // Subscribe to deployment updates
    const deploymentSubscription = supabase
      .channel(`deployment-${deploymentId}`)
      .on('postgres_changes', {
        event: 'UPDATE',
        schema: 'public',
        table: 'android_deployments',
        filter: `id=eq.${deploymentId}`
      }, (payload) => {
        setProgress(prev => ({
          ...prev,
          visual_progress: payload.new.visual_progress || 0,
          progress_message: payload.new.progress_message || '',
          status: payload.new.status
        }));
      })
      .subscribe();

    // Subscribe to progress events
    const eventsSubscription = supabase
      .channel(`progress-events-${deploymentId}`)
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'deployment_progress_events',
        filter: `deployment_id=eq.${deploymentId}`
      }, (payload) => {
        setProgress(prev => ({
          ...prev,
          events: [...prev.events, payload.new as ProgressEvent]
        }));
      })
      .subscribe();

    return () => {
      deploymentSubscription.unsubscribe();
      eventsSubscription.unsubscribe();
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