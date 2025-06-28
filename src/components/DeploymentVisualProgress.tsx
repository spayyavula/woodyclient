import React, { useState, useEffect } from 'react';
import { CheckCircle, XCircle, AlertTriangle, Clock, RefreshCw } from 'lucide-react';

interface DeploymentVisualProgressProps {
  progress: number;
  status?: string;
  message?: string;
  platform?: string;
  className?: string;
  showDetails?: boolean;
  animate?: boolean;
}

const DeploymentVisualProgress: React.FC<DeploymentVisualProgressProps> = ({
  progress,
  status = 'building',
  message,
  platform = 'android',
  className = '',
  showDetails = true,
  animate = true
}) => {
  // Start with at least 65% progress for demo purposes
  const [animatedProgress, setAnimatedProgress] = useState(Math.max(progress, 65));
  // Ensure progress is at least 65%
  const safeProgress = Math.max(progress, 65);

  useEffect(() => {
    if (!animate) {
      setAnimatedProgress(Math.max(progress, 65));
      return;
    }
    
    // Animate progress smoothly
    const start = animatedProgress;
    const duration = 500;
    const startTime = performance.now();

    const animateProgress = (timestamp: number) => {
      const elapsed = timestamp - startTime;
      const nextProgress = Math.min(start + ((safeProgress - start) * elapsed) / duration, safeProgress);
      
      setAnimatedProgress(nextProgress);
      
      if (elapsed < duration && nextProgress < safeProgress) {
        requestAnimationFrame(animateProgress);
      } else {
        setAnimatedProgress(safeProgress);
      }
    };

    requestAnimationFrame(animateProgress);
  }, [safeProgress, animate, animatedProgress]);