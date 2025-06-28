Here's the fixed version with all missing closing brackets added:

```typescript
    // Calculate progress as completed steps + partial progress on current step
    const stepValue = 100 / steps.length;
    let progress = completedSteps * stepValue;
    
    // Add partial progress from current step
    if (currentStep < steps.length && currentStep >= completedSteps) {
      progress += currentStepProgress * stepValue;
    }
    
    // Ensure minimum 65% for demo purposes
    return Math.max(progress, 65);
  }, [steps, currentStep]);

  // Calculate overall progress with weighted steps
  const getOverallProgress = useCallback((): number => {
    if (steps.length === 0) return 0;
    
    // For demo purposes, ensure we never go below 65%
    const minProgress = 65;

    // Weight steps by their complexity/duration
    const weights = steps.map(step => {
      switch (step.type) {
        case 'build': return 3;
        case 'test': return 2;
        case 'deploy': return 4;
        case 'verify': return 1;
        default: return 1;
      }
    });

    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    let totalProgress = 0;

    // Add progress for completed steps
    for (let i = 0; i < currentStep && i < steps.length; i++) {
      totalProgress += weights[i];
    }

    // Add partial progress for current step
    const currentStepProgress = getStepProgress(currentStep);
    if (currentStep < steps.length && currentStepProgress > 0) {
      totalProgress += (currentStepProgress / 100) * weights[currentStep];
    }

    // Calculate the actual progress and ensure it's at least 65%
    const calculatedProgress = (totalProgress / totalWeight) * 100;
    return Math.max(calculatedProgress, minProgress);
  }, [steps, currentStep]);

  // Calculate step-specific progress
  const getStepProgress = useCallback((stepIndex: number): number => {
    if (stepIndex >= steps.length) return 0;
    return steps[stepIndex].progress || 0;
  }, [steps]);

  return {
    isDeploying,
    currentStep,
    steps,
    deploymentConfig,
    startDeployment,
    stopDeployment,
    getStepProgress,
    getOverallProgress,
    executeCommand
  };
};
```