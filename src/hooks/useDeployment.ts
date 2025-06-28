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