import React, { useState, useEffect } from 'react';
import { 
  Rocket, 
  X, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  Clock,
  ExternalLink,
  ArrowRight,
  Smartphone,
  Globe,
  Server
} from 'lucide-react';
import DeploymentStatus from './DeploymentStatus';

interface DeploymentStatusPanelProps {
  isVisible: boolean;
  onClose: () => void;
  deployUrl?: string;
  deployId?: string;
}

const DeploymentStatusPanel: React.FC<DeploymentStatusPanelProps> = ({
  isVisible,
  onClose,
  deployUrl,
  deployId
}) => {
  const [deploymentStatus, setDeploymentStatus] = useState<'success' | 'error' | 'in_progress' | 'idle'>(
    deployUrl ? 'success' : 'idle'
  );
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  
  useEffect(() => {
    if (deployUrl) {
      setDeploymentStatus('success');
      setLastUpdated(new Date());
    }
  }, [deployUrl]);
  
  const refreshStatus = async () => {
    setIsRefreshing(true);
    
    try {
      // In a real app, you would fetch the deployment status from your API
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // For demo purposes, we'll just set it to success if we have a deployUrl
      if (deployUrl) {
        setDeploymentStatus('success');
      }
      
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error refreshing deployment status:', error);
    } finally {
      setIsRefreshing(false);
    }
  };
  
  const startNewDeployment = () => {
    setDeploymentStatus('in_progress');
    
    // Simulate a deployment process
    setTimeout(() => {
      setDeploymentStatus('success');
      setLastUpdated(new Date());
    }, 10000);
  };
  
  if (!isVisible) return null;
  
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-xl border border-gray-700 w-full max-w-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-500 rounded-xl shadow-lg">
              <Rocket className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Deployment Status</h2>
              <p className="text-gray-400">Monitor your application deployments</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>
        
        <div className="p-6">
          <div className="space-y-6">
            {/* Last Updated */}
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-400">
                {lastUpdated ? (
                  <>Last updated: {lastUpdated.toLocaleTimeString()}</>
                ) : (
                  <>Status not checked yet</>
                )}
              </div>
              <button
                onClick={refreshStatus}
                disabled={isRefreshing || deploymentStatus === 'in_progress'}
                className="flex items-center space-x-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-white rounded-lg text-sm transition-colors"
              >
                <RefreshCw className={`w-3 h-3 ${isRefreshing ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>
            </div>
            
            {/* Web Deployment Status */}
            <DeploymentStatus
              deployUrl={deployUrl}
              deployId={deployId}
              status={deploymentStatus}
              platform="web"
              onRefresh={refreshStatus}
            />
            
            {/* Android Deployment Status */}
            <DeploymentStatus
              status="idle"
              platform="android"
            />
            
            {/* iOS Deployment Status */}
            <DeploymentStatus
              status="idle"
              platform="ios"
            />
            
            {/* Help Text */}
            <div className="text-sm text-gray-400 mt-4">
              <p>Need help with deployments? Check out our <a href="#" className="text-blue-400 hover:text-blue-300">deployment documentation</a> for detailed instructions.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeploymentStatusPanel;