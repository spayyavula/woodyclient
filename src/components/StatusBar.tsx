import React from 'react';
import { GitBranch, Zap, Circle, CheckCircle, AlertCircle, Users, Wifi } from 'lucide-react';

interface StatusBarProps {
  currentFile: string;
  language: string;
  lineCount: number;
  currentLine: number;
  buildStatus: 'idle' | 'building' | 'success' | 'error';
  collaboratorCount?: number;
}

const StatusBar: React.FC<StatusBarProps> = ({ 
  currentFile, 
  language, 
  lineCount, 
  currentLine, 
  buildStatus,
  collaboratorCount = 0
}) => {
  const getStatusIcon = () => {
    switch (buildStatus) {
      case 'building':
        return <Circle className="w-3 h-3 text-yellow-400 animate-pulse" />;
      case 'success':
        return <CheckCircle className="w-3 h-3 text-green-400" />;
      case 'error':
        return <AlertCircle className="w-3 h-3 text-red-400" />;
      default:
        return <Circle className="w-3 h-3 text-gray-500" />;
    }
  };

  const getStatusText = () => {
    switch (buildStatus) {
      case 'building':
        return 'Building...';
      case 'success':
        return 'Build succeeded';
      case 'error':
        return 'Build failed';
      default:
        return 'Ready';
    }
  };

  return (
    <div className="bg-blue-600 text-white px-4 py-1 flex items-center justify-between text-xs">
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          {getStatusIcon()}
          <span>{getStatusText()}</span>
        </div>
        
        <div className="flex items-center space-x-1">
          <GitBranch className="w-3 h-3" />
          <span>main</span>
        </div>

        <div className="flex items-center space-x-1">
          <Zap className="w-3 h-3 text-orange-300" />
          <span>Rust {language === 'rust' ? '1.75.0' : language === 'dart' ? 'Flutter 3.16' : language === 'kotlin' ? 'Kotlin 1.9' : ''}</span>
        </div>
      </div>

      <div className="flex items-center space-x-6">
        <span className="flex items-center space-x-2">
          {language === 'rust' && <span className="w-2 h-2 bg-orange-500 rounded-full"></span>}
          {language === 'dart' && <span className="w-2 h-2 bg-cyan-500 rounded-full"></span>}
          {language === 'kotlin' && <span className="w-2 h-2 bg-purple-500 rounded-full"></span>}
          <span>{currentFile || 'No file selected'}</span>
        </span>
        <span>Ln {currentLine}, Col 1</span>
        <span>{lineCount} lines</span>
        <span>UTF-8</span>
        <span>Spaces: 4</span>
        <span className="flex items-center space-x-1">
          <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
          <span>Mobile Ready</span>
        </span>
        
        {collaboratorCount > 0 && (
          <span className="flex items-center space-x-1">
            <Users className="w-3 h-3 text-blue-400" />
            <span>{collaboratorCount} online</span>
          </span>
        )}
        
        <span className="flex items-center space-x-1">
          <Wifi className="w-3 h-3 text-green-400" />
          <span>Real-time sync</span>
        </span>
      </div>
    </div>
  );
};

export default StatusBar;