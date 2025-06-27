import React from 'react';
import { 
  Play, 
  Square, 
  Save, 
  Folder, 
  FileText, 
  Search, 
  Settings, 
  Terminal,
  GitBranch,
  Bug,
  Smartphone,
  Monitor,
  Tablet,
  Users,
  ShoppingCart
} from 'lucide-react';

interface ToolbarProps {
  onToggleTerminal: () => void;
  onToggleCollaboration: () => void;
  onToggleMarketplace: () => void;
  onSave: () => void;
  onRun: () => void;
  onShowTemplates: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({ 
  onToggleTerminal, 
  onToggleCollaboration, 
  onToggleMarketplace,
  onSave, 
  onRun, 
  onShowTemplates 
}) => {
  const [selectedPlatform, setSelectedPlatform] = React.useState('android');

  return (
    <div className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center justify-between">
      <div className="flex items-center space-x-1">
        <div className="flex items-center space-x-1 mr-4">
          <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
            <Folder className="w-4 h-4" />
          </button>
          <button 
            onClick={onShowTemplates}
            className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors"
            title="Project Templates"
          >
            <FileText className="w-4 h-4" />
          </button>
          <button 
            onClick={onSave}
            className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors"
          >
            <Save className="w-4 h-4" />
          </button>
        </div>
        
        <div className="w-px h-6 bg-gray-600 mx-2" />
        
        {/* Platform Selection */}
        <div className="flex items-center space-x-1 mr-4">
          <select 
            value={selectedPlatform}
            onChange={(e) => setSelectedPlatform(e.target.value)}
            className="bg-gray-700 text-gray-200 text-sm px-3 py-1.5 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
          >
            <option value="android">Android</option>
            <option value="ios">iOS</option>
            <option value="flutter">Flutter</option>
            <option value="desktop">Desktop</option>
          </select>
          <div className="flex items-center space-x-1">
            {selectedPlatform === 'android' && <Smartphone className="w-4 h-4 text-green-400" />}
            {selectedPlatform === 'ios' && <Smartphone className="w-4 h-4 text-blue-400" />}
            {selectedPlatform === 'flutter' && <Tablet className="w-4 h-4 text-cyan-400" />}
            {selectedPlatform === 'desktop' && <Monitor className="w-4 h-4 text-purple-400" />}
          </div>
        </div>
        
        <div className="w-px h-6 bg-gray-600 mx-2" />
        
        <div className="flex items-center space-x-1">
          <button 
            onClick={onRun}
            className="flex items-center px-3 py-1.5 bg-orange-600 hover:bg-orange-700 rounded text-white text-sm font-medium transition-colors"
          >
            <Play className="w-4 h-4 mr-1" />
            Build & Run
          </button>
          <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
            <Square className="w-4 h-4" />
          </button>
          <button className="flex items-center px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm font-medium transition-colors">
            <Smartphone className="w-4 h-4 mr-1" />
            Deploy
          </button>
          <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
            <Bug className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex items-center space-x-1">
        <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
          <Search className="w-4 h-4" />
        </button>
        <button 
          onClick={onToggleCollaboration}
          className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors relative"
          title="Collaboration"
        >
          <Users className="w-4 h-4" />
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse" />
        </button>
        <button 
          onClick={onToggleMarketplace}
          className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors relative"
          title="Developer Marketplace"
        >
          <ShoppingCart className="w-4 h-4" />
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-orange-400 rounded-full animate-pulse" />
        </button>
        <button 
          onClick={onToggleTerminal}
          className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors"
        >
          <Terminal className="w-4 h-4" />
        </button>
        <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
          <GitBranch className="w-4 h-4" />
        </button>
        <button className="p-2 hover:bg-gray-700 rounded text-gray-300 hover:text-white transition-colors">
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default Toolbar;